"""
src/models/train.py — Multi-Horizon Model Training 
======================================================
Her tahmin ufku (2h, 4h, 6h, 8h) için ayrı LightGBM modeli.

  - MedianPruner ile erken durdurma
  - Exception-safe study optimize
  - n_jobs over-subscription fix
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from src.utils.io_helper import save_model
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

FORECAST_HORIZONS = [2, 4, 6, 8]


# Konfigürasyon

cfg         = load_config()
paths_cfg   = cfg.get("paths", {})
mlflow_cfg  = cfg.get("mlflow", {})
MODEL_DIR   = paths_cfg.get("models_dir")
GOLD_PATH   = paths_cfg.get("feature_gold_dir")
REPORTS_DIR = paths_cfg.get("reports_dir", "reports")

mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
mlflow.set_experiment(mlflow_cfg.get("experiment_name", "Blast_Furnace_Silicon_Prediction"))

_NON_FEATURE_COLS = [
    "si_dt", "next_si_dt", "hours_to_next_cast", "is_quarantine", "is_startup",
    "delta_target", "future_dt", "future_Si", "prediction_horizon_hours",
] + [f"target_Si_{h}h" for h in FORECAST_HORIZONS]


def _load_and_prepare(gold_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(gold_path)
    df["si_dt"] = pd.to_datetime(df["si_dt"])
    df = df.sort_values("si_dt").reset_index(drop=True)
    logger.info(f"Gold satır: {len(df)} | Kolon: {len(df.columns)}")

    if "hours_to_next_cast" in df.columns:
        gap_indices = df[df["hours_to_next_cast"] > 8].index
        df["is_startup"]    = 0
        df["is_quarantine"] = False
        for idx in gap_indices:
            gap_end = df.loc[idx, "next_si_dt"] if "next_si_dt" in df.columns else df.loc[idx, "si_dt"]
            q_mask = (df["si_dt"] >= gap_end) & (df["si_dt"] <= gap_end + pd.Timedelta(hours=12))
            s_mask = (df["si_dt"] >= gap_end) & (df["si_dt"] <= gap_end + pd.Timedelta(hours=24))
            df.loc[q_mask, "is_quarantine"] = True
            df.loc[s_mask, "is_startup"]    = 1
        df = df[~df["is_quarantine"]].copy()
        logger.info(f"Karantina sonrası satır: {len(df)}")

    drop_cols = [c for c in _NON_FEATURE_COLS if c in df.columns]
    target_cols = [f"target_Si_{h}h" for h in FORECAST_HORIZONS if f"target_Si_{h}h" in df.columns]

    targets = df[target_cols].copy()
    X_base  = df.drop(columns=drop_cols, errors="ignore")

    logger.info(f"Feature sayısı: {len(X_base.columns)} | Hedef kolonları: {target_cols}")
    return X_base, targets


# ---------------------------------------------------------------------------
# Optuna Objective
# ---------------------------------------------------------------------------

def _make_objective(X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit, horizon_h: int, n_jobs_lgb: int):
    def objective(trial):
        lr_upper  = 0.10 if horizon_h <= 2 else 0.06
        depth_max = 18   if horizon_h <= 2 else 14

        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 500, 3000),
            "max_depth":         trial.suggest_int("max_depth", 6, depth_max),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, lr_upper, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 31, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective":         "regression",
            "metric":            "rmse",
            "random_state":      42,
            "verbose":           -1,
            "n_jobs":            n_jobs_lgb,
        }

        cv_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            valid_tr  = y_tr.notna()
            valid_val = y_val.notna()
            if valid_tr.sum() < 50 or valid_val.sum() < 10:
                continue

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr[valid_tr], y_tr[valid_tr],
                eval_set=[(X_val[valid_val], y_val[valid_val])],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            preds = model.predict(X_val[valid_val])
            rmse_fold = np.sqrt(mean_squared_error(y_val[valid_val], preds))
            cv_scores.append(rmse_fold)

            # Pruner: fold sonuçları kötüyse erken kes
            trial.report(rmse_fold, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(cv_scores) if cv_scores else float("inf")

    return objective


# ---------------------------------------------------------------------------
# Callback & Diagnostics
# ---------------------------------------------------------------------------

def _make_progress_callback(horizon_h: int, n_trials: int):
    """Her 5 trial'da ve son trial'da net log basar."""
    def callback(study: optuna.Study, trial: optuna.Trial):
        if trial.number % 5 == 0 or trial.number == n_trials - 1:
            best_val = study.best_value if study.best_trial else float("inf")
            logger.info(
                f"[{horizon_h}h] Optuna {trial.number + 1:02d}/{n_trials} | "
                f"Bu trial RMSE: {trial.value:.4f} | En iyi RMSE: {best_val:.4f}"
            )
    return callback


def _save_diagnostics(y_test, preds, model, X, mae, r2, run_name: str):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp_path   = os.path.join(REPORTS_DIR, f"feature_importance_{run_name}.png")
    imp_csv    = os.path.join(REPORTS_DIR, f"feature_importance_{run_name}.csv")

    plt.figure(figsize=(12, 10))
    importance.head(30).plot(kind="barh")
    plt.title(f"Top 30 Feature Importance — {run_name} (R²: {r2:.4f})")
    plt.tight_layout()
    plt.savefig(imp_path, dpi=150)
    plt.close()
    importance.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(imp_csv, index=False)

    diag_path = os.path.join(REPORTS_DIR, f"model_diagnostics_{run_name}.png")
    residuals = y_test - preds
    n_show    = min(500, len(y_test))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].scatter(y_test, preds, alpha=0.5, s=10)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[0, 0].set(xlabel="Actual Si", ylabel="Predicted Si", title=f"Predicted vs Actual (R²={r2:.3f})")
    axes[1, 0].scatter(preds, residuals, alpha=0.5, s=10)
    axes[1, 0].axhline(0, color="r", linestyle="--")
    axes[1, 0].set(xlabel="Predicted Si", ylabel="Residuals", title="Residual Plot")
    axes[1, 1].plot(y_test.iloc[-n_show:].values, label="Actual",    alpha=0.7)
    axes[1, 1].plot(preds[-n_show:],               label="Predicted", alpha=0.7)
    axes[1, 1].set(xlabel="Time Index", ylabel="Si", title=f"Time Series (last {n_show} pts)")
    axes[1, 1].legend()
    axes[0, 1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 1].set(xlabel="Prediction Error", ylabel="Frequency", title=f"Error Dist. (MAE={mae:.4f})")
    plt.suptitle(run_name, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(diag_path, dpi=150, bbox_inches="tight")
    plt.close()

    return imp_path, imp_csv, diag_path


# ---------------------------------------------------------------------------
# Tek horizon eğitimi
# ---------------------------------------------------------------------------

def _train_single_horizon(
    horizon_h: int,
    X_base: pd.DataFrame,
    targets: pd.DataFrame,
    tscv: TimeSeriesSplit,
    n_trials: int,
) -> dict:
    target_col = f"target_Si_{horizon_h}h"
    run_name   = f"LGBM_Si_{horizon_h}h"
    model_file = f"bf_model_lgb_{horizon_h}h.joblib"

    if target_col not in targets.columns:
        logger.warning(f"{target_col} gold'da bulunamadı, atlanıyor.")
        return {}

    y_full = targets[target_col]
    valid_mask = y_full.notna()
    X = X_base[valid_mask].reset_index(drop=True)
    y = y_full[valid_mask].reset_index(drop=True)
    logger.info(f"[{horizon_h}h] Eğitim satırı: {len(X)}")

    if len(X) < 200:
        logger.warning(f"[{horizon_h}h] Yetersiz veri ({len(X)} satır), atlanıyor.")
        return {}

    n_jobs_lgb = max(1, (os.cpu_count() or 4) // 2)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="minimize", pruner=pruner)

    logger.info(f"[{horizon_h}h] Optuna: {n_trials} trial, {tscv.n_splits}-fold TS-CV, pruner=Median...")

    study.optimize(
        _make_objective(X, y, tscv, horizon_h, n_jobs_lgb),
        n_trials=n_trials,
        show_progress_bar=True,                
        catch=(Exception,),
    )

    if len(study.trials) == 0 or study.best_trial is None:
        logger.error(f"[{horizon_h}h] Hiç başarılı trial olmadı, model eğitilemedi.")
        return {}

    logger.info(f"[{horizon_h}h] En iyi CV RMSE: {study.best_value:.4f}")

    final_params = {
        **study.best_params,
        "objective":    "regression",
        "random_state": 42,
        "verbose":      -1,
        "n_jobs":       n_jobs_lgb,
    }

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("horizon_hours", horizon_h)

        model = lgb.LGBMRegressor(**final_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)

        si_col       = X_test["Si"] if "Si" in X_test.columns else pd.Series([y_test.mean()] * len(y_test))
        baseline_mae = mean_absolute_error(y_test, si_col)
        improvement  = (1 - mae / baseline_mae) * 100 if baseline_mae > 0 else 0.0

        mlflow.log_params(final_params)
        mlflow.log_metrics({
            "mae": mae, "rmse": rmse, "r2": r2,
            "baseline_mae": baseline_mae, "improvement_pct": improvement
        })

        imp_path, imp_csv, diag_path = _save_diagnostics(y_test, preds, model, X, mae, r2, run_name)
        for artifact in [imp_path, imp_csv, diag_path]:
            mlflow.log_artifact(artifact)

        if not MODEL_DIR:
            raise EnvironmentError("paths.yaml içinde 'models_dir' tanımlı değil!")
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, model_file)
        save_model(model, model_path)
        mlflow.log_artifact(model_path)

        logger.info(
            f"[{horizon_h}h] ✅ MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f} "
            f"| Baseline MAE: {baseline_mae:.4f} | İyileştirme: %{improvement:.2f}"
        )

    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "baseline_mae": baseline_mae, "improvement_pct": improvement,
        "model_path": model_path,
    }


# ---------------------------------------------------------------------------
# Ana Training Fonksiyonu
# ---------------------------------------------------------------------------

def run_training(
    n_trials: int = 50,
    n_splits: int = 5,
    horizons: list[int] | None = None,
) -> dict[int, dict]:
    if not GOLD_PATH or not os.path.exists(GOLD_PATH):
        raise FileNotFoundError(f"Gold verisi bulunamadı: {GOLD_PATH}")

    target_horizons = horizons or FORECAST_HORIZONS
    X_base, targets = _load_and_prepare(GOLD_PATH)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_metrics: dict[int, dict] = {}
    for h in target_horizons:
        logger.info(f"\n{'='*60}\n  HORIZON: {h} SAAT\n{'='*60}")
        metrics = _train_single_horizon(h, X_base, targets, tscv, n_trials)
        if metrics:
            all_metrics[h] = metrics

    logger.info("\n===== EĞİTİM ÖZETİ =====")
    for h, m in sorted(all_metrics.items()):
        logger.info(
            f"  {h:2d}h → MAE: {m['mae']:.4f} | RMSE: {m['rmse']:.4f} "
            f"| R²: {m['r2']:.4f} | İyileştirme: %{m['improvement_pct']:.2f}"
        )

    return all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blast Furnace Si — Multi-Horizon Training")
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--trials",   type=int, default=50)
    parser.add_argument("--splits",   type=int, default=5)
    args = parser.parse_args()

    results = run_training(n_trials=args.trials, n_splits=args.splits, horizons=args.horizons)
    print("\n--- SONUÇLAR ---")
    for h, m in sorted(results.items()):
        print(f"\n[{h}h]")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")