"""
analysis/shap_explainer.py — SHAP Feature Explainability
=========================================================
Kullanım:
  - Pipeline'dan: run_shap_analysis()         ← her training sonrası otomatik
  - Tekil tüm:    docker exec bf_orchestrator python -m src.analysis.shap_explainer
  - Tek horizon:  docker exec bf_orchestrator python -m src.analysis.shap_explainer --horizon 8

Çıktılar (reports/ altına, her horizon için ayrı):
  shap_importance_Xh.png   — bar plot (top features)
  shap_beeswarm_Xh.png     — beeswarm (her feature'ın etkisi ve yönü)
"""

import os
import argparse
import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

cfg         = load_config()
paths_cfg   = cfg.get("paths", {})
MODEL_DIR   = paths_cfg.get("models_dir",       "models")
GOLD_PATH   = paths_cfg.get("feature_gold_dir", "data/gold")
REPORTS_DIR = paths_cfg.get("reports_dir",      "reports")

FORECAST_HORIZONS = [2, 4, 6, 8]

# train.py ile senkron — feature olarak kullanılmayan kolonlar
_DROP_COLS = [
    "si_dt", "next_si_dt", "hours_to_next_cast", "is_quarantine",
    "is_startup", "delta_target", "future_dt", "future_Si",
    "prediction_horizon_hours", "is_anomalous",
] + [f"target_Si_{h}h" for h in FORECAST_HORIZONS]


def _run_single_horizon(horizon_h: int, df_base: pd.DataFrame, sample_n: int) -> None:
    """Tek bir horizon için SHAP analizi çalıştırır."""
    model_file = os.path.join(MODEL_DIR, f"bf_model_lgb_{horizon_h}h.joblib")

    if not os.path.exists(model_file):
        logger.warning(f"[{horizon_h}h] Model bulunamadı, atlanıyor: {model_file}")
        return

    logger.info(f"[{horizon_h}h] Model yükleniyor: {model_file}")
    model = joblib.load(model_file)

    # Feature hazırlığı — train.py ile aynı drop mantığı
    X = df_base.drop(columns=[c for c in _DROP_COLS if c in df_base.columns])

    # Model feature listesiyle senkronize et
    model_features = model.feature_name_
    for col in set(model_features) - set(X.columns):
        logger.warning(f"[{horizon_h}h] Eksik kolon, 0 ile dolduruluyor: {col}")
        X[col] = 0
    X = X[model_features]

    # Bu horizon için geçerli target'ı olan satırları tercih et
    target_col = f"target_Si_{horizon_h}h"
    if target_col in df_base.columns:
        valid_mask = df_base[target_col].notna()
        if valid_mask.sum() > 100:
            X = X[valid_mask.values]
            logger.info(f"[{horizon_h}h] Geçerli target satırı: {len(X)}")

    # Örneklem
    if len(X) > sample_n:
        X = X.sample(n=sample_n, random_state=42)
        logger.info(f"[{horizon_h}h] SHAP için {sample_n} satır örneklendi.")

    logger.info(f"[{horizon_h}h] SHAP TreeExplainer başlatılıyor...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Bar plot
    bar_path = os.path.join(REPORTS_DIR, f"shap_importance_{horizon_h}h.png")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {horizon_h}h Horizon")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[{horizon_h}h] Kaydedildi: {bar_path}")

    # Beeswarm
    bee_path = os.path.join(REPORTS_DIR, f"shap_beeswarm_{horizon_h}h.png")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Beeswarm — {horizon_h}h Horizon")
    plt.savefig(bee_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[{horizon_h}h] Kaydedildi: {bee_path}")


def run_shap_analysis(
    sample_n: int = 2000,
    horizons: list[int] | None = None,
) -> None:
    """
    SHAP analizi çalıştırır.

    Args:
        sample_n:  Hesaplama maliyetini düşürmek için örneklem sayısı.
        horizons:  Analiz edilecek ufuklar; None → tüm ufuklar [2, 4, 6, 8]
    """
    gold_full_path = os.path.join(GOLD_PATH, "feature_store") \
        if not GOLD_PATH.endswith("feature_store") else GOLD_PATH

    if not os.path.exists(gold_full_path):
        logger.error(f"Gold verisi bulunamadı: {gold_full_path}")
        return

    logger.info(f"Gold veri okunuyor: {gold_full_path}")
    df_base = pd.read_parquet(gold_full_path)

    target_horizons = horizons or FORECAST_HORIZONS
    for h in target_horizons:
        logger.info(f"\n{'='*50}\n  SHAP — {h}h Horizon\n{'='*50}")
        try:
            _run_single_horizon(h, df_base, sample_n)
        except Exception as e:
            logger.error(f"[{h}h] SHAP hatası: {e}")
            continue

    logger.info(f"SHAP analizi tamamlandı → {REPORTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP Explainability — Multi-Horizon")
    parser.add_argument("--horizon",  type=int, default=None,
                        help="Tek horizon analizi (örn. --horizon 8). Default: tümü.")
    parser.add_argument("--sample_n", type=int, default=2000,
                        help="SHAP örneklem sayısı. Default: 2000.")
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else None
    run_shap_analysis(sample_n=args.sample_n, horizons=horizons)