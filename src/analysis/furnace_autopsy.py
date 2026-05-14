"""
analysis/furnace_autopsy.py — Furnace Autopsy & Anomaly Analysis
=================================================================
Kullanım:
  - Pipeline'dan: run_autopsy()        ← her training sonrası otomatik
  - Tekil:        docker exec bf_orchestrator python -m src.analysis.furnace_autopsy


  - bf_model_lgb_Xh.joblib (4 model)
  - Feature importance her horizon için ayrı loglanır
  - DB export: gold_furnace_analysis tablosu aynı kalır

  
       Tüm feature'lar gold_refiner tarafından üretilip
       gold parquet'e yazıldığından burası direkt oradan okur.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import IsolationForest
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

cfg       = load_config()
paths_cfg = cfg.get("paths", {})

FORECAST_HORIZONS = [2, 4, 6, 8]

# target kolonları + metadata — feature olarak kullanılmaz
_DROP_COLS = [
    "si_dt", "next_si_dt", "hours_to_next_cast", "is_quarantine",
    "is_startup", "delta_target", "future_dt", "future_Si",
    "prediction_horizon_hours", "is_anomalous",
] + [f"target_Si_{h}h" for h in FORECAST_HORIZONS]


def run_autopsy() -> None:
    """
    Fırın otopsisi: anomali tespiti, duruş şoku analizi, görsel rapor.
    Feature'lar gold_refiner tarafından üretilmiş gold parquet'ten okunur.
    DB export ayrı try bloğunda — DB yoksa analiz yine de tamamlanır.
    """
    gold_path   = os.path.join(paths_cfg.get("feature_gold_dir", "/app/data/gold"), "feature_store")
    reports_dir = paths_cfg.get("reports_dir", "reports")
    models_dir  = paths_cfg.get("models_dir",  "models")

    os.makedirs(reports_dir, exist_ok=True)

    try:
        logger.info(f"Fırın Otopsisi Başlıyor... Veri: {gold_path}")

        # Gold parquet'te feature'lar gold_refiner tarafından zaten üretilmiş olarak gelir.
        # apply_blast_furnace_features çağrısına gerek yok.
        df = pd.read_parquet(gold_path)
        df = df.sort_values("si_dt").reset_index(drop=True)

        # --- Anomali Tespiti (IsolationForest) ---
        logger.info("Anomali tespiti yapılıyor...")
        iso_features = ["Si", "Th", "Tc", "Fb", "CO2", "H2", "gas_ratio", "thermal_efficiency"]
        iso_features = [f for f in iso_features if f in df.columns]

        iso_model          = IsolationForest(contamination=0.05, random_state=42)
        df["is_anomalous"] = iso_model.fit_predict(df[iso_features].fillna(0))

        total_count   = len(df)
        anomaly_count = (df["is_anomalous"] == -1).sum()
        health_pct    = (1 - anomaly_count / total_count) * 100

        logger.info(f"Toplam gözlem: {total_count} döküm")
        logger.info(f"Fırın sağlık durumu: %{health_pct:.2f} stabil")

        critical_si = df[df["Si"] > 1.0]
        logger.info(f"Kritik Si seviyesi (>1.0): {len(critical_si)} kez")

        # --- Duruş Şoku Analizi ---
        gap_indices = df[df["hours_to_next_cast"] > 50].index if "hours_to_next_cast" in df.columns else []

        for idx in gap_indices:
            pre_gap  = df.iloc[idx]
            post_gap = df.iloc[idx + 1] if idx + 1 < len(df) else None
            if post_gap is not None:
                si_delta = post_gap["Si"] - pre_gap["Si"]
                logger.info(
                    f"Olay: {pre_gap['si_dt']} | "
                    f"Duruş: {pre_gap['hours_to_next_cast']:.1f}h | "
                    f"Si delta: {si_delta:+.2f} ({pre_gap['Si']:.2f}→{post_gap['Si']:.2f})"
                )

        # --- Model Feature Importance (tüm horizon'lar) ---
        X_cols = [c for c in df.columns if c not in _DROP_COLS]

        for h in FORECAST_HORIZONS:
            model_file = os.path.join(models_dir, f"bf_model_lgb_{h}h.joblib")
            if not os.path.exists(model_file):
                logger.warning(f"[{h}h] Model bulunamadı, atlanıyor: {model_file}")
                continue

            model      = joblib.load(model_file)
            importance = pd.Series(model.feature_importances_, index=model.feature_name_)
            logger.info(
                f"[{h}h] En önemli 5 feature:\n"
                f"{importance.sort_values(ascending=False).head(5).to_string()}"
            )

        # --- Görsel Rapor ---
        fig_path = os.path.join(reports_dir, "furnace_autopsy.png")
        plt.figure(figsize=(20, 10))
        plt.plot(df["si_dt"], df["Si"], color="gray", alpha=0.3, label="Silicon akışı")

        anoms = df[df["is_anomalous"] == -1]
        plt.scatter(anoms["si_dt"], anoms["Si"], color="red", s=10, label="Anomaliler", zorder=3)

        for idx in gap_indices:
            plt.axvline(x=df.loc[idx, "si_dt"], color="black", linestyle="--", alpha=0.5)

        plt.title("Fırın Otopsisi: Duruşlar ve Termal Şok Analizi")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info(f"Grafik kaydedildi: {fig_path}")
        logger.info("Otopsi tamamlandı.")

    except Exception as e:
        logger.error(f"Otopsi sırasında hata: {e}")
        raise

    # --- DB Export  ---
    try:
        from src.utils.database_manager import get_db_engine
        engine = get_db_engine()
        df.to_sql("gold_furnace_analysis", engine, if_exists="replace", index=False)
        logger.info("Otopsi verileri PostgreSQL 'gold_furnace_analysis' tablosuna aktarıldı.")
    except Exception as e:
        logger.warning(f"DB export atlandı (bağlantı hatası): {e}")


if __name__ == "__main__":
    run_autopsy()