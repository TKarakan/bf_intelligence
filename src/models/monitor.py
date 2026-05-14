"""
src/models/monitor.py — Model Drift Monitor
============================================
Kullanım:
  - Pipeline'dan: monitor.check_and_retrain(predictions_df)
  - Tekil test:   docker exec bf_orchestrator python -m src.models.monitor

  - 7 günlük rolling MAE karşılaştırması
  - Baseline MAE ile relative drift kontrolü
 
"""

import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

cfg           = load_config()
paths_cfg     = cfg.get("paths", {})
settings_cfg  = cfg.get("settings", {})
REPORTS_DIR   = paths_cfg.get("reports_dir", "reports")
DRIFT_THRESHOLD = settings_cfg.get("drift_threshold", 0.12)
BASELINE_MAE    = settings_cfg.get("baseline_mae", 0.10)


class ModelMonitor:
    def __init__(self, threshold: float | None = None):
        self.threshold    = threshold or DRIFT_THRESHOLD
        self.baseline_mae = BASELINE_MAE
        
        logger.info(f"ModelMonitor başlatıldı | "
                   f"Drift eşiği: {self.threshold:.4f} | "
                   f"Baseline MAE: {self.baseline_mae:.4f}")

   
    # Performans ve Drift Kontrolü    
    def check_performance_drift(self, predictions_df: pd.DataFrame) -> dict:
        """
        Hem anlık hem 7 günlük rolling MAE ile drift kontrolü yapar.
        """
        if len(predictions_df) < 24:
            logger.info("Yeterli veri yok (min 24 tahmin). Drift kontrolü atlandı.")
            return {"drift_detected": False, "current_mae": None, "rolling_7d_mae": None}

        if not {"actual_Si", "predicted_Si"}.issubset(predictions_df.columns):
            raise ValueError("predictions_df 'actual_Si' ve 'predicted_Si' kolonlarını içermelidir.")

        df = predictions_df.copy()

        # Timestamp yoksa simüle et (sıralı hourly tahmin varsayıyoruz)
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(
                end=pd.Timestamp.now(), 
                periods=len(df), 
                freq='H'
            )

        df = df.sort_values('timestamp').reset_index(drop=True)

        # MAE hesaplamaları
        current_mae = mean_absolute_error(df["actual_Si"], df["predicted_Si"])

        df['abs_error'] = abs(df["actual_Si"] - df["predicted_Si"])
        
        # 7 günlük rolling average (168 saat)
        rolling_mae = df['abs_error'].rolling(
            window=168, 
            min_periods=24
        ).mean().iloc[-1]

        # Drift kararları
        current_drift = current_mae > self.threshold
        rolling_drift = rolling_mae > (self.baseline_mae * 1.25)  # %25'ten fazla kötüleşme

        drift_detected = current_drift or rolling_drift

        metrics = {
            "timestamp": pd.Timestamp.now(),
            "current_mae": round(float(current_mae), 4),
            "rolling_7d_mae": round(float(rolling_mae), 4),
            "baseline_mae": round(float(self.baseline_mae), 4),
            "drift_detected": bool(drift_detected),
            "current_vs_threshold": bool(current_drift),
            "rolling_vs_baseline": bool(rolling_drift),
            "sample_count": len(df)
        }

        self.log_drift_metrics(metrics)

        logger.info(
            f"Current MAE: {current_mae:.4f} | "
            f"7g Rolling MAE: {rolling_mae:.4f} | "
            f"Baseline: {self.baseline_mae:.4f} | "
            f"Drift: {drift_detected}"
        )

        if drift_detected:
            logger.warning("⚠️  MODEL DRIFT TESPİT EDİLDİ!")

        return metrics

    
    # Drift Geçmişi Kaydı (Grafana / PowerBI / Raporlama için)
    def log_drift_metrics(self, metrics: dict):
        """Drift metriklerini CSV'ye kaydeder."""
        os.makedirs(REPORTS_DIR, exist_ok=True)

        history_path = os.path.join(REPORTS_DIR, "model_drift_history.csv")

        new_row = pd.DataFrame([metrics])
        new_row.to_csv(
            history_path, 
            mode="a", 
            header=not os.path.exists(history_path), 
            index=False
        )

    
    # Drift + Otomatik Retrain
    def check_and_retrain(self, predictions_df: pd.DataFrame) -> dict | None:
        """
        Pipeline için ana entegrasyon noktası.
        Drift tespit edilirse retraining çalıştırır.
        """
        metrics = self.check_performance_drift(predictions_df)

        if metrics.get("drift_detected"):
            logger.info("🔄 Drift tespit edildi → Yeniden eğitim başlatılıyor...")
            
            try:
                from src.models.train import run_training
                
                # Drift sonrası daha hafif retrain (daha hızlı)
                train_metrics = run_training(n_trials=30, n_splits=5)
                
                logger.info(f"✅ Retrain tamamlandı | Yeni MAE: {train_metrics.get('mae', 0):.4f}")
                return train_metrics
                
            except Exception as e:
                logger.error(f"Retraining sırasında hata oluştu: {e}")
                return None

        logger.info("✅ Model performansı kabul edilebilir aralıkta.")
        return None



# Tekil çalıştırma — Test / Simülasyon
if __name__ == "__main__":
   
    monitor = ModelMonitor()
    # Simülasyon verisi
    import numpy as np
    np.random.seed(42)
    n = 200
    
    actual = np.random.normal(0.46, 0.08, n)
    # Hafif drift senaryosu
    predicted = actual + np.random.normal(0.08, 0.06, n)

    sim_df = pd.DataFrame({"actual_Si": actual,"predicted_Si": predicted})

    result = monitor.check_and_retrain(sim_df)
    
    if result and result.get("mae"):
        print(f"\nYeniden eğitim tamamlandı. Yeni MAE: {result['mae']:.4f}")
    else:
        print("\nDrift tespit edilmedi, model sağlıklı.")