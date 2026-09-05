"""
src/models/predict.py — BF Intelligence Canlı Tahmin (v4.1)
===========================================================
Online Prediction Pipeline:
  1. DB'den son satırı (materialized features) çek.
  2. UI'dan gelen taze sensör verilerini bu satırın üzerine yaz.
  3. Sadece t anı ile t-1 anı arasındaki farklara dayalı (lag_1, diff, velocity)
     feature'ları güncelle.
  4. Kalan tüm karmaşık window/rolling feature'ları DB'den geldiği gibi bırak.
  5. API/UI kolaylığı için single-horizon tahmin metodunu ekle.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.utils.database_manager import get_db_engine

logger = get_logger(__name__)

FORECAST_HORIZONS = [2, 4, 6, 8]

cfg        = load_config()
paths_cfg  = cfg.get("paths", {})

def _resolve_dir(raw_path: str, default: str) -> str:
    path = raw_path or default
    if path.startswith("/app/") and not os.path.exists("/app"):
        path = path.replace("/app/", "", 1)
    return path

MODELS_DIR = _resolve_dir(paths_cfg.get("models_dir", "models"), "models")
GOLD_PATH  = _resolve_dir(paths_cfg.get("feature_gold_dir", "data/gold"), "data/gold")


class BFPredictor:
    def __init__(self, horizons: list[int] | None = None):
        self.models: dict[int, object] = {}

        target_horizons = horizons or FORECAST_HORIZONS
        for h in target_horizons:
            model_file = f"bf_model_lgb_{h}h.joblib"
            model_path = os.path.join(MODELS_DIR, model_file)
            if not os.path.exists(model_path):
                logger.warning(f"Model bulunamadı, atlanıyor: {model_path}")
                continue
            self.models[h] = joblib.load(model_path)
            logger.info(f"[{h}h] Model yüklendi: {model_path}")

        if not self.models:
            raise FileNotFoundError("Hiçbir model yüklenemedi! Beklenen dizin: models_dir")

        try:
            self.engine = get_db_engine()
        except Exception as e:
            logger.warning(f"PostgreSQL bağlantısı kurulamadı (Parquet fallback devrede): {e}")
            self.engine = None

    def _get_latest_from_db(self) -> pd.DataFrame:
        """
        Velocity ve Acceleration hesaplayabilmek için t-1 ve t-2'ye ihtiyaç var.
        Önce PostgreSQL tablosunu (gold_furnace_analysis) dener.
        Erişilemezse Gold Parquet (feature_store) üzerinden son 2 satırı okur.
        """
        if self.engine is not None:
            try:
                query = text("SELECT * FROM gold_furnace_analysis ORDER BY si_dt DESC LIMIT 2")
                with self.engine.begin() as conn:
                    df = pd.read_sql(query, conn)
                if not df.empty:
                    return df.sort_values("si_dt").reset_index(drop=True)
            except Exception as e:
                logger.warning(f"DB'den veri çekilemedi, Parquet fallback deneniyor: {e}")

        # Fallback: Gold Feature Store Parquet
        try:
            parquet_path = os.path.join(GOLD_PATH, "feature_store") if not GOLD_PATH.endswith("feature_store") else GOLD_PATH
            if os.path.exists(parquet_path):
                df_gold = pd.read_parquet(parquet_path)
                if not df_gold.empty:
                    df_gold["si_dt"] = pd.to_datetime(df_gold["si_dt"])
                    logger.info("✅ En güncel fırın durumu Gold Parquet feature_store'dan çekildi.")
                    return df_gold.sort_values("si_dt").tail(2).reset_index(drop=True)
        except Exception as e:
            logger.error(f"Parquet fallback'ten veri çekilemedi: {e}")

        return pd.DataFrame()

    def predict_silicon(self, sensor_data: pd.DataFrame, horizons: list[int] | None = None) -> dict[int, float]:
        """
        Birden fazla horizon için tahmin üretir. Sözlük döndürür.
        """
        target_horizons = horizons or list(self.models.keys())
        
        df_db = self._get_latest_from_db()
        if df_db.empty:
            logger.warning("DB ve Parquet geçmiş verisi bulunamadı. Gelen sensör verisi üzerinden tahmin üretiliyor.")
            first_model = next(iter(self.models.values()))
            current_features = pd.Series(0.0, index=first_model.feature_name_)
            last_row = current_features.copy()
            prev_row = current_features.copy()
        else:
            last_row = df_db.iloc[-1].copy()
            prev_row = df_db.iloc[-2] if len(df_db) > 1 else last_row
            current_features = last_row.copy()

        # 2. UI/Sensor verileriyle ana kolonları güncelle
        base_sensors = ["Fb", "Th", "dP", "CO2", "H2", "R", "Tc", "Si", "Fo"]
        for col in base_sensors:
            if col in sensor_data.columns:
                current_features[col] = sensor_data[col].values[0]

        # 3. Termal ve basınç türetmelerini anlık hesapla
        tp_cols = [f"Tp{i}" for i in range(1, 11)]
        tt_cols = [f"Tt{i}" for i in range(1, 5)]
        
        if all(c in sensor_data.columns for c in tp_cols):
            tp_vals = sensor_data[tp_cols].values[0]
            current_features["mean_tp"] = np.mean(tp_vals)
            current_features["tp_imbalance"] = np.max(tp_vals) - np.min(tp_vals)
            for c in tp_cols: current_features[c] = sensor_data[c].values[0]

        if all(c in sensor_data.columns for c in tt_cols):
            tt_vals = sensor_data[tt_cols].values[0]
            current_features["mean_tt"] = np.mean(tt_vals)
            for c in tt_cols: current_features[c] = sensor_data[c].values[0]

        current_features["furnace_thermal_drop"]   = current_features["Th"] - current_features.get("mean_tt", 0)
        current_features["blast_heat_addition"]    = current_features["Th"] - current_features["Tc"]
        current_features["thermal_position_index"] = current_features.get("mean_tp", 0) / (current_features.get("mean_tt", 0) + 1e-5)
        current_features["permeability_index"]     = current_features["Fb"] / (current_features["dP"] + 1e-5)
        
        if "dPu" in sensor_data.columns and "dPl" in sensor_data.columns:
            current_features["pressure_distribution_ratio"] = sensor_data["dPu"].values[0] / (sensor_data["dPl"].values[0] + 1e-5)

        # 4. State-based (t vs t-1) farkları hesapla
        for col in ["Si", "Fb", "Th", "Tc", "dP", "CO2", "H2", "R"]:
            current_features[f"{col}_lag_1h"] = last_row[col]
            current_features[f"{col}_diff_1h"] = current_features[col] - last_row[col]
            current_features[f"{col}_diff_pct"] = (current_features[col] - last_row[col]) / (last_row[col] + 1e-5)

        dt = 1.0 
        for src in ["Si", "Th", "Fb", "CO2"]:
            current_features[f"{src.lower()}_velocity"] = (current_features[src] - last_row[src]) / dt

        prev_si_vel = (last_row["Si"] - prev_row["Si"]) / dt
        current_features["si_acceleration"] = (current_features["si_velocity"] - prev_si_vel) / dt
        
        prev_th_vel = (last_row["Th"] - prev_row["Th"]) / dt
        current_features["th_acceleration"] = (current_features["th_velocity"] - prev_th_vel) / dt

        # 5. Model Inference
        df_live = pd.DataFrame([current_features])

        for col in df_live.columns:
            df_live[col] = pd.to_numeric(df_live[col], errors="coerce")
        df_live = df_live.fillna(0.0)
        predictions = {}

        for h in sorted(target_horizons):
            model_features = self.models[h].feature_name_
            missing = [c for c in model_features if c not in df_live.columns]
            for col in missing:
                df_live[col] = 0.0
                
            X_live = df_live[model_features]
            pred = self.models[h].predict(X_live)
            predictions[h] = float(pred[0])

        return predictions

    def predict_silicon_single(self, sensor_data: pd.DataFrame, horizon_h: int = 8) -> float:
        """
        Spesifik bir zaman dilimi için tekil tahmin döndürür.
        Özellikle API endpoint'leri ve basit UI gösterimleri için uygundur.
        """
        result = self.predict_silicon(sensor_data, horizons=[horizon_h])
        return result[horizon_h]

if __name__ == "__main__":
    # Test
    dummy_data = pd.DataFrame([{
        "Fb": 3800.0, "Th": 1150.0, "dP": 1.45, "CO2": 18.5, "H2": 4.2,
        "R": 3.8, "Tc": 65.0, "Si": 0.45,
        "Tt1": 120.0, "Tt2": 125.0, "Tt3": 118.0, "Tt4": 122.0,
        "Tp1": 450.0, "Tp2": 460.0, "Tp3": 455.0, "Tp4": 458.0,
        "Tp5": 452.0, "Tp6": 451.0, "Tp7": 459.0, "Tp8": 457.0,
        "Tp9": 453.0, "Tp10": 455.0
    }])

    predictor = BFPredictor()
    single_res = predictor.predict_silicon_single(dummy_data, horizon_h=8)
    print(f"\n8 Saat Sonraki Tahmin Edilen Si: {single_res:.4f}")