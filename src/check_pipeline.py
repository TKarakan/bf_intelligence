#!/usr/bin/env python3
"""
check_pipeline.py
-----------------
Tüm Medallion katmanlarını (Bronze, Silver, Gold), fırın duruşlarını
ve eğitilmiş multi-horizon modellerini kontrol eder.
"""

import os
import sys
import pandas as pd
from src.utils.config_loader import load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

cfg = load_config()
paths_cfg = cfg.get("paths", {})

def _resolve_dir(raw_path: str, default: str) -> str:
    path = raw_path or default
    if path.startswith("/app/") and not os.path.exists("/app"):
        path = path.replace("/app/", "", 1)
    return path

BRONZE_BASE = _resolve_dir(paths_cfg.get("raw_bronze_dir", "data/bronze"), "data/bronze")
SILVER_BASE = _resolve_dir(paths_cfg.get("refined_silver_dir", "data/silver"), "data/silver")
GOLD_BASE   = _resolve_dir(paths_cfg.get("feature_gold_dir", "data/gold"), "data/gold")
MODELS_DIR  = _resolve_dir(paths_cfg.get("models_dir", "models"), "models")

BRONZE_SENSORS = os.path.join(BRONZE_BASE, "sensors")
BRONZE_TARGETS = os.path.join(BRONZE_BASE, "targets")
SILVER = os.path.join(SILVER_BASE, "cleaned")
GOLD   = os.path.join(GOLD_BASE, "feature_store") if not GOLD_BASE.endswith("feature_store") else GOLD_BASE

FORECAST_HORIZONS = [2, 4, 6, 8]


def check_bronze():
    print("=" * 60)
    print(" BRONZE KATMANI")
    print("=" * 60)
    
    # Sensors
    if os.path.exists(BRONZE_SENSORS):
        try:
            df = pd.read_parquet(BRONZE_SENSORS)
            print(f"Sensors: {len(df)} satır | Kolonlar: {len(df.columns)}")
            if "dt" in df.columns:
                print("İlk 2 kayıt:")
                print(df[["dt"] + [c for c in ["Fb", "Th", "Si"] if c in df.columns]].head(2))
        except Exception as e:
            print(f"Sensors okuma hatası: {e}")
    else:
        print("ℹ️ Bronze/sensors yolu bulunamadı (henüz ingest edilmemiş olabilir).")
    
    # Targets
    if os.path.exists(BRONZE_TARGETS):
        try:
            df = pd.read_parquet(BRONZE_TARGETS)
            print(f"\nTargets: {len(df)} satır")
            if "dt" in df.columns and "Si" in df.columns:
                print(df[["dt", "Si"]].head(2))
        except Exception as e:
            print(f"Targets okuma hatası: {e}")
    else:
        print("ℹ️ Bronze/targets yolu bulunamadı.")


def check_silver():
    print("\n" + "=" * 60)
    print(" SILVER KATMANI")
    print("=" * 60)
    
    if os.path.exists(SILVER):
        try:
            df = pd.read_parquet(SILVER)
            print(f"Toplam satır: {len(df)} | Kolon sayısı: {len(df.columns)}")
            if "si_dt" in df.columns and "Si" in df.columns:
                df["si_dt"] = pd.to_datetime(df["si_dt"])
                print(f"Zaman aralığı: {df['si_dt'].min()} -> {df['si_dt'].max()}")
                print(f"Si istatistikleri:\n{df['Si'].describe().to_string()}")
        except Exception as e:
            print(f"Silver okuma hatası: {e}")
    else:
        print("ℹ️ Silver/cleaned yolu bulunamadı.")


def check_gold():
    print("\n" + "=" * 60)
    print(" GOLD KATMANI & FEATURE STORE")
    print("=" * 60)
    
    if os.path.exists(GOLD):
        try:
            df = pd.read_parquet(GOLD)
            print(f"Toplam satır: {len(df)} | Toplam kolon: {len(df.columns)}")
            
            target_cols = [c for c in df.columns if c.startswith("target_Si_")]
            print(f"Multi-Horizon Hedef Kolonları: {target_cols}")
            
            preview_cols = ["si_dt", "Si"] + target_cols
            preview_cols = [c for c in preview_cols if c in df.columns]
            if "hours_to_next_cast" in df.columns:
                preview_cols.append("hours_to_next_cast")
                
            print("\nİlk 3 satır:")
            print(df[preview_cols].head(3))
            
            # Model dosyaları kontrolü
            print("\nEğitilmiş Modeller:")
            import joblib
            for h in FORECAST_HORIZONS:
                m_path = os.path.join(MODELS_DIR, f"bf_model_lgb_{h}h.joblib")
                if os.path.exists(m_path):
                    model = joblib.load(m_path)
                    n_features = len(model.feature_name_) if hasattr(model, "feature_name_") else "?"
                    print(f"  ✅ [{h}h] {m_path} (Özellik sayısı: {n_features})")
                else:
                    print(f"  ❌ [{h}h] Model dosyası yok: {m_path}")
        except Exception as e:
            print(f"Gold okuma hatası: {e}")
    else:
        print("❌ Gold feature store bulunamadı!")


def check_furnace_gaps():
    print("\n" + "=" * 60)
    print(" FIRIN DURUŞ & BOŞLUK ANALİZİ")
    print("=" * 60)
    
    if not os.path.exists(GOLD):
        return

    try:
        df = pd.read_parquet(GOLD)
        if "hours_to_next_cast" not in df.columns:
            return

        df["si_dt"] = pd.to_datetime(df["si_dt"])
        df = df.sort_values("si_dt").reset_index(drop=True)

        big_gaps = df[df["hours_to_next_cast"] > 50]
        if not big_gaps.empty:
            print(f"🚩 >50 Saatlik Kritik Duruş Sayısı: {len(big_gaps)}")
            for _, row in big_gaps.iterrows():
                print(f"  Duruş Zamanı: {row['si_dt']} | Süre: {row['hours_to_next_cast']:.1f}h | Son Si: {row['Si']:.2f}")
        else:
            print("50 saatten büyük fırın duruşu tespit edilmedi.")
    except Exception as e:
        print(f"Duruş analizi hatası: {e}")


def main():
    print("🔍 BF INTELLIGENCE — SİSTEM SAĞLIK & PİPELİNE DENETİMİ")
    check_bronze()
    check_silver()
    check_gold()
    check_furnace_gaps()
    print("\n" + "=" * 60)
    print(" DENETİM TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    main()