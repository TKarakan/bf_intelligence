#!/usr/bin/env python3
"""
pipeline_check.py
-----------------
Tüm Medallion katmanlarını kontrol eder ve özet rapor üretir.
"""

import os
import pandas as pd
from pyspark.sql import SparkSession

# Yollar
BRONZE_SENSORS = "/app/data/bronze/sensors"
BRONZE_TARGETS = "/app/data/bronze/targets"
SILVER = "/app/data/silver/cleaned"
GOLD = "/app/data/gold/feature_store"

def check_bronze():
    
    print("=" * 60)
    print(" BRONZE KATMANI")
    print("=" * 60)
    
    spark = SparkSession.builder.appName("Check").getOrCreate()
    
    # Sensors
    if os.path.exists(BRONZE_SENSORS):
        df = spark.read.parquet(BRONZE_SENSORS)
        print(f"Sensors: {df.count()} satır")
        print(f"Kolonlar: {df.columns}")
        df.select("dt").show(3, truncate=False)
        df.select("dt").tail(3)
    else:
        print(" Bronze/sensors yok!")
    
    # Targets
    if os.path.exists(BRONZE_TARGETS):
        df = spark.read.parquet(BRONZE_TARGETS)
        print(f"\nTargets: {df.count()} satır")
        print(f"Kolonlar: {df.columns}")
        df.select("dt", "Si").show(3, truncate=False)
    else:
        print(" Bronze/targets yok!")
    
    spark.stop()

def check_silver():
    """Silver katmanını kontrol et."""
    print("\n" + "=" * 60)
    print(" SILVER KATMANI")
    print("=" * 60)
    
    spark = SparkSession.builder.appName("Check").getOrCreate()
    
    if os.path.exists(SILVER):
        df = spark.read.parquet(SILVER)
        n = df.count()
        print(f"Toplam satır: {n}")
        print(f"Kolonlar: {df.columns}")
        
        # Zaman dağılımı
        print("\nİlk 5 satır:")
        df.orderBy("si_dt").show(5, truncate=False)
        
        print("\nSon 5 satır:")
        df.orderBy("si_dt").tail(5)
        
        # Saatlik interval dağılımı
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window
        
        w = Window.orderBy("si_dt")
        intervals = df.withColumn("next_si_dt", F.lead("si_dt", 1).over(w)) \
                      .withColumn("interval_hours", 
                          (F.col("next_si_dt").cast("long") - F.col("si_dt").cast("long")) / 3600.0)
        
        print("\nInterval dağılımı (saat):")
        intervals.select("interval_hours").summary("min", "25%", "50%", "75%", "max").show()
        
        # Si dağılımı
        print("\nSi dağılımı:")
        df.select("Si").summary("min", "25%", "50%", "75%", "max").show()
        
    else:
        print(" Silver yok!")
    
    spark.stop()

def check_gold():
    """Gold katmanını kontrol et."""
    print("\n" + "=" * 60)
    print(" GOLD KATMANI")
    print("=" * 60)
    
    if os.path.exists(GOLD):
        df = pd.read_parquet(GOLD)
        print(f"Toplam satır: {len(df)}")
        print(f"Kolonlar: {df.columns.tolist()}")
        
        print("\nİlk 5 satır (si_dt, Si, target):")
        print(df[['si_dt', 'Si', 'target_Si', 'hours_to_next_cast']].head())
        
        print("\nSon 5 satır:")
        print(df[['si_dt', 'Si', 'target_Si', 'hours_to_next_cast']].tail())
        
        print("\nTarget dağılımı:")
        print(df['target_Si'].describe())
        
        print("\nHours to next cast dağılımı:")
        print(df['hours_to_next_cast'].describe())
        
        # Korelasyonlar
        print("\nSi ile target arasındaki korelasyon:")
        print(df['Si'].corr(df['target_Si']))
        
        # Feature importance (eğer model varsa)
        model_path = "/app/models/bf_model_v4.joblib"
        if os.path.exists(model_path):
            import joblib
            model = joblib.load(model_path)
            print(f"\nModel bulundu: {model_path}")
            print(f"Feature importance (top 10):")
            importance = pd.Series(model.feature_importances_, index=df.drop(columns=['si_dt', 'next_si_dt', 'hours_to_next_cast', 'target_Si', 'target_Si_4h_ahead']).columns)
            print(importance.sort_values(ascending=False).head(10))
        else:
            print("\n Model dosyası bulunamadı")
    else:
        print("❌ Gold yok!")

def bosluk_yakala():
    df = pd.read_parquet('/app/data/gold/feature_store')
    df = df.sort_values("si_dt")

    # En büyük boşluğu bul
    big_gap = df[df['hours_to_next_cast'] > 50].copy()

    if not big_gap.empty:
        print("\n--- 🚩 KRİTİK BOŞLUK TESPİT EDİLDİ ---")
        for i, row in big_gap.iterrows():
            print(f"Duruş Başlangıcı: {row['si_dt']}")
            print(f"Duruş Süresi: {row['hours_to_next_cast']:.2f} saat")
            print(f"Duruş Öncesi Son Silicon (Si): {row['Si']}")
            print("-" * 30)
    else:
        print("100 saatten büyük boşluk bulunamadı.")

def bosluk_sonrasi():
    df = pd.read_parquet('/app/data/gold/feature_store')
    df = df.sort_values("si_dt").reset_index(drop=True)

    # 50 saatten büyük boşlukların indekslerini bul
    gap_indices = df[df['hours_to_next_cast'] > 50].index

    print(f"---  FIRIN UYANIŞ ANALİZİ (Duruş Sonrası İlk Satırlar) ---")

    for idx in gap_indices:
        # Duruştan önceki son satır ve sonraki ilk 3 satır
        analysis_slice = df.iloc[max(0, idx): idx + 4]
        
        print(f"\n📍 Boşluk Başlangıcı: {df.iloc[idx]['si_dt']} (Süre: {df.iloc[idx]['hours_to_next_cast']:.2f} saat)")
        print(analysis_slice[['si_dt', 'Si', 'Th', 'mean_tp', 'hours_to_next_cast']])
        print("-" * 50)
    

def main():
    print(" PIPELINE KONTROL BAŞLIYOR")
    
    check_bronze()
    check_silver()
    check_gold()

    bosluk_yakala()
    bosluk_sonrasi()
    
    print("\n" + "=" * 60)
    print(" KONTROL TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    main()