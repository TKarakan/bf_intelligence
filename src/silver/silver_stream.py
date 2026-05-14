"""
silver_stream.py — Batch Mode Silver Refiner
---------------------------------------------
Bronze → Silver: Temizlik, İmputasyon, Saatlik Ortalama, Join

Çıktı şeması (Gold'un beklediği):
  si_dt (timestamp), Si (double), Fb, Th, Tc, Fo, R, Ph, Pc, Pt, 
  dP, dPu, dPl, CO2, H2, Tt1..Tt4, Tp1..Tp10
"""

import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, TimestampType, DoubleType

from src.silver.silver_refiner import PhysicalRefiner
from src.utils.logger import get_logger
from src.utils.config_loader import load_config, get_sensor_schema

logger = get_logger(__name__)


try:
    cfg = load_config()
    PATHS = cfg.get("paths", {})
    BRONZE_BASE = PATHS.get("raw_bronze_dir", "/app/data/bronze")
    SILVER_OUT  = f"{PATHS.get('refined_silver_dir', '/app/data/silver')}/cleaned"
    SENSOR_SCHEMA = get_sensor_schema(cfg)
except Exception as e:
    logger.error(f"Config hatası: {e}")
    raise


PHYSICAL_COLS = [
    f.name for f in SENSOR_SCHEMA.fields 
    if f.name != "dt" and isinstance(f.dataType, DoubleType)
]


SILVER_OUTPUT_COLS = ["si_dt", "Si"] + PHYSICAL_COLS


def run_silver_batch(spark: SparkSession):

    logger.info(" SILVER BATCH İŞLEME BAŞLIYOR...")

    refiner = PhysicalRefiner(spark, BRONZE_BASE)

    sensors_path = f"{BRONZE_BASE}/sensors"
    targets_path = f"{BRONZE_BASE}/targets"

    if not os.path.exists(sensors_path):
        raise FileNotFoundError(f"Bronze sensors bulunamadı: {sensors_path}")
    if not os.path.exists(targets_path):
        raise FileNotFoundError(f"Bronze targets bulunamadı: {targets_path}")


    logger.info("Sensör verisi okunuyor...")
    sensors_raw = spark.read.parquet(sensors_path)

    # Debug: schema kontrolü
    logger.info(f"Bronze sensors schema: {[f.name for f in sensors_raw.schema.fields]}")

    sensors = (
        sensors_raw
        .withColumn("sensor_dt", F.to_timestamp(F.col("dt")))
        .drop("dt")
        .transform(refiner.apply_outlier_flags)
        .transform(refiner.impute_with_oracle)
        .withColumn("rn", F.row_number().over(Window.partitionBy("sensor_dt").orderBy(F.col("source_type").desc())))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .withColumn("join_key", F.date_trunc("hour", F.col("sensor_dt")))
    )


    logger.info("Saatlik ortalamalar hesaplanıyor...")


    agg_exprs = [F.avg(F.col(c)).alias(c) for c in PHYSICAL_COLS]

    sensors_hourly = (
        sensors.groupBy("join_key")
        .agg(*agg_exprs, F.max("is_outlier").alias("is_outlier"))
    )


    logger.info("Target verisi okunuyor...")
    targets_raw = spark.read.parquet(targets_path)

    # Debug
    logger.info(f"Bronze targets schema: {[f.name for f in targets_raw.schema.fields]}")

    targets = (
        targets_raw
        .withColumn("si_dt_raw", F.to_timestamp(F.col("dt")))
        .withColumn("join_key", F.date_trunc("hour", F.col("si_dt_raw")))
        .withColumn("rn", F.row_number().over(
            Window.partitionBy("join_key")
                  .orderBy(F.col("si_dt_raw").desc())
        ))
        .filter(F.col("rn") == 1)
        .drop("rn", "dt", "si_dt_raw")
    )

    n_targets = targets.count()
    logger.info(f"Target'tan {n_targets} saatlik eşsiz kayıt bulundu.")
    if n_targets == 0:
        raise ValueError("Target verisi boş! Bronze'da target parquet'i kontrol et.")


    logger.info("Sensör ve Target join ediliyor...")

    joined = sensors_hourly.join(
        targets, 
        on="join_key", 
        how="inner"
    )

    n_joined = joined.count()
    logger.info(f"Join sonrası {n_joined} satır.")
    if n_joined == 0:
        raise ValueError("Join sonucu boş! source_type veya zaman eşleşmesi yok.")


    final_df = joined.select(
        F.col("join_key").alias("si_dt"),
        F.col("Si"),
        *[F.col(c) for c in PHYSICAL_COLS]
    )


    final_df = final_df.dropna(subset=["si_dt", "Si"] + PHYSICAL_COLS)

    n_final = final_df.count()
    logger.info(f"Silver'a yazılacak nihai satır sayısı: {n_final}")

    if n_final == 0:
        raise ValueError("Silver çıktısı boş! Null temizliği çok agresif olabilir.")


    final_df.write.mode("overwrite").parquet(SILVER_OUT)
    logger.info(f"✅ Silver yazıldı: {SILVER_OUT}")

    return final_df


def start_silver_stream(spark: SparkSession):

    raise NotImplementedError(
        "Streaming mode kaldırıldı. Lütfen run_silver_batch() kullanın."
    )