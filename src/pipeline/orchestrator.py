"""
orchestrator.py — Madalyon Pipeline Orkestrasyonu
-------------------------------------------------
Bronze → Silver → Gold katmanlarını sırasıyla çalıştırır.

Kullanım:
    from src.pipeline.orchestrator import run_medallion_cycle
    run_medallion_cycle(spark, fresh=False)
"""

import sys
import time

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType
from src.utils.config_loader import load_config, get_sensor_schema, get_target_schema
from src.utils.kafka_utils import get_kafka_config
from src.utils.logger import get_logger
from src.utils.cleanup import cleanup_pipeline
from src.utils.pipeline_utils import wait_for_data, is_idle

from src.bronze.excel_extractor import extract_excel_sheets
from src.bronze.kafka_streamer import BlastFurnaceStreamer
from src.silver.silver_stream import run_silver_batch
from src.gold.gold_refiner import GoldFeatureStore


logger = get_logger(__name__)


try:
    _CFG           = load_config()
    _PATHS         = _CFG.get("paths", {})
    _KAFKA_CFG     = get_kafka_config()
    _SENSOR_SCHEMA = get_sensor_schema(_CFG)
    _TARGET_SCHEMA = get_target_schema(_CFG)
except Exception as e:
    logger.error(f"Config yüklenemedi: {e}")
    raise

BRONZE_BASE = _PATHS.get("raw_bronze_dir", "/app/data/bronze")
SILVER_OUT  = f"{_PATHS.get('refined_silver_dir', '/app/data/silver')}/cleaned"


def _configure_dynamic_overwrite(spark):
    """Spark'ı dynamic partition overwrite moduna alır."""
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    logger.info("⚙️  Dynamic partition overwrite modu etkinleştirildi.")


def _make_bronze_query(spark, topic: str, base_schema, folder: str):
    """
    Kafka'dan okuyan, year/month/day ile partition eden Bronze stream query'si.
    """
    ingestion_schema = base_schema.add("source_type", StringType(), True)

    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", _KAFKA_CFG["bootstrap.servers"])
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
        .selectExpr("CAST(value AS STRING) AS json_str")
        .select(F.from_json(F.col("json_str"), ingestion_schema).alias("d"))
        .select("d.*")
        .withColumn("_ingest_ts", F.coalesce(F.col("dt").cast("timestamp"), F.current_timestamp()))
        .withColumn("year",  F.year("_ingest_ts").cast(IntegerType()))
        .withColumn("month", F.month("_ingest_ts").cast(IntegerType()))
        .withColumn("day",   F.dayofmonth("_ingest_ts").cast(IntegerType()))
        .drop("_ingest_ts")
        .writeStream
        .format("parquet")
        .partitionBy("year", "month", "day")
        .option("path",               f"{BRONZE_BASE}/{folder}")
        .option("checkpointLocation", f"{BRONZE_BASE}/checkpoints/{folder}")
        .start()
    )


def start_bronze_streams(spark):
    """Bronze stream query'lerini başlatır."""
    return [
        _make_bronze_query(spark, "blast-furnace-data", _SENSOR_SCHEMA, "sensors"),
        _make_bronze_query(spark, "silicon-data",       _TARGET_SCHEMA,  "targets"),
    ]


def run_medallion_cycle(spark, fresh: bool = False):
    """
    Katmanları sırasıyla çalıştıran ana orkestrasyon fonksiyonu.

    Args:
        spark: SparkSession.
        fresh: True ise tüm veriyi sıfırlar.
    """
    cleanup_pipeline(fresh=fresh)
    _configure_dynamic_overwrite(spark)

    # [1/5] Excel Extraction (blocking)
    logger.info("[1/5] Excel Extraction başlatılıyor...")
    
    extract_excel_sheets()
    logger.info("[1/5] TAMAM")

    # [2/5] Kafka Streamer (blocking)
    logger.info("[2/5] Kafka Streamer başlatılıyor...")
    
    BlastFurnaceStreamer().run()
    logger.info("[2/5] TAMAM")

    # [3/5] Bronze Streaming
    logger.info("[3/5] Bronze stream başlatılıyor...")
    bronze_queries = start_bronze_streams(spark)

    if not wait_for_data(f"{BRONZE_BASE}/sensors", "Bronze/sensors", timeout=180):
        sys.exit(1)
    if not wait_for_data(f"{BRONZE_BASE}/targets", "Bronze/targets", timeout=60):
        sys.exit(1)

    is_idle(bronze_queries[0], "Bronze Sensors")
    is_idle(bronze_queries[1], "Bronze Targets")

    for q in bronze_queries:
        try:
            q.stop()
            time.sleep(2)  # ← Son batch'in tamamlanması için bekle
        except Exception:
            pass
    logger.info("[3/5] TAMAM")

    # [4/5] Silver (batch, dynamic overwrite)
    logger.info("[4/5] Silver batch işleniyor...")
    
    run_silver_batch(spark)
    logger.info("[4/5] TAMAM")

    # [5/5] Gold Feature Store
    logger.info("[5/5] Gold Feature Store işleniyor...")
    
    GoldFeatureStore(spark).run_gold_pipeline()
    logger.info("[5/5] TAMAM")