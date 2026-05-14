from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from src.utils.spark_utils import get_spark_session
from src.utils.kafka_utils import get_kafka_config
from src.utils.config_loader import load_config, get_sensor_schema, get_target_schema
from src.utils.logger import get_logger

logger = get_logger(__name__)


try:
    cfg           = load_config()
    PATHS         = cfg.get("paths", {})
    KAFKA_CFG     = get_kafka_config()  
    SENSOR_SCHEMA = get_sensor_schema(cfg)
    TARGET_SCHEMA = get_target_schema(cfg)
except Exception as e:
    logger.error(f"Config yüklenemedi: {e}")
    raise


BRONZE_BASE = PATHS.get("raw_bronze_dir", "/app/data/bronze")

TOPIC_MAP = {
    "blast-furnace-data": {
        "folder": "sensors",
        "schema": SENSOR_SCHEMA,
    },
    "silicon-data": {
        "folder": "targets",
        "schema": TARGET_SCHEMA,
    },
}


def start_bronze_stream(spark: SparkSession, topic: str, info: dict):
    folder = info["folder"]
    schema = info["schema"]

    output_path     = f"{BRONZE_BASE}/{folder}"
    checkpoint_path = f"{BRONZE_BASE}/checkpoints/{folder}"

    df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_CFG["bootstrap.servers"])
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 10_000)
        .load()

        .selectExpr("CAST(value AS STRING) AS json_str")

        .select(from_json(col("json_str"), schema).alias("data"))
        .select("data.*")
    )

    query = (
        df.writeStream
        .format("parquet")
        .outputMode("append")
        .option("path", output_path)
        .option("checkpointLocation", checkpoint_path)
        .trigger(processingTime="10 seconds")
        .start()
    )

    logger.info(f"Bronze akışı aktif: {topic} → {output_path}")
    return query


def main():
    
    spark = get_spark_session(app_name="BF-Ingest-to-Bronze")

    queries = []
    for topic, info in TOPIC_MAP.items():
        q = start_bronze_stream(spark, topic, info)
        queries.append(q)

    logger.info(
        f"{len(queries)} topic Bronze'a akıyor. "
        f"Çıkmak için Ctrl+C."
    )
    spark.streams.awaitAnyTermination()