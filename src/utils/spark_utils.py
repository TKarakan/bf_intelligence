from pyspark.sql import SparkSession
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
import os

logger = get_logger(__name__)

def get_spark_session(app_name="BlastFurnaceStreaming"):
    cfg = load_config()
    
    
    kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", cfg.get("kafka", {}).get("bootstrap_servers", "kafka:29092"))
    
    
    spark_master = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
    
    kafka_package = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"

    try:
        spark = (SparkSession.builder
            .appName(app_name)
            .master(spark_master)
            .config("spark.jars.packages", kafka_package)
            .config("spark.kafka.bootstrap.servers", kafka_servers)
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()) 
        
        spark.sparkContext.setLogLevel("WARN")
        logger.info(f" Spark bağlandı. Master: {spark_master}")
        return spark
    except Exception as e:
        logger.error(f"Spark başlatılamadı: {e}")
        raise

def stop_spark_session(spark):
    if spark:
        spark.stop()
        logger.info("Spark Session kapatıldı.")