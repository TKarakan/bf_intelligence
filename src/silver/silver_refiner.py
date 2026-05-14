from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from src.utils.logger import get_logger
from src.utils.config_loader import get_sensor_schema

logger = get_logger(__name__)


try:
    _SENSOR_SCHEMA = get_sensor_schema()
except Exception as e:
    logger.error(f"Schema yüklenemedi: {e}")
    raise


class PhysicalRefiner:
    def __init__(self, spark: SparkSession, bronze_path: str):
        self.spark = spark
        self.bronze_path = f"{bronze_path}/sensors"
        self._pool = None

        # Ağırlıklar (ileride YAML'a taşınabilir)
        self.weights = {
            'Fb': 1.2, 'Th': 1.2, 'R': 1.0,
            'CO2': 0.8, 'H2': 0.8, 'dP': 0.8,
            'Ph': 0.5, 'Pc': 0.5, 'Pt': 0.5,
            'Tp': 0.2, 'Tt': 0.1
        }

        
        sensor_fields = _SENSOR_SCHEMA.fieldNames()
        self.tt_cols = sorted([c for c in sensor_fields if c.startswith("Tt")])
        self.tp_cols = sorted([c for c in sensor_fields if c.startswith("Tp")])

        logger.info(f"Sıcaklık kolonları yüklendi: tt={self.tt_cols}, tp={self.tp_cols}")

    def _build_hybrid_pool(self):
        if self._pool is not None:
            return self._pool

        logger.info("🚀 Hibrit Referans Havuzu oluşturuluyor...")

        df = self.spark.read.parquet(self.bronze_path)

        flagged = self.apply_outlier_flags(df)

        # Oracle (second) + Temiz Gerçek (first & is_outlier=0)
        pool_df = flagged.filter(
            (F.col("source_type") == "second") |
            ((F.col("source_type") == "first") & (F.col("is_outlier") == 0))
        )

        num_cols = [f.name for f in pool_df.schema.fields if isinstance(f.dataType, DoubleType)]

        exprs = [F.mean(F.col(c)).alias(c) for c in num_cols]

        means_row = pool_df.agg(*exprs).collect()[0]

        self._pool = means_row.asDict()
        return self._pool

    def apply_outlier_flags(self, df: DataFrame) -> DataFrame:
        # Dinamik tt/tp kolonları — schemas.yaml'dan
        n_tt = len(self.tt_cols)
        n_tp = len(self.tp_cols)

        avg_tt = sum(F.col(c) for c in self.tt_cols) / n_tt if n_tt > 0 else F.lit(0)
        avg_tp = sum(F.col(c) for c in self.tp_cols) / n_tp if n_tp > 0 else F.lit(0)

        is_valid = (
            (F.col("Th") > F.col("Tc")) & (F.col("Th").between(0, 1500)) &
            (F.col("Tc").between(0, 150)) & (F.col("Fo").between(0, 35000)) &
            (F.col("Fb").between(500, 4500)) & (F.col("R").between(1.2, 8.0)) &
            (F.col("Pc") > F.col("Pt")) & (F.col("dP").between(0.2, 2.5)) &
            (F.col("CO2").between(8, 30)) & (F.col("H2").between(0.2, 12)) &
            (avg_tt < avg_tp)  # Tepe < Çeper gradyenti
        )
        return df.withColumn("is_outlier", F.when(is_valid, 0).otherwise(1))

    def impute_with_oracle(self, df: DataFrame) -> DataFrame:
        means = self._build_hybrid_pool()

        for col, val in means.items():
            if col in df.columns and val is not None:
                df = df.withColumn(col,
                    F.when((F.col("is_outlier") == 1) & (F.col(col).isNull() | (F.col(col) <= 0)),
                    F.lit(val)).otherwise(F.col(col)))
        return df