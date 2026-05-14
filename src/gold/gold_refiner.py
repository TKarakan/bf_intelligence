import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, DoubleType, TimestampType
)
from src.utils.logger import get_logger
from src.utils.config_loader import load_config, get_sensor_schema

logger = get_logger(__name__)

FORECAST_HORIZONS = [2, 4, 6, 8]


try:
    _CFG           = load_config()
    _PATHS         = _CFG.get("paths", {})
    _SENSOR_SCHEMA = get_sensor_schema(_CFG)
except Exception as e:
    logger.error(f"Config yüklenemedi: {e}")
    raise


class GoldFeatureStore:

    def __init__(self, spark: SparkSession):
        self.spark       = spark
        self.silver_path = f"{_PATHS.get('refined_silver_dir', '/app/data/silver')}/cleaned"
        self.gold_path   = f"{_PATHS.get('feature_gold_dir', '/app/data/gold')}/feature_store"

        # Partition tanımlı window'lar (uyarı önleme)
        _base_ts = Window.partitionBy(F.lit(0)).orderBy(F.col("si_dt").cast("long"))
        self.win_1h  = _base_ts.rangeBetween(-3600,      0)
        self.win_2h  = _base_ts.rangeBetween(-2  * 3600, 0)
        self.win_4h  = _base_ts.rangeBetween(-4  * 3600, 0)
        self.win_8h  = _base_ts.rangeBetween(-8  * 3600, 0)
        self.win_12h = _base_ts.rangeBetween(-12 * 3600, 0)
        self.win_24h = _base_ts.rangeBetween(-24 * 3600, 0)

        self.w_lag   = Window.partitionBy(F.lit(0)).orderBy("si_dt")
        self.w_lag_4 = Window.partitionBy(F.lit(0)).orderBy("si_dt").rowsBetween(-4, 0)
        self.w_lag_8 = Window.partitionBy(F.lit(0)).orderBy("si_dt").rowsBetween(-8, 0)

    def _get_silver_read_schema(self) -> StructType:
        physical_fields = [
            f for f in _SENSOR_SCHEMA.fields
            if f.name != "dt" and isinstance(f.dataType, DoubleType)
        ]
        return StructType([
            StructField("si_dt", TimestampType(), True),
            StructField("Si",    DoubleType(),    True),
        ] + physical_fields)

    def deduplicate_data(self, df: DataFrame) -> DataFrame:
        return df.dropDuplicates(["si_dt"]).dropna(subset=["si_dt"])

    def clean_time_gaps(self, df: DataFrame) -> DataFrame:
        df = df.withColumn("dt_sec", F.col("si_dt").cast("long"))
        df = df.withColumn("delta_t_hours",(F.col("dt_sec") - F.lag("dt_sec", 1).over(self.w_lag)) / 3600.0)
        df = df.filter((F.col("delta_t_hours") < 6.0) | (F.col("delta_t_hours").isNull()))
        return df.drop("dt_sec")

    def generate_time_features(self, df: DataFrame) -> DataFrame:
        logger.info("Time Features: Cyclical zaman özellikleri...")
        df = df.withColumn("hour",         F.hour("si_dt"))
        df = df.withColumn("day_of_week",  F.dayofweek("si_dt"))
        df = df.withColumn("day_of_month", F.dayofmonth("si_dt"))
        df = df.withColumn("month",        F.month("si_dt"))
        df = df.withColumn("hour_sin", F.sin(2 * F.lit(3.14159) * F.col("hour") / 24))
        df = df.withColumn("hour_cos", F.cos(2 * F.lit(3.14159) * F.col("hour") / 24))
        df = df.withColumn("dow_sin",  F.sin(2 * F.lit(3.14159) * F.col("day_of_week") / 7))
        df = df.withColumn("dow_cos",  F.cos(2 * F.lit(3.14159) * F.col("day_of_week") / 7))
        return df

    def generate_lag_features(self, df: DataFrame) -> DataFrame:
        logger.info("Lag Features: Geçmiş değerler...")
        for lag in [1, 2, 3, 4, 8, 12]:
            df = df.withColumn(f"Si_lag_{lag}h", F.lag("Si", lag).over(self.w_lag))
        df = df.withColumn("Si_roll_mean_4h",  F.avg("Si").over(self.win_4h))
        df = df.withColumn("Si_roll_std_4h",   F.stddev("Si").over(self.win_4h))
        df = df.withColumn("Si_roll_mean_8h",  F.avg("Si").over(self.win_8h))
        df = df.withColumn("Si_roll_mean_24h", F.avg("Si").over(self.win_24h))
        key_sensors = ["Fb", "Th", "Tc", "dP", "CO2", "H2", "R"]
        for sensor in key_sensors:
            if sensor in df.columns:
                df = df.withColumn(f"{sensor}_lag_1h",       F.lag(sensor, 1).over(self.w_lag))
                df = df.withColumn(f"{sensor}_lag_4h",       F.lag(sensor, 4).over(self.w_lag))
                df = df.withColumn(f"{sensor}_roll_mean_4h", F.avg(sensor).over(self.win_4h))
        return df

    def generate_thermal_features(self, df: DataFrame) -> DataFrame:
        logger.info("Thermal Features...")
        tp_cols = [f"Tp{i}" for i in range(1, 11)]
        tt_cols = [f"Tt{i}" for i in range(1, 5)]
        df = df.withColumn("mean_tp", sum(F.col(c) for c in tp_cols) / 10.0)
        df = df.withColumn("mean_tt", sum(F.col(c) for c in tt_cols) / 4.0)
        tp_max = F.greatest(*[F.col(c) for c in tp_cols])
        tp_min = F.least(*[F.col(c) for c in tp_cols])
        df = df.withColumn("tp_imbalance",            tp_max - tp_min)
        df = df.withColumn("furnace_thermal_drop",     F.col("Th") - F.col("mean_tt"))
        df = df.withColumn("blast_heat_addition",      F.col("Th") - F.col("Tc"))
        df = df.withColumn("thermal_position_index",   F.col("mean_tp") / (F.col("mean_tt") + 1e-5))
        df = df.withColumn("tp_stability_4h",          F.avg("tp_imbalance").over(self.win_4h))
        df = df.withColumn("thermal_drop_trend_4h",    F.avg("furnace_thermal_drop").over(self.win_4h))
        df = df.withColumn("Th_roll_mean_4h",          F.avg("Th").over(self.win_4h))
        df = df.withColumn("Tc_roll_mean_4h",          F.avg("Tc").over(self.win_4h))
        return df

    def generate_pressure_features(self, df: DataFrame) -> DataFrame:
        logger.info("Pressure Features...")
        df = df.withColumn("permeability_index",          F.col("Fb") / (F.col("dP") + 1e-5))
        df = df.withColumn("pressure_distribution_ratio", F.col("dPu") / (F.col("dPl") + 1e-5))
        df = df.withColumn("dP_roll_mean_4h",             F.avg("dP").over(self.win_4h))
        df = df.withColumn("dP_roll_std_4h",              F.stddev("dP").over(self.win_4h))
        return df

    def generate_gas_chemistry_features(self, df: DataFrame) -> DataFrame:
        logger.info("Gas Features...")
        df = df.withColumn("specific_co2_generation", F.col("CO2") / (F.col("Fb") + 1e-5))
        df = df.withColumn("gas_reduction_ratio",      F.col("CO2") / (F.col("H2") + 1e-5))
        df = df.withColumn("co2_volatility_1h",        F.stddev("CO2").over(self.win_1h))
        df = df.withColumn("co2_volatility_4h",        F.stddev("CO2").over(self.win_4h))
        df = df.withColumn("co2_gen_trend_4h",         F.avg("specific_co2_generation").over(self.win_4h))
        df = df.withColumn("H2_roll_mean_4h",          F.avg("H2").over(self.win_4h))
        return df

    def generate_interaction_features(self, df: DataFrame) -> DataFrame:
        logger.info("Interaction Features...")
        df = df.withColumn("blast_heat_index",         F.col("Fb") * F.col("Th"))
        df = df.withColumn("oxygen_enrichment_ratio",  F.col("Fo") / (F.col("Fb") + 1e-5))
        df = df.withColumn("permeability_k_index",     F.col("Fb") / (F.col("dP") + 1e-5))
        df = df.withColumn("specific_pressure_drop",   F.col("dP") / (F.col("Fb") + 1e-5))
        df = df.withColumn("thermal_burden_balance",   F.col("blast_heat_index") / (F.col("R") + 1e-5))
        df = df.withColumn("gas_utilization_proxy",    F.col("CO2") / (F.col("Fb") + 1e-5))
        df = df.withColumn("fb_th_ratio",              F.col("Fb") / (F.col("Th") + 1e-5))
        df = df.withColumn("co2_h2_product",           F.col("CO2") * F.col("H2"))
        df = df.withColumn("dp_r_product",             F.col("dP") * F.col("R"))
        return df

    def generate_velocity_features(self, df: DataFrame) -> DataFrame:
        logger.info("Velocity & Acceleration...")
        df = df.withColumn("ts_long", F.col("si_dt").cast("long"))
        df = df.withColumn(
            "delta_t_hours",
            (F.col("ts_long") - F.lag("ts_long", 1).over(self.w_lag)) / 3600.0
        )
        for col_name, src in [("si", "Si"), ("th", "Th"), ("fb", "Fb"), ("co2", "CO2")]:
            df = df.withColumn(
                f"{col_name}_velocity",
                (F.col(src) - F.lag(src, 1).over(self.w_lag)) / (F.col("delta_t_hours") + 1e-5)
            )
        for col_name in ["si", "th"]:
            df = df.withColumn(
                f"{col_name}_acceleration",
                (F.col(f"{col_name}_velocity") - F.lag(f"{col_name}_velocity", 1).over(self.w_lag))
                / (F.col("delta_t_hours") + 1e-5)
            )
        return df.drop("ts_long")

    def generate_diff_features(self, df: DataFrame) -> DataFrame:
        logger.info("Diff Features: Değişim oranları...")
        key_cols = ["Si", "Fb", "Th", "Tc", "dP", "CO2", "H2", "R"]
        for col in key_cols:
            if col in df.columns:
                df = df.withColumn(
                    f"{col}_diff_1h",
                    F.col(col) - F.lag(col, 1).over(self.w_lag)
                )
                df = df.withColumn(
                    f"{col}_diff_pct",
                    (F.col(col) - F.lag(col, 1).over(self.w_lag))
                    / (F.lag(col, 1).over(self.w_lag) + 1e-5)
                )
        return df

    def finalize_ml_dataset(self, df: DataFrame) -> DataFrame:
        logger.info("Finalization: Multi-horizon hedefler oluşturuluyor (zaman bazlı)...")

        w_lead = Window.partitionBy(F.lit(0)).orderBy("si_dt")

        df = df.withColumn("next_si_dt", F.lead("si_dt", 1).over(w_lead))
        df = df.withColumn(
            "hours_to_next_cast",
            (F.col("next_si_dt").cast("long") - F.col("si_dt").cast("long")) / 3600.0
        )

        df = df.withColumn("_ts_long", F.col("si_dt").cast("long"))

        TOL_SEC = int(1.5 * 3600)

        for h in FORECAST_HORIZONS:
            target_sec = h * 3600
            win_h = (
                Window.partitionBy(F.lit(0)).orderBy("_ts_long")
                .rangeBetween(target_sec - TOL_SEC, target_sec + TOL_SEC)
            )
            df = df.withColumn(f"target_Si_{h}h", F.last("Si", ignorenulls=True).over(win_h))

            win_h_dt = (
                Window.partitionBy(F.lit(0)).orderBy("_ts_long")
                .rangeBetween(target_sec - TOL_SEC, target_sec + TOL_SEC)
            )
            df = df.withColumn(f"_last_ts_{h}h", F.last("_ts_long", ignorenulls=True).over(win_h_dt))
            df = df.withColumn(
                f"target_Si_{h}h",
                F.when(
                    F.abs(F.col(f"_last_ts_{h}h") - (F.col("_ts_long") + target_sec)) <= TOL_SEC,
                    F.col(f"target_Si_{h}h")
                ).otherwise(F.lit(None))
            )
            df = df.drop(f"_last_ts_{h}h")

        df = df.drop("_ts_long")

        any_target_valid = None
        for h in FORECAST_HORIZONS:
            cond = F.col(f"target_Si_{h}h").isNotNull()
            any_target_valid = cond if any_target_valid is None else (any_target_valid | cond)

        df = df.filter(any_target_valid)

        essential_features = ["si_velocity", "th_acceleration", "permeability_index"]
        df = df.dropna(subset=essential_features)

        target_cols = [f"target_Si_{h}h" for h in FORECAST_HORIZONS]
        logger.info(f"Multi-horizon dataset hazır. Hedef kolonları: {target_cols}")
        return df

    def run_gold_pipeline(self):
        logger.info("GOLD PIPELINE ATEŞLENDİ! (Multi-Horizon)")

        silver_schema = self._get_silver_read_schema()

        if not os.path.exists(self.silver_path):
            logger.error(f"Silver dizini yok: {self.silver_path}")
            return

        files = [f for f in os.listdir(self.silver_path) if f.endswith(".parquet")]
        if not files:
            logger.error("Silver'da parquet yok")
            return

        raw_silver = self.spark.read.schema(silver_schema).parquet(self.silver_path)
        n = raw_silver.count()
        logger.info(f"Silver'dan {n} satır okundu.")
        if n == 0:
            logger.error("Silver boş!")
            return

        logger.info(f"Silver kolonları: {raw_silver.columns}")

        gold_df = (
            self.deduplicate_data(raw_silver)
            .transform(self.clean_time_gaps)
            .transform(self.generate_time_features)
            .transform(self.generate_lag_features)
            .transform(self.generate_thermal_features)
            .transform(self.generate_pressure_features)
            .transform(self.generate_gas_chemistry_features)
            .transform(self.generate_interaction_features)
            .transform(self.generate_velocity_features)
            .transform(self.generate_diff_features)
            .transform(self.finalize_ml_dataset)
        )

        gold_df.write.mode("overwrite").parquet(self.gold_path)

        result = self.spark.read.parquet(self.gold_path)
        final_count = result.count()
        final_cols  = len(result.columns)
        target_cols = [c for c in result.columns if c.startswith("target_Si_")]
        logger.info(
            f"GOLD HAZIR! {final_count} satır, {final_cols} kolon → {self.gold_path}\n"
            f"  Hedef kolonları: {target_cols}"
        )