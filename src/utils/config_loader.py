import yaml
from pathlib import Path
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, TimestampType
)


def load_config():
    base_dir = Path(__file__).resolve().parent.parent.parent
    config_dir = base_dir / "config"

    # Yüklenecek tüm konfigürasyon dosyaları
    config_files = ["paths.yaml", "settings.yaml", "schemas.yaml"]
    full_config = {}

    for file_name in config_files:
        file_path = config_dir / file_name
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    # Dosya içeriğini ana sözlüğe birleştiriyoruz
                    full_config.update(config_data)
        else:
            print(f" Uyarı: {file_name} bulunamadı!")

    return full_config




# ── PySpark Type Mapping ──
_TYPE_MAP = {
    "string":    StringType(),
    "double":    DoubleType(),
    "timestamp": TimestampType(),
}


def _map_type(type_str: str):
    return _TYPE_MAP.get(type_str.lower(), StringType())


# ── Schema Builders  ──

def get_sensor_schema(config: dict = None) -> StructType:
    """Bronze Kafka ingest için SENSOR_SCHEMA üretir."""
    cfg = config or load_config()
    bf = cfg.get("blast_furnace", {})

    fields = []

    # common.dt → StringType (ham JSON'dan gelir)
    common = bf.get("common", {})
    if "dt" in common:
        fields.append(StructField("dt", StringType(), True))

    # besleme
    for col, info in bf.get("besleme", {}).get("columns", {}).items():
        fields.append(StructField(col, _map_type(info["type"]), True))

    # gaz_dinamigi
    for col, info in bf.get("gaz_dinamigi", {}).get("columns", {}).items():
        fields.append(StructField(col, _map_type(info["type"]), True))

    # kimya
    for col, info in bf.get("kimya", {}).get("columns", {}).items():
        fields.append(StructField(col, _map_type(info["type"]), True))

    # sicakliklar — tt_cols
    sicaklik = bf.get("sicakliklar", {})
    for col in sicaklik.get("tt_cols", []):
        fields.append(StructField(col, _map_type(sicaklik.get("type", "double")), True))

    # sicakliklar — tp_cols
    for col in sicaklik.get("tp_cols", []):
        fields.append(StructField(col, _map_type(sicaklik.get("type", "double")), True))

    return StructType(fields)


def get_target_schema(config: dict = None) -> StructType:
    """Bronze Kafka ingest için TARGET_SCHEMA üretir."""
    cfg = config or load_config()
    bf = cfg.get("blast_furnace", {})

    fields = []

    common = bf.get("common", {})
    if "dt" in common:
        fields.append(StructField("dt", StringType(), True))

    for col, info in bf.get("target", {}).items():
        fields.append(StructField(col, _map_type(info["type"]), True))

    return StructType(fields)


def get_silver_schema(config: dict = None) -> StructType:
    """Silver/Gold için timestamp'li schema. dt → si_dt."""
    cfg = config or load_config()
    bf = cfg.get("blast_furnace", {})

    fields = [
        StructField("si_dt", TimestampType(), True),
        StructField("Si",    DoubleType(),    True),
    ]

    for col, info in bf.get("besleme", {}).get("columns", {}).items():
        fields.append(StructField(col, _map_type(info["type"]), True))

    for col, info in bf.get("gaz_dinamigi", {}).get("columns", {}).items():
        fields.append(StructField(col, _map_type(info["type"]), True))

    for col, info in bf.get("kimya", {}).get("columns", {}).items():
        fields.append(StructField(col, _map_type(info["type"]), True))

    sicaklik = bf.get("sicakliklar", {})
    for col in sicaklik.get("tt_cols", []):
        fields.append(StructField(col, _map_type(sicaklik.get("type", "double")), True))
    for col in sicaklik.get("tp_cols", []):
        fields.append(StructField(col, _map_type(sicaklik.get("type", "double")), True))

    return StructType(fields)