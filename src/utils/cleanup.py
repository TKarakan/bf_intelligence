"""
cleanup.py — Pipeline Cleanup Utility
--------------------------------------
--fresh flag ile çalışan idempotent cleanup.

Mount point dizinler (Docker volume) için sadece içindekileri siler,
dizin kendisini korur. Böylece 'Device or resource busy' hatası önlenir.

Kullanım:
    from src.utils.cleanup import cleanup_pipeline
    cleanup_pipeline(fresh=True)
    cleanup_pipeline(fresh=False)
"""

import os
import subprocess
from src.utils.config_loader import load_config
from src.utils.kafka_utils import get_kafka_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level config
try:
    _CFG         = load_config()
    _PATHS       = _CFG.get("paths", {})
    _KAFKA_CFG   = get_kafka_config()
except Exception as e:
    logger.error(f"Config yüklenemedi: {e}")
    raise

BRONZE_BASE  = _PATHS.get("raw_bronze_dir",    "/app/data/bronze")
SILVER_BASE  = _PATHS.get("refined_silver_dir", "/app/data/silver")
GOLD_BASE    = _PATHS.get("feature_gold_dir",   "/app/data/gold")
REPORTS_DIR  = _PATHS.get("reports_dir",        "reports")
MODELS_DIR   = _PATHS.get("models_dir",         "models")
SILVER_OUT   = f"{SILVER_BASE}/cleaned"


def _is_mount_point(path: str) -> bool:
    """
    Bir dizinin mount point olup olmadığını kontrol eder.
    Parent ile farklı device ID → mount point.
    """
    try:
        return os.stat(path).st_dev != os.stat(os.path.dirname(path) or "/").st_dev
    except Exception:
        return False


def _rm_rf(path: str):
    """
    Dizini veya mount point içeriğini siler.
    Mount point ise sadece içindekileri temizler, dizin kendisini korur.
    """
    if not os.path.exists(path):
        return

    if _is_mount_point(path):
        # Mount point: sadece içindekileri sil
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            subprocess.run(["rm", "-rf", item_path], capture_output=True)
        logger.info(f"   🗑️  Temizlendi (mount point): {path}/*")
    else:
        # Normal dizin: komple sil
        result = subprocess.run(
            ["rm", "-rf", path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"   🗑️  Silindi: {path}")
        else:
            logger.warning(f"   ⚠️  Silinemedi: {path} | {result.stderr.strip()}")


def cleanup_pipeline(fresh: bool = False):
    """
    Önceki run'dan kalan veri, checkpoint, rapor ve modelleri temizler.
    """
    if not fresh:
        logger.info("ℹ️  Cleanup atlandı (--fresh flag'i verilmedi). "
                    "Mevcut veriler korunuyor, sadece değişen partitionlar güncellenecek.")
        return

    logger.info("🧹 --fresh modu: Önceki run verileri, raporları ve modelleri temizleniyor...")

    dirs_to_clean = [
        f"{BRONZE_BASE}/checkpoints",
        f"{BRONZE_BASE}/sensors",
        f"{BRONZE_BASE}/targets",
        f"{SILVER_BASE}/checkpoints",
        SILVER_OUT,
        f"{GOLD_BASE}/feature_store",
        REPORTS_DIR,
        MODELS_DIR,
    ]

    for d in dirs_to_clean:
        _rm_rf(d)

    # Kafka consumer group'u da temizle
    try:
        subprocess.run(
            ["kafka-consumer-groups.sh", "--bootstrap-server", _KAFKA_CFG["bootstrap.servers"],
             "--delete", "--group", "BF-Ingest-to-Bronze"],
            capture_output=True, timeout=10
        )
        logger.info("   🗑️  Kafka consumer group temizlendi.")
    except Exception:
        pass

    logger.info("✅ Temizlik tamamlandı.")