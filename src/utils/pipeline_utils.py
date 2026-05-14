"""
pipeline_utils.py — Pipeline Yardımcıları
------------------------------------------
Bekleme, idle tespiti, arka plan/blok süreç yönetimi.
"""

import time
import glob
import os
import subprocess
from typing import List
from src.utils.logger import get_logger

logger = get_logger(__name__)

_active_procs: List[subprocess.Popen] = []


def wait_for_data(path: str, label: str, timeout: int = 180, interval: int = 10) -> bool:
    """Alt klasörleri de tarayarak Parquet dosyası arar."""
    elapsed = 0
    while elapsed < timeout:
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
            if files:
                logger.info(f"✅ {label} hazır. ({elapsed}s)")
                return True
        time.sleep(interval)
        elapsed += interval
        logger.info(f"⏳ {label} bekleniyor... ({elapsed}s / {timeout}s)")
    logger.error(f"❌ {label} {timeout}s içinde dolmadı.")
    return False


def is_idle(query, label: str = "Stream", consecutive: int = 3, interval: int = 20) -> bool:
    """
    Streaming query'nin veri işlemeyi bitirip durulduğunu tespit eder.
    lastProgress None ise stream henüz başlamamış demektir — idle sayılmaz.
    """
    idle_count = 0
    while True:
        time.sleep(interval)
        progress = query.lastProgress
        if progress:
            batch_id = progress.get("batchId", 0)
            rows     = progress.get("numInputRows", 0)
            logger.info(f"💓 {label} | Batch: {batch_id} | Son İşlenen: {rows} satır")
            idle_count = idle_count + 1 if rows == 0 else 0
        else:
            logger.info(f"⏳ {label} hazırlanıyor...")
            idle_count = 0  # ← Stream henüz başlamadı, idle sayma

        if idle_count >= consecutive:
            logger.info(f"✅ {label} duruldu.")
            return True


def run_background(cmd: List[str], label: str) -> subprocess.Popen:
    logger.info(f"🛰️  {label} arka planda başlatılıyor...")
    proc = subprocess.Popen(cmd)
    _active_procs.append(proc)
    return proc


def run_blocking(cmd: List[str], label: str):
    logger.info(f"▶️  {label} başlatılıyor...")
    subprocess.run(cmd, check=True)
    logger.info(f"✅ {label} tamamlandı.")


def terminate_all():
    for proc in _active_procs:
        try:
            proc.terminate()
        except Exception:
            pass
    _active_procs.clear()


def get_active_procs() -> List[subprocess.Popen]:
    return _active_procs