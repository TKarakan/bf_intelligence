"""
modes.py — Pipeline Çalışma Modları
------------------------------------
mode_full     : Tam pipeline (extract → bronze → silver → gold → train → analysis → ui)
mode_medallion: Sadece bronze → silver → gold (veri katmanları, train yok)
mode_train    : Sadece training + analiz
mode_ui       : Sadece Streamlit UI
mode_extract  : Sadece Excel extraction + Kafka streamer
"""

import os
import sys
from src.utils.spark_utils import get_spark_session
from src.utils.pipeline_utils import run_background, terminate_all
from src.utils.cleanup import cleanup_pipeline
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.pipeline.orchestrator import run_medallion_cycle
from src.pipeline.analysis import run_analysis

logger = get_logger(__name__)


def _cleanup_train_artifacts():
    """Önceki eğitim artefaktlarını (modeller, raporlar) siler. MLflow DB dokunulmaz."""
    cfg = load_config()
    paths_cfg = cfg.get("paths", {})
    model_dir = paths_cfg.get("models_dir")
    reports_dir = paths_cfg.get("reports_dir", "reports")

    # 1) Modeller
    if model_dir and os.path.exists(model_dir):
        removed = 0
        for f in os.listdir(model_dir):
            if f.startswith("bf_model") and f.endswith(".joblib"):
                os.remove(os.path.join(model_dir, f))
                removed += 1
        if removed:
            logger.info(f"🗑️  {removed} eski model silindi: {model_dir}")

    # 2) Raporlar (PNG + CSV) — sadece train/analysis çıktıları
    if os.path.exists(reports_dir):
        removed = 0
        for f in os.listdir(reports_dir):
            if f.endswith((".png", ".csv")):
                os.remove(os.path.join(reports_dir, f))
                removed += 1
        if removed:
            logger.info(f"🗑️  {removed} eski rapor silindi: {reports_dir}")

    logger.info("✅ Train artefaktları temizlendi. Fresh eğitim başlıyor...")


def mode_full(fresh: bool = False, skip_analysis: bool = False):
    """Tam pipeline: extract → bronze → silver → gold → train → analysis → ui."""
    if fresh:
        _cleanup_train_artifacts()

    spark = get_spark_session(app_name="BF-Orchestrator")
    try:
        # [1-5] Veri katmanları
        run_medallion_cycle(spark, fresh=fresh)

        # [6] Training
        from src.models.train import run_training
        logger.info("[6] Model training başlatılıyor...")
        all_metrics = run_training()

        # Her horizon için ayrı log
        for h, m in sorted(all_metrics.items()):
            logger.info(
                f"[6] [{h}h] Training TAMAM | "
                f"MAE: {m['mae']:.4f} | R²: {m['r2']:.4f} | "
                f"İyileştirme: %{m['improvement_pct']:.2f}"
            )

        # [7] Analiz (opsiyonel)
        run_analysis(skip=skip_analysis)

        # [8] UI
        run_background(
            [sys.executable, "-m", "streamlit", "run", "src/ui/app.py", "--server.port", "8501"],
            "Streamlit UI"
        )
        logger.info("🎉 Sistem hazır! → http://localhost:8501")
    finally:
        spark.stop()


def mode_medallion(fresh: bool = False):
    """
    Sadece veri katmanları: extract → bronze → silver → gold.
    Training, analiz ve UI çalıştırılmaz.
    """
    spark = get_spark_session(app_name="BF-Medallion-Only")
    try:
        run_medallion_cycle(spark, fresh=fresh)
        logger.info("🎉 Medallion katmanları tamamlandı! (Train/Analysis/UI atlandı)")
    finally:
        spark.stop()


def mode_train(skip_analysis: bool = False, fresh: bool = False):
    """
    Sadece training + analiz.
    fresh=True → önceki modelleri ve raporları (PNG/CSV) siler. MLflow DB kalır.
    """
    if fresh:
        _cleanup_train_artifacts()

    from src.models.train import run_training
    all_metrics = run_training()
    for h, m in sorted(all_metrics.items()):
        logger.info(f"[{h}h] Training tamamlandı | MAE: {m['mae']:.4f}")
    run_analysis(skip=skip_analysis)


def mode_ui():
    """Sadece Streamlit UI başlatır."""
    run_background(
        [sys.executable, "-m", "streamlit", "run", "src/ui/app.py", "--server.port", "8501"],
        "Streamlit UI"
    )


def mode_extract(fresh: bool = False):
    """Sadece Bronze extraction + Kafka streamer'ı çalıştırır."""
    cleanup_pipeline(fresh=fresh)
    from src.bronze.excel_extractor import extract_excel_sheets
    extract_excel_sheets()
    from src.bronze.kafka_streamer import BlastFurnaceStreamer
    BlastFurnaceStreamer().run()
    logger.info("✅ Extract tamamlandı.")