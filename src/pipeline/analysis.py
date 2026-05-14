"""
analysis.py — Pipeline Analiz Katmanı
--------------------------------------
Training sonrası SHAP + furnace autopsy çalıştırır.

Kullanım:
    
    run_analysis(skip=False)  # tam analiz
    run_analysis(skip=True)   # atla
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_analysis(skip: bool = False) -> None:
    """
    Training sonrası SHAP + autopsy çalıştırır.

    Args:
        skip: True ise SHAP ve furnace_autopsy adımlarını atlar.
    """
    if skip:
        logger.info("ℹ️  --skip-analysis: Analiz katmanı atlandı.")
        return

    logger.info("🔬 Analiz katmanı başlıyor...")

    # SHAP Explainability
    try:
        from src.analysis.shap_explainer import run_shap_analysis
        run_shap_analysis()
        logger.info("✅ SHAP analizi tamamlandı.")
    except Exception as e:
        logger.warning(f"SHAP analizi başarısız (pipeline devam ediyor): {e}")

    # Furnace Autopsy
    try:
        from src.analysis.furnace_autopsy import run_autopsy
        run_autopsy()
        logger.info("✅ Fırın otopsisi tamamlandı.")
    except Exception as e:
        logger.warning(f"Fırın otopsisi başarısız (pipeline devam ediyor): {e}")