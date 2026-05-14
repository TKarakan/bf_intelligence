"""
main_pipeline.py — Pipeline Entrypoint
========================================
Sadece argparse, mod dispatch ve signal handler.

Tüm iş mantığı src/pipeline/ altındaki modüllerde:
  • modes.py        → mode_full / mode_medallion / train / ui / extract
  • orchestrator.py → run_medallion_cycle (bronze/silver/gold)
  • analysis.py     → run_analysis (SHAP + autopsy)

"""

import argparse
import sys
import time
import signal

from src.utils.logger import get_logger
from src.utils.pipeline_utils import get_active_procs, terminate_all
from src.pipeline.modes import mode_full, mode_medallion, mode_train, mode_ui, mode_extract

logger = get_logger(__name__)


def _shutdown(sig, frame):
    """Temiz çıkış sinyali — tüm arka plan süreçlerini sonlandırır."""
    logger.info("👋 Kapatılıyor...")
    terminate_all()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="BF Intelligence — Birleşik Madalyon Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
    docker exec bf_orchestrator    python -m src.pipeline.main_pipeline                                            # Tam pipeline
    docker exec bf_orchestrator    python -m src.pipeline.main_pipeline --fresh                                    # Tüm veriyi + modelleri sıfırla
    docker exec bf_orchestrator    python -m src.pipeline.main_pipeline --mode medallion                           # Sadece bronze→silver→gold
    docker exec bf_orchestrator    python -m src.pipeline.main_pipeline --mode train                               # Sadece training + analiz
    docker exec bf_orchestrator    python -m src.pipeline.main_pipeline --mode train --fresh                       # Eski modelleri/raporları sil, yeniden eğit
    docker exec bf_orchestrator    python -m src.pipeline.main_pipeline --mode train --skip-analysis
    
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "medallion", "train", "ui", "extract"],
        default="full",
        help="Çalıştırılacak mod (varsayılan: full)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="Çalışmadan önce moda özgü verileri sil. "
             "full/medallion/extract → bronze/silver/gold; "
             "train → modeller + raporlar (PNG/CSV). MLflow DB dokunulmaz.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        default=False,
        dest="skip_analysis",
        help="SHAP ve furnace_autopsy adımlarını atla.",
    )
    args = parser.parse_args()

    # Sinyal handler'ları
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("=" * 60)
    logger.info(
        f"BF PIPELINE BAŞLIYOR | "
        f"mod={args.mode} | fresh={args.fresh} | skip_analysis={args.skip_analysis}"
    )
    logger.info("=" * 60)

    if args.mode == "full":
        mode_full(fresh=args.fresh, skip_analysis=args.skip_analysis)
    elif args.mode == "medallion":
        mode_medallion(fresh=args.fresh)
    elif args.mode == "train":
        mode_train(skip_analysis=args.skip_analysis, fresh=args.fresh)
    elif args.mode == "ui":
        mode_ui()
    elif args.mode == "extract":
        mode_extract(fresh=args.fresh)

    # Arka plan süreçler varsa (UI gibi) bekle
    if get_active_procs():
        logger.info("⏳ Arka plan süreçleri çalışıyor. Çıkmak için Ctrl+C.")
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            _shutdown(None, None)


if __name__ == "__main__":
    main()