from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
import src.utils.io_helper as io


logger = get_logger(__name__)

# Tek bir noktadan config yüklemesi
try:
    cfg      = load_config()
    PATHS    = cfg.get('paths',    {})
    SETTINGS = cfg.get('settings', {}) 
    DB_CFG   = cfg.get('database', {})  
except Exception as e:
    logger.warning(f"Config yüklenemedi, varsayılan değerler kullanılacak: {e}")
    PATHS, SETTINGS, DB_CFG = {}, {}, {}

def _now():
    return datetime.now().strftime("%Y%m%d_%H%M")

def _get_project_name():
    return SETTINGS.get("project", {}).get("name","bf_intelligence")

def _get_path(key: str, default:str):
    path = Path(PATHS.get(key, default))
    path.mkdir(parents=True, exist_ok = True)
    return path

def export_inference_results(df, experiment_id): #Modelin ürettiği tahminleri PowerBI için saklasın.
    
    try:
        timestamp = _now()
        results_dir = Path("outputs/results")
        filename = f"results_{experiment_id}_{timestamp}.csv"
        export_path = results_dir / filename
        
        io.save_data(df, export_path, format="csv")
        logger.info(f"Analiz sonuçları dışarı aktarıldı: {filename}")
        return export_path
    except Exception as e:
        logger.error(f"Inference ihraç hatası: {e}")
        raise