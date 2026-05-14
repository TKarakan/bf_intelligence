from sqlalchemy import create_engine
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
import os

logger = get_logger(__name__)

def get_db_connection_string():
    cfg = load_config()
    db_cfg = cfg.get('database', {})

    
    user = os.getenv("DB_USER", db_cfg.get("user", "admin"))
    pw   = os.getenv("DB_PASS", db_cfg.get("password", "adminpassword"))
    host = os.getenv("DB_HOST", db_cfg.get("host", "postgres")) # Docker içi servis adı
    port = os.getenv("DB_PORT", db_cfg.get("port", "5432"))
    name = os.getenv("DB_NAME", db_cfg.get("dbname", "bf_intelligence"))
    
    return f"postgresql://{user}:{pw}@{host}:{port}/{name}"


def get_db_engine():
    try:
        db_url = get_db_connection_string()
        engine = create_engine(db_url)
        # Bağlantıyı test et
        with engine.connect() as conn:
            logger.info("Postgres bağlantısı başarılı.")
        return engine

    except Exception as e:
        logger.error(f"Veritabanı ihraç hatası (Bağlantı ayarlarını kontrol et): {e}")
        raise