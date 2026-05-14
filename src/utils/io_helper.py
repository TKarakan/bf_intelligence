import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)  

def load_csv(path, **kwargs):  
    try:
        path = Path(path) 
        df   = pd.read_csv(path, **kwargs) 
        logger.info(f"Yüklendi: {path.name} | Satır: {len(df)}")  
        return df 
    except Exception as e:
        logger.error(f"Yükleme hatası ({path}): {e}")  
        raise  

def load_model(path):   
    try:
        model = joblib.load(str(path)) 
        logger.info(f"Model yüklendi: {path}") 
        return model
    except Exception as e:
        logger.error(f"Model yüklenemedi: {e}")
        raise

def save_model(model, path):
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)  
        joblib.dump(model, str(path)) 
        logger.info(f"Model kaydedildi: {path}")
    except Exception as e:
        logger.error(f"Model kaydedilemedi: {e}")
        raise

def save_figure(fig_name, report_dir_path):
   
    try:
        # Klasör yolunu oluştur ve yoksa yarat
        target_path = Path(report_dir_path)
        target_path.mkdir(parents=True, exist_ok=True)
        
        full_file_path = target_path / fig_name
        
        # Kaydetme işlemi
        plt.savefig(full_file_path, bbox_inches='tight', dpi=300)
        plt.close() # Bellek yönetimi için önemli
        
        logger.info(f"Rapor görseli kaydedildi: {full_file_path}")
    except Exception as e:
        logger.error(f"Görsel kaydedilirken hata: {e}")
        raise

