import os
import pandas as pd
from src.utils.config_parser import load_config
from src.utils.io_helper import load_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataIngestor:
    def __init__(self, config_path="config/paths.yaml"):
        """Config dosyasını yükler ve ham veri dizinini belirler."""
        self.config = load_config(config_path)
        self.raw_dir = self.config["paths"]["raw_dir"]

    def fetch_dataset(self, version="first"):
        """
        Sensör ve hedef verilerini çeker, sütun isimlerini standardize eder.
        """
        try:
            info = self.config["datasets"][version]
            
            # Dosya yollarını işletim sistemine uygun şekilde birleştir
            sensor_path = os.path.join(self.raw_dir, info['sensors'])
            target_path = os.path.join(self.raw_dir, info['target'])
            
            # Verileri yükle
            sensors = load_csv(sensor_path)
            target = load_csv(target_path)
            
            # --- KRİTİK DÜZELTME: SÜTUN İSİMLERİNİ TEMİZLE ---
            # 1. Başındaki ve sonundaki boşlukları sil (strip)
            # 2. Hepsini küçük harfe çevir (lower)
            # Böylece 'DT', ' dt' veya 'Dt' her zaman 'dt' olur.
            sensors.columns = [str(c).strip().lower() for c in sensors.columns]
            target.columns = [str(c).strip().lower() for c in target.columns]
            
            logger.info(f"Ingestion: {version} veri seti yüklendi ve {len(sensors.columns)} sütun standardize edildi.")
            
            # dt sütunu var mı kontrol et (Debug için)
            if 'dt' not in sensors.columns or 'dt' not in target.columns:
                logger.error(f"HATA: 'dt' sütunu bulunamadı! Mevcut sütunlar: {sensors.columns.tolist()}")
                raise KeyError("Veri setinde 'dt' (zaman damgası) sütunu eksik.")

            return sensors, target
            
        except Exception as e:
            logger.error(f"Ingestion Hatası: {e}")
            raise