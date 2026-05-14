import pandas as pd
import os
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

try:
    cfg = load_config()
    PATHS = cfg.get('paths', {})
    DATASETS = cfg.get('datasets', {})
except Exception as e:
    logger.error(f"Yapılandırma yüklenirken kritik hata: {e}")
    raise

def extract_excel_sheets():
    try:
        logger.info("Veri ayrıştırma işlemi (Extraction) başlıyor...")
        
        
        source_dir = PATHS.get('raw_source_dir', 'data/raw/source')
        bronze_dir = PATHS.get('bronze_dir', 'data/bronze')
        
        os.makedirs(bronze_dir, exist_ok=True)

        files_to_process = {
            "1_blast_furnace_data_first_dataset.xlsx": "first",
            "2_blast_furnace_data_second_dataset.xlsx": "second"
        }

        for file_name, ds_key in files_to_process.items():
            full_path = os.path.join(source_dir, file_name)
            
            if not os.path.exists(full_path):
                logger.warning(f"Kaynak dosya bulunamadı: {full_path}. Atlanıyor...")
                continue
                
            logger.info(f"Ayrıştırılıyor: {file_name}")
            xlsx = pd.read_excel(full_path, sheet_name=None)
            
            for sheet_name, df in xlsx.items():
                s_name_lower = sheet_name.lower().strip()

                if ds_key == "second" and "symbols" in s_name_lower:
                    logger.info(f"Referans dosyadaki ({file_name}) mükerrer symbols sayfası atlanıyor.")
                    continue

                if "data" in s_name_lower:
                    new_name = DATASETS.get(ds_key, {}).get('sensors')
                elif "si" in s_name_lower:
                    new_name = DATASETS.get(ds_key, {}).get('target')
                elif "symbols" in s_name_lower:
                    new_name = DATASETS.get(ds_key, {}).get('symbols')
                else:
                    continue       
                if new_name:
                    output_file_path = os.path.join(bronze_dir, new_name)
                    df.to_csv(output_file_path, encoding="utf-8-sig", index=False)
                    logger.info(f"Standartlaştırıldı: {new_name}")
                else:
                    logger.warning(f"Config içinde {ds_key} için isim tanımı bulunamadı.")

    except Exception as e:
        logger.error(f"Extraction sürecinde beklenmedik hata: {e}", exc_info=True)      
