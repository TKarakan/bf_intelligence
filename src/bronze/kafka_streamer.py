import pandas as pd
import json
import os


from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.utils.kafka_utils import get_producer, delivery_report


logger = get_logger(__name__)

base_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=base_dir / ".env", override=True)


try:
    cfg = load_config()
    KAFKA_CFG = cfg.get('kafka', {})
    PATHS     = cfg.get('paths', {})
    DATASETS  = cfg.get('datasets', {})
    
    
    TOPIC_SENSORS = os.getenv("KAFKA_TOPIC_NAME", "blast-furnace-data")
    YAML_BROKER   = KAFKA_CFG.get('bootstrap_servers', 'kafka:29092')
    KAFKA_BROKER  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", YAML_BROKER)
    TOPIC_SILICON = "silicon-data"
    BRONZE_DIR    = PATHS.get('raw_bronze_dir', 'data/bronze')

    print(f"--- BAĞLANTI ADRESİ: {KAFKA_BROKER} ---")
except Exception as e:
    logger.error(f"Global yapılandırma yüklenemedi: {e}")
    raise

class BlastFurnaceStreamer:
    def __init__(self):
        
        self.producer = get_producer(client_id="bf-ingestion-streamer")

    def stream_csv_to_kafka(self, file_name, topic_name, source_type):
        
        file_path = os.path.join(BRONZE_DIR, file_name)
    
        if not os.path.exists(file_path):
            logger.warning(f"Dosya bulunamadı: {file_path}. Akış atlanıyor.")
            return

        logger.info(f"!!! {file_name} -> {topic_name} akışı başlıyor...")
        
        try:
            df = pd.read_csv(file_path)
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f"!!! {file_name}"):

                data = row.to_dict()


                data['source_type'] = source_type
                
                
                if 'dt' in data:
                    data['dt'] = str(data['dt'])

                self.producer.produce(
                    topic=topic_name,
                    value=json.dumps(data).encode('utf-8'),
                    callback=delivery_report
                )
                self.producer.poll(0)

                if i % 1000 == 0:
                    self.producer.flush()
                
            self.producer.flush()
            logger.info(f"{file_name} tamamlandı. Toplam: {len(df)} satır.")

        except Exception as e:
            logger.error(f"Akış hatası ({file_name}): {e}")

    def run(self):
        
        for ds_key, files in DATASETS.items():
            logger.info(f"--- {ds_key.upper()} Grubu İşleniyor ---")
            
            if 'sensors' in files:
                self.stream_csv_to_kafka(files['sensors'], TOPIC_SENSORS, ds_key)
            
            if 'target' in files:
                self.stream_csv_to_kafka(files['target'], TOPIC_SILICON, ds_key)

