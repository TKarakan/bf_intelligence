import yaml
from pathlib import Path

def load_config(config_path):
    
    try:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Yapılandırma dosyası bulunamadı: {config_path}")
            
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
            return config
        
    except Exception as e:
        print(f"Config yükleme hatası: {e}")
        raise