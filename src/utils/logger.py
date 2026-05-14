import logging
import sys
import os

def get_logger(name):
    logger = logging.getLogger(name)
    
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Formatör: Zaman - İsim - Seviye - Mesaj
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 1. Konsol Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. Dosya Handler (Klasör yoksa oluşturur)
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        parts = name.split('.')
        if len(parts) > 1:
            module_name = parts[1]
        else:
            module_name = "app"

        log_filename = f"{module_name}.log"
        log_path = os.path.join(log_dir, log_filename)
            
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger