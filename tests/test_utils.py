import pytest
import pandas as pd
from pathlib import Path
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.io_helper import load_csv, save_data
from src.utils.exporter import export_processed_data

# --- FIXTURES ---
@pytest.fixture
def sample_df():
    return pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

# --- TESTS ---

def test_config_loading():
    """Config dosyalarının yüklendiğini doğrula"""
    cfg = load_config()
    assert "paths" in cfg
    assert "project" in cfg
    assert "blast_furnace" in cfg

def test_logger_functionality():
    """Logger'ın nesne oluşturduğunu doğrula"""
    logger = get_logger("test_module")
    assert logger.name == "test_module"

def test_io_helper_save(sample_df, tmp_path):
    """Veri kaydetme işlemini test et"""
    test_path = tmp_path / "io_test.csv"
    save_data(sample_df, test_path)
    assert Path(test_path).exists()
    loaded = pd.read_csv(test_path)
    assert len(loaded) == 2

def test_exporter_logic(sample_df):
    """Exporter'ın zaman damgalı dosya üretme mantığını test et"""
    path = export_processed_data(sample_df, "smoke_test")
    assert Path(path).exists()
    assert "smoke_test" in str(path)
