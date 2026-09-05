import pytest
import pandas as pd
import numpy as np
from src.data.cleaner import DataCleaner
from src.data.processor import DataProcessor

# --- FIXTURES ---

@pytest.fixture
def messy_sensor_df():
    """Kirli bir sensör verisi simüle eder"""
    return pd.DataFrame({
        "dt": ["2023-01-01 00:00:00", "2023-01-01 01:00:00", "2023-01-01 01:00:00"],
        "Fb": [3400.0, np.nan, 3500.0],
        "Tc": [80.0, 81.0, 82.0]
    })

@pytest.fixture
def target_df():
    """Hedef (Si) verisi simüle eder"""
    return pd.DataFrame({
        "dt": ["2023-01-01 00:15:00", "2023-01-01 01:50:00"],
        "Si": [0.5, 0.45]
    })

# --- TESTS ---

def test_cleaner_datetime_conversion(messy_sensor_df):
    """Cleaner'ın dt sütununu datetime objesine çevirdiğini doğrula"""
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_common(messy_sensor_df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df['dt'])

def test_cleaner_drops_duplicates(messy_sensor_df):
    """Cleaner'ın mükerrer kayıtları sildiğini doğrula"""
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_common(messy_sensor_df)
    assert len(cleaned_df) == 2

def test_cleaner_ffill_logic(messy_sensor_df):
    """Eksik verilerin (NaN) bir önceki değerle dolduğunu doğrula"""
    cleaner = DataCleaner()
    df = cleaner.process_sensors(messy_sensor_df)
    assert df["Fb"].isnull().sum() == 0
    assert df.loc[df["dt"] == "2023-01-01 01:00:00", "Fb"].values[0] == 3400.0

def test_processor_merge_asof_logic(messy_sensor_df, target_df):
    """Processor'ın Si ölçüm anına en yakın geçmişteki sensör verisini bağladığını doğrula."""
    cleaner = DataCleaner()
    s_clean = cleaner.process_sensors(messy_sensor_df)
    t_clean = cleaner.process_target(target_df)
    
    master_df = pd.merge_asof(t_clean, s_clean, on='dt', direction='backward')
    
    assert master_df.iloc[0]["Fb"] == 3400.0
    assert master_df.iloc[1]["Fb"] == 3400.0
