import pytest
import pandas as pd
import numpy as np
from src.features.causality import CausalityAnalyzer
from src.features.engineering import FeatureEngineer

# --- FIXTURES ---

@pytest.fixture
def master_table():
    """Processor'dan çıkmış gibi birleştirilmiş (merged) veri simüle eder"""
    np.random.seed(42)
    rows = 20
    return pd.DataFrame({
        "dt": pd.date_range("2023-01-01", periods=rows, freq="h"),
        "Si": np.random.uniform(0.3, 0.7, rows),
        "Fb": np.random.uniform(3000, 4000, rows),
        "Th": np.random.uniform(1000, 1200, rows),
        "Pt": np.random.uniform(1.0, 1.5, rows)
    })

# --- TESTS ---

def test_causality_best_lags_output(master_table):
    """Nedensellik analizinin her sensör için bir lag değeri bulduğunu doğrula"""
    analyzer = CausalityAnalyzer()
    best_lags = analyzer.find_best_lags(master_table, target_col='Si', max_lag=3)
    
    expected_cols = ['Fb', 'Th', 'Pt']
    assert all(col in best_lags for col in expected_cols)
    assert all(isinstance(val, (int, np.integer)) for val in best_lags.values())

def test_feature_engineer_lag_creation(master_table):
    """Belirlenen lag değerlerine göre yeni sütunların oluştuğunu doğrula"""
    lags = {'Fb': 2}
    engineer = FeatureEngineer(best_lags=lags)
    df_transformed = engineer.transform(master_table)
    
    assert "Fb_lag_2" in df_transformed.columns
    assert len(df_transformed) == len(master_table) - 2

def test_feature_engineer_rolling_stats(master_table):
    """Kritik sensörler için hareketli istatistiklerin hesaplandığını doğrula"""
    engineer = FeatureEngineer()
    df_transformed = engineer.transform(master_table)
    
    assert "Fb_rolling_mean_4h" in df_transformed.columns
    assert "Th_rolling_std_4h" in df_transformed.columns
    assert len(df_transformed) <= len(master_table) - 3

def test_no_future_leakage(master_table):
    """Lagged feature'ların gelecekteki veriyi almadığını doğrula."""
    lags = {'Fb': 1}
    engineer = FeatureEngineer(best_lags=lags)
    df_transformed = engineer.transform(master_table)
    
    sample_idx = 5
    lagged_val = df_transformed.iloc[sample_idx]["Fb_lag_1"]
    
    current_dt = df_transformed.iloc[sample_idx]["dt"]
    original_prev_val = master_table[master_table["dt"] < current_dt].iloc[-1]["Fb"]
    
    assert lagged_val == original_prev_val
