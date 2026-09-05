import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CausalityAnalyzer:
    @staticmethod
    def find_best_lags(df, target_col='Si', max_lag=12):
        try:
            best_lags = {}
            numeric_df = df.select_dtypes(include=[np.number])
            if target_col not in numeric_df.columns:
                return {}
                
            feature_cols = numeric_df.columns.drop(target_col)
            for col in feature_cols:
                corrs = [abs(df[target_col].corr(df[col].shift(lag))) for lag in range(max_lag + 1)]
                best_lags[col] = int(np.argmax(corrs))
            
            logger.info("Causality: En iyi lag değerleri hesaplandı.")
            return best_lags
        except Exception as e:
            logger.error(f"Causality Hatası: {e}")
            return {}