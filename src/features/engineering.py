from src.features.causality import CausalityAnalyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    def __init__(self, max_lag: int = 12, best_lags: dict = None, include_rolling: bool = None):
        self.max_lag = max_lag
        self.best_lags_ = best_lags or {}
        self.include_rolling = include_rolling if include_rolling is not None else (best_lags is None)
        self._is_fitted = True

    def fit(self, df, target_col='Si'):
        try:
            self.best_lags_ = CausalityAnalyzer.find_best_lags(df, target_col, self.max_lag)
            self._is_fitted = True
            self.include_rolling = True
            logger.info("Engineering: Lag analizi fit edildi.")
            return self
        except Exception as e:
            logger.error(f"Engineering Fit Hatası: {e}")
            raise

    def transform(self, df):
        if not self.__sklearn_is_fitted__():
            raise RuntimeError("FeatureEngineer henüz fit edilmedi!")
        
        try:
            df = df.copy()
            for col, lag in self.best_lags_.items():
                if lag > 0 and col in df.columns:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            
            # Rolling Window İstatistikleri (Mean & Std)
            if self.include_rolling:
                critical_sensors = ['Fb', 'Th', 'Pt']
                for col in critical_sensors:
                    if col in df.columns:
                        roll_mean = df[col].rolling(window=4).mean()
                        roll_std  = df[col].rolling(window=4).std()
                        df[f"{col}_roll_mean_4h"] = roll_mean
                        df[f"{col}_rolling_mean_4h"] = roll_mean
                        df[f"{col}_roll_std_4h"] = roll_std
                        df[f"{col}_rolling_std_4h"] = roll_std
            
            final_df = df.dropna().reset_index(drop=True)
            logger.info(f"Engineering: Özellikler eklendi. Sütun sayısı: {len(final_df.columns)}")
            return final_df
        except Exception as e:
            logger.error(f"Engineering Transform Hatası: {e}")
            raise

    def __sklearn_is_fitted__(self):
        return self._is_fitted