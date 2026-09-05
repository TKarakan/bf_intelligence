import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataCleaner:
    def __init__(self):
        self.medians_ = {}
        self._is_fitted = False

    def clean_common(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ortak temizlik: dt sütununu datetime'a çevirir, mükerrerleri siler ve sıralar."""
        df = df.copy()
        if 'dt' in df.columns:
            df['dt'] = pd.to_datetime(df['dt'])
            df = df.drop_duplicates(subset=['dt']).sort_values('dt').reset_index(drop=True)
        return df

    def process_sensors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sensör verisini temizler: duplicate silinir, forward-fill ile eksikler doldurulur."""
        df = self.clean_common(df)
        df = df.ffill().bfill()
        return df

    def process_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hedef verisini temizler: duplicate silinir, tarih standardize edilir."""
        df = self.clean_common(df)
        return df

    def fit(self, df):
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns
            self.medians_ = df[numeric_cols].median().to_dict()
            self._is_fitted = True
            logger.info("Cleaner: Sayısal medyanlar öğrenildi (Fit edildi).")
            return self
        except Exception as e:
            logger.error(f"Cleaner Fit Hatası: {e}")
            raise

    def transform(self, df):
        if not self.__sklearn_is_fitted__():
            raise RuntimeError("DataCleaner henüz fit edilmedi!")
        
        try:
            df = df.copy()
            if 'dt' in df.columns:
                df['dt'] = pd.to_datetime(df['dt'])
            
            # Öğrenilen medyanlarla doldur
            df = df.fillna(self.medians_)
            df = df.ffill().drop_duplicates(subset=['dt']).sort_values('dt')
            logger.info(f"Cleaner: Veri temizlendi (Satır: {len(df)})")
            return df
        except Exception as e:
            logger.error(f"Cleaner Transform Hatası: {e}")
            raise

    def __sklearn_is_fitted__(self):
        return self._is_fitted