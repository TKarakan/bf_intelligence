from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CategoricalEncoder:
    def __init__(self, columns=None):
        self.columns = columns or ['State of blast furnace']
        self.encoders_ = {}
        self._is_fitted = False

    def fit(self, df):
        try:
            for col in self.columns:
                if col in df.columns:
                    le = LabelEncoder()
                    data = df[col].fillna('unknown').astype(str)
                    le.fit(list(data.unique()) + ['unknown'])
                    self.encoders_[col] = le
            self._is_fitted = True
            logger.info(f"Encoder: {self.columns} için sınıflar öğrenildi.")
            return self
        except Exception as e:
            logger.error(f"Encoder Fit Hatası: {e}")
            raise

    def transform(self, df):
        if not self.__sklearn_is_fitted__():
            raise RuntimeError("CategoricalEncoder henüz fit edilmedi!")
        
        try:
            df = df.copy()
            for col, le in self.encoders_.items():
                if col in df.columns:
                    df[col] = df[col].fillna('unknown').astype(str)
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
                    df[col] = le.transform(df[col])
            return df
        except Exception as e:
            logger.error(f"Encoder Transform Hatası: {e}")
            raise

    def __sklearn_is_fitted__(self):
        return self._is_fitted