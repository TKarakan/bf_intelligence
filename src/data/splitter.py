from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataSplitter:
    """Veriyi zaman serisi kurallarına göre (kronolojik) böler."""
    def __init__(self, test_size=0.2):
        self.test_size = test_size

    def split(self, df):
        try:
            # Verinin zamana göre sıralı olduğundan emin oluyoruz
            df = df.sort_values('dt').reset_index(drop=True)
            
            # Bölme noktasını hesapla
            split_idx = int(len(df) * (1 - self.test_size))
            
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            logger.info(f"Splitter: Veri bölündü. Train: {len(train_df)}, Test: {len(test_df)}")
            logger.info(f"Zaman Aralığı: {train_df['dt'].min()} -> {test_df['dt'].max()}")
            
            return train_df, test_df
        except Exception as e:
            logger.error(f"Splitter Hatası: {e}")
            raise