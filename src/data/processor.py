import pandas as pd
from src.data.ingestion import DataIngestor
from src.data.cleaner import DataCleaner
from src.features.encoder import CategoricalEncoder
from src.features.engineering import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self):
        self.ingestor = DataIngestor()
        self.cleaner = DataCleaner()
        self.encoder = CategoricalEncoder()
        self.engineer = FeatureEngineer()

    def prepare_raw_data(self, version="first"):
        """Ham veriyi çeker, tipleri eşitler ve zaman bazlı birleştirir."""
        sensors, target = self.ingestor.fetch_dataset(version)
        
        # --- TİP VE SIRALAMA GARANTİSİ ---
        sensors['dt'] = pd.to_datetime(sensors['dt'])
        target['dt'] = pd.to_datetime(target['dt'])
        
        # merge_asof için sıralama şart
        s_df = sensors.sort_values('dt').reset_index(drop=True)
        t_df = target.sort_values('dt').reset_index(drop=True)
        
        logger.info(f"Processor: {version} verileri merge_asof ile birleştiriliyor.")
        
        # Birleştirme ve ardından oluşabilecek boş satırları (NaN) temizleme
        master_df = pd.merge_asof(t_df, s_df, on='dt', direction='backward')
        return master_df.dropna().reset_index(drop=True)

    def fit_transform(self, train_df):
        """Eğitim setinde sırasıyla tüm bileşenleri fit eder ve uygular."""
        logger.info("Processor: Full Fit-Transform başlatıldı.")
        
        # Senin kurduğun o güzel zincirleme yapı:
        train_df = self.cleaner.fit(train_df).transform(train_df)
        train_df = self.encoder.fit(train_df).transform(train_df)
        train_df = self.engineer.fit(train_df).transform(train_df)
        
        # Modelin kafası karışmasın diye 'dt'yi en son çıkarıyoruz
        if 'dt' in train_df.columns:
            train_df = train_df.drop(columns=['dt'])
            
        return train_df

    def transform(self, test_df):
        """Eğitilmiş (fitted) bileşenleri yeni veriye uygular."""
        logger.info("Processor: Sadece Transform uygulanıyor.")
        
        test_df = self.cleaner.transform(test_df)
        test_df = self.encoder.transform(test_df)
        test_df = self.engineer.transform(test_df)
        
        if 'dt' in test_df.columns:
            test_df = test_df.drop(columns=['dt'])
            
        return test_df