import mlflow.sklearn
import pandas as pd

class FurnacePredictor:
    """FastAPI'nin kullanacağı tahmin arayüzü."""
    def __init__(self, model_uri, processor_instance):
        self.model = mlflow.sklearn.load_model(model_uri)
        self.processor = processor_instance # Önceden eğitilmiş (fitted) processor

    def predict(self, raw_input_dict):
        """
        Ham veri (JSON/Dict) gelir -> Processor'da fitted kurallarla transform edilir -> Tahmin döner.
        """
        df_input = pd.DataFrame([raw_input_dict])
        
        # Eğitimde öğrenilen kuralları (Medyan, Lag, Encode) uygula
        processed_input = self.processor.transform(df_input)
        
        # Tahmin yap
        prediction = self.model.predict(processed_input.drop(columns=['dt'], errors='ignore'))
        return float(prediction[0])