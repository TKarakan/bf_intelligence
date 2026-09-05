import mlflow.sklearn
import mlflow
from xgboost import XGBRegressor
from src.models.tuner import ModelTuner

class ModelTrainer:
    def train_and_log(self, X_train, y_train, X_val, y_val, processor_instance):
        # 1. Tuning
        tuner = ModelTuner(n_trials=10)
        best_params = tuner.run_tuning(X_train, y_train, X_val, y_val)
        mlflow.log_params(best_params)

        # 2. Fit
        model = XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        
        # 3. Log Model & Processor
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_dict(processor_instance.cleaner.medians_, "processor/medians.json")
        mlflow.log_dict(processor_instance.engineer.best_lags_, "processor/best_lags.json")
        
        return model