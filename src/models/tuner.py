import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelTuner:
    """Modeller için en uygun hiperparametreleri (Tuning) Optuna ile bulur."""
    def __init__(self, n_trials=20):
        self.n_trials = n_trials

    def objective(self, trial, X_train, y_train, X_val, y_val):
        # XGBoost için arama uzayı
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'random_state': 42
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    def run_tuning(self, X_train, y_train, X_val, y_val):
        logger.info(f"Tuning başlatıldı: {self.n_trials} deneme yapılacak.")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=self.n_trials)
        
        logger.info(f"En iyi parametreler bulundu: {study.best_params}")
        return study.best_params