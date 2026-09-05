import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import mlflow

class ModelEvaluator:
    @staticmethod
    def evaluate(model, X_test, y_test):
        preds = model.predict(X_test)
        
        metrics = {
            "mae": mean_absolute_error(y_test, preds),
            "rmse": root_mean_squared_error(y_test, preds),
            "r2": r2_score(y_test, preds)
        }
        
        # Metrikleri aktif MLflow run'ına kaydet
        for name, val in metrics.items():
            mlflow.log_metric(name, val)
            print(f"{name.upper()}: {val:.4f}")

        # Feature Importance Grafiği
        plt.figure(figsize=(10, 8))
        importances = model.feature_importances_
        feature_names = X_test.columns
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
        plt.title("Blast Furnace - Feature Importance")
        plt.tight_layout()
        
        # Grafiği yerel olarak kaydet ve MLflow artifact olarak yolla
        plot_path = "feature_importance.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        
        return metrics