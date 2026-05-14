"""
src/api/main.py — BF Intelligence FastAPI Servisi
=================================================
Çalıştırma:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
cfg = load_config()

PATHS_CFG    = cfg.get("paths", {})
SETTINGS_CFG = cfg.get("settings", {})

SI_ALERT_HIGH = SETTINGS_CFG.get("si_alert_high", 0.70)
SI_ALERT_LOW  = SETTINGS_CFG.get("si_alert_low",  0.20)

VALID_HORIZONS = [2, 4, 6, 8]


class _PredictorHolder:
    instance = None

    @classmethod
    def load(cls):
        from src.models.predict import BFPredictor
        cls.instance = BFPredictor()
        logger.info("BFPredictor yüklendi.")

    @classmethod
    def get(cls):
        if cls.instance is None:
            raise RuntimeError("Model henüz yüklenmedi.")
        return cls.instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _PredictorHolder.load()
    except FileNotFoundError as e:
        logger.warning(f"Model yüklenemedi (ilk çalıştırma?): {e}")
    yield
    logger.info("Uygulama kapatılıyor.")


app = FastAPI(
    title="BF Intelligence API",
    description="Yüksek Fırın Silicon Tahmin Servisi",
    version="3.0.0",
    lifespan=lifespan,
)


class SensorData(BaseModel):
    horizon_hours: Literal[2, 4, 6, 8] = Field(default=8, description="Tahmin ufku (saat)")

    # Besleme
    Fb:  float = Field(..., description="Hava debisi")
    Th:  float = Field(..., description="Sıcak hava sıcaklığı")
    R:   float = Field(..., description="Cevher/kok oranı")
    Fo:  float = Field(default=21.0, description="Oksijen oranı")  # ← YENİ
    # Basınç
    dP:  float = Field(..., description="Basınç farkı")
    # Kimya
    CO2: float = Field(..., description="CO2 oranı (%)")
    H2:  float = Field(..., description="H2 oranı (%)")
    # Mevcut Silicon
    Si:  float = Field(..., description="Anlık silicon değeri")

    # Opsiyonel sensörler
    Tc:  float = Field(default=65.0,  description="Soğutma suyu sıcaklığı")
    Ph:  float = Field(default=2.1,   description="Üst basınç")
    Pc:  float = Field(default=1.8,   description="Orta basınç")
    Pt:  float = Field(default=1.2,   description="Alt basınç")
    Tt1: float = Field(default=120.0); Tt2: float = Field(default=120.0)
    Tt3: float = Field(default=120.0); Tt4: float = Field(default=120.0)
    Tp1: float = Field(default=450.0); Tp2: float = Field(default=450.0)
    Tp3: float = Field(default=450.0); Tp4: float = Field(default=450.0)
    Tp5: float = Field(default=450.0); Tp6: float = Field(default=450.0)
    Tp7: float = Field(default=450.0); Tp8: float = Field(default=450.0)
    Tp9: float = Field(default=450.0); Tp10: float = Field(default=450.0)

    class Config:
        json_schema_extra = {
            "example": {
                "horizon_hours": 4,
                "Fb": 3800.0, "Th": 1150.0, "R": 3.8, "Fo": 21.0,
                "dP": 1.45, "CO2": 18.5, "H2": 4.2, "Si": 0.45
            }
        }


class PredictionResponse(BaseModel):
    status:        str
    horizon_hours: int
    prediction:    float
    delta:         float
    alert:         str
    alert_msg:     str


@app.get("/health", tags=["ops"])
def health():
    predictor     = _PredictorHolder.instance
    model_ready   = predictor is not None
    loaded_horizons = sorted(predictor.models.keys()) if model_ready else []
    return {
        "status":           "ok" if model_ready else "degraded",
        "model_ready":      model_ready,
        "loaded_horizons":  loaded_horizons,
        "si_thresholds":    {"high": SI_ALERT_HIGH, "low": SI_ALERT_LOW},
    }


@app.post("/reload", tags=["ops"])
def reload_model():
    try:
        _PredictorHolder.load()
        loaded = sorted(_PredictorHolder.instance.models.keys())
        return {"status": "ok", "message": "Model yeniden yüklendi.", "horizons": loaded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def get_prediction(data: SensorData):
    predictor = _PredictorHolder.instance
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model henüz hazır değil.",
        )

    h = data.horizon_hours
    if h not in predictor.models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{h}h modeli yüklü değil.",
        )

    try:
        sensor_cols = {k: v for k, v in data.model_dump().items() if k != "horizon_hours"}
        df          = pd.DataFrame([sensor_cols])
        results     = predictor.predict_silicon(df, horizons=[h])
        prediction  = results[h]

        pred_val = round(float(prediction), 4)
        delta    = round(pred_val - data.Si, 4)

        if pred_val > SI_ALERT_HIGH or pred_val < SI_ALERT_LOW:
            alert, alert_msg = "RED", f"KRİTİK: {h}h sonrası Si tahmini ({pred_val}) sınırlar dışında."
        elif abs(delta) > 0.10:
            alert, alert_msg = "YELLOW", f"UYARI: {h}h içinde {delta:+.4f} sapma bekleniyor."
        else:
            alert, alert_msg = "GREEN", f"{h}h ufku için çalışma rejimi stabil."

        return PredictionResponse(
            status="success", horizon_hours=h, prediction=pred_val,
            delta=delta, alert=alert, alert_msg=alert_msg,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)