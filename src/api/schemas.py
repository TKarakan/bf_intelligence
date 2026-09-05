from pydantic import BaseModel, Field
from typing import Optional

class FurnaceInput(BaseModel):
    """
    Fırın sensörlerinden gelen anlık ham veri şeması.
    Alan isimleri veri setindeki sütunlarla birebir aynıdır.
    """
    dt: str = Field(..., example="2026-04-22 14:00:00")
    Fb: float; Ph: float; Pc: float; Tc: float; Fo: float
    dP: float; dPu: float; dPl: float; Pt: float; Th: float
    CO2: float; H2: float
    Tt1: float; Tt2: float; Tt3: float; Tt4: float
    Tp1: float; Tp2: float; Tp3: float; Tp4: float; Tp5: float
    Tp6: float; Tp7: float; Tp8: float; Tp9: float; Tp10: float
    R: float
    state_of_blast_furnace: Optional[str] = "normal run"

class PredictionOutput(BaseModel):
    """Tahmin sonucunu döndüren çıktı şeması."""
    prediction: float
    unit: str = "% Si"
    status: str