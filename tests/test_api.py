import pytest
from src.api.main import health, SensorData, PredictionResponse, app

def test_health_endpoint():
    result = health()
    assert isinstance(result, dict)
    assert "status" in result
    assert "si_thresholds" in result
    assert result["si_thresholds"]["high"] > result["si_thresholds"]["low"]

def test_sensor_data_schema():
    payload = {
        "horizon_hours": 4,
        "Fb": 3800.0,
        "Th": 1150.0,
        "R": 3.8,
        "Fo": 21.0,
        "dP": 1.45,
        "CO2": 18.5,
        "H2": 4.2,
        "Si": 0.45
    }
    data = SensorData(**payload)
    assert data.horizon_hours == 4
    assert data.Fb == 3800.0
    assert data.Si == 0.45
    assert data.Fo == 21.0

def test_prediction_response_schema():
    resp = PredictionResponse(
        status="success",
        horizon_hours=4,
        prediction=0.48,
        delta=0.03,
        alert="GREEN",
        alert_msg="4h ufku için çalışma rejimi stabil."
    )
    assert resp.status == "success"
    assert resp.alert == "GREEN"
    assert resp.delta == 0.03

def test_client_if_available():
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
    except (RuntimeError, ImportError):
        # httpx yüklü değilse direct endpoint testleri zaten doğruluyor
        pass
