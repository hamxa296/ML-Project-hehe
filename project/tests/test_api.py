from fastapi.testclient import TestClient
from api.main import app
import pytest

client = TestClient(app)

def test_health_check():
    """Ensure the health check endpoint responds correctly depending on model state."""
    response = client.get("/health")
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        assert response.json() == {"status": "healthy", "model": "loaded"}

def test_predict_single_missing_model():
    """Ensure prediction handles model absence gracefully."""
    response = client.post("/predict", json={"TransactionAmt": 100})
    assert response.status_code in [200, 400, 503]

def test_predict_batch_missing():
    """Ensure batch prediction handles missing files properly."""
    response = client.post("/batch_predict")
    # 422 Unprocessable Entity due to missing required file parameter
    assert response.status_code == 422
