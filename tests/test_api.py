from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api import app, HeartData, PredictionResponse

client = TestClient(app)

def test_health_check_no_model():
    """Test health check when model is not loaded."""
    with patch("api.model", None):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "unhealthy", "reason": "Model not loaded"}

def test_health_check_with_model():
    """Test health check when model is loaded."""
    with patch("api.model", MagicMock()):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

def test_predict_no_model():
    """Test prediction endpoint when model is missing."""
    data = {
        "age": 60, "sex": 1, "cp": 0, "trestbps": 140, "chol": 200, "fbs": 0,
        "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.0, "slope": 1,
        "ca": 0, "thal": 2
    }
    with patch("api.model", None):
        response = client.post("/predict", json=data)
        assert response.status_code == 503
        assert response.json()["detail"] == "Model not loaded"

def test_predict_success(mock_pipeline):
    """Test successful prediction."""
    data = {
        "age": 60, "sex": 1, "cp": 0, "trestbps": 140, "chol": 200, "fbs": 0,
        "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.0, "slope": 1,
        "ca": 0, "thal": 2
    }
    
    with patch("api.model", mock_pipeline):
        response = client.post("/predict", json=data)
        assert response.status_code == 200
        resp_json = response.json()
        assert "prediction" in resp_json
        assert "confidence" in resp_json
        assert isinstance(resp_json["prediction"], int)
        assert isinstance(resp_json["confidence"], float)

def test_predict_validation_error():
    """Test missing fields."""
    data = {
        "age": 60 # missing other fields
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 422
