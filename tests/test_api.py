
from fastapi.testclient import TestClient
from src.api.main import app

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

def test_predict_batch():
    with TestClient(app) as client:
        response = client.post(
            "/predict_batch",
            json={
                "comments": [
                    "Amazing video!",
                    "Terrible content!",
                    "This is a video."
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3

def test_empty_batch():
    with TestClient(app) as client:
        response = client.post(
            "/predict_batch",
            json={"comments": []}
        )
        assert response.status_code == 422

def test_invalid_data():
    with TestClient(app) as client:
        response = client.post(
            "/predict_batch",
            json={"comments": "not a list"}
        )
        assert response.status_code == 422

def test_missing_field():
    with TestClient(app) as client:
        response = client.post(
            "/predict_batch",
            json={}
        )
        assert response.status_code == 422
