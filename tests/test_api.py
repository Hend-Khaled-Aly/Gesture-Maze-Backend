from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Gesture Maze Backend" in response.json()["message"]

def test_predict():
    # Replace with the correct number of features your model expects
    sample_input = {
        "features": [0.1] * 63 
    }
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "direction" in response.json()
    assert response.json()["direction"] in ["up", "down", "left", "right", "unknown"]
