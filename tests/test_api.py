import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_prediction():
    response = client.post("/predict", json={"features":[5.1,3.5,1.4,0.2]})
    assert response.status_code == 200
    assert "prediction" in response.json()