from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    x = np.array(data.features).reshape(1, -1)
    pred = model.predict(x)
    return {"prediction": int(pred[0])}