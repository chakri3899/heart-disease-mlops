from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from typing import List

app = FastAPI()

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

logging.basicConfig(level=logging.INFO)

# Define input schema
class InputData(BaseModel):
    features: List[float]

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(input_data: InputData):
    data = input_data.features
    logging.info(f"Received input: {data}")

    data = np.array(data).reshape(1, -1)
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)

    return {
        "prediction": int(prediction[0]),
        "confidence": float(max(probability[0]))
    }