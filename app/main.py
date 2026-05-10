from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import logging

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------------
# Request counter (for metrics)
# -----------------------------
request_count = 0

# -----------------------------
# Input schema
# -----------------------------
class InputData(BaseModel):
    features: list

# -----------------------------
# Health endpoints
# -----------------------------
@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# -----------------------------
# Metrics endpoint
# -----------------------------
@app.get("/metrics")
def metrics():
    return {"total_requests": request_count}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: InputData):
    global request_count

    try:
        request_count += 1

        features = data.features

        # Validate input length
        if len(features) != 13:
            return {"error": "Expected 13 input features"}

        logger.info(f"Received input: {features}")

        # Convert to numpy array
        input_data = np.array(features).reshape(1, -1)

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        pred = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]

        # ✅ Correct confidence (based on predicted class)
        confidence = probs[int(pred)]

        logger.info(f"Prediction: {pred}, Confidence: {confidence}")

        return {
            "prediction": int(pred),
            "confidence": float(confidence)
        }

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": str(e)}