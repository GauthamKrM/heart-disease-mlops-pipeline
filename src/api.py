import os
import pandas as pd
import mlflow.sklearn
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager


model = None


def load_model():
    model_path = os.getenv("MODEL_PATH", "models/model")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    return mlflow.sklearn.load_model(model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = load_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model loading failed: {e}")
        model = None
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    lifespan=lifespan
)

Instrumentator().instrument(app).expose(app)


class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float


@app.get("/health")
def health_check():
    if model is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: HeartData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_df = pd.DataFrame([data.model_dump()])

        prediction = model.predict(input_df)[0]

        # Safe probability handling
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)
            confidence = float(max(probabilities[0]))
        else:
            confidence = 1.0  # fallback for models without probability

        return {
            "prediction": int(prediction),
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
