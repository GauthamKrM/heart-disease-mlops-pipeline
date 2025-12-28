import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import mlflow.sklearn
from contextlib import asynccontextmanager
import os

# Load model (global variable)
# In production, we might load this from a remote MLflow registry or S3 bucket.
# For this assignmet, we assume the model artifact is available locally or packaged in the image.
# We will look for the latest run in the default experiment if local, or a specific path.
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model_path = os.getenv("MODEL_PATH", "models/model")

        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            model = mlflow.sklearn.load_model(model_path)
        else:
            print(f"Model path {model_path} not found. Attempting to load from latest mlruns...")
            pass
    except Exception as e:
        print(f"Error loading model: {e}")
    yield

app = FastAPI(title="Heart Disease Prediction API", lifespan=lifespan)

# Instrument Prometheus metrics
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
        prediction = model.predict(input_df)
        # Some models return probability too, but pyfunc usually returns just predict. 
        # If the underlying model supports predict_proba, accessing it via pyfunc is tricky 
        # without unwrapping. We will stick to class prediction for the basic assignment.
        probabilities = model.predict_proba(input_df)
        # Determine the confidence
        confidence = float(max(probabilities[0]))
        return {
            "prediction": int(prediction[0]),
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
