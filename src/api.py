import os
import pandas as pd
import mlflow.sklearn
import uvicorn

# FastAPI and related imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Prometheus instrumentation for monitoring
from prometheus_fastapi_instrumentator import Instrumentator

# For managing application startup and shutdown lifecycle
from contextlib import asynccontextmanager

# Global variable to store the loaded ML model
model = None


def load_model():
    """
    Load the trained ML model from the path specified in the environment variable.
    Defaults to 'models/model' if MODEL_PATH is not set.
    """
    model_path = os.getenv("MODEL_PATH", "models/model")

    # Validate that the model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Load model using MLflow
    return mlflow.sklearn.load_model(model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler.
    Loads the model at application startup and keeps it in memory.
    """
    global model
    try:
        model = load_model()
        print("Model loaded successfully.")
    except Exception as e:
        # If model loading fails, API will still run but prediction endpoint will be disabled
        print(f"Model loading failed: {e}")
        model = None

    # Application runs while control is yielded    
    yield

# Initialize FastAPI application with metadata and lifespan handler
app = FastAPI(
    title="Heart Disease Prediction API",
    lifespan=lifespan
)

# Enable Prometheus metrics endpoint for monitoring
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
    """
    Output schema for prediction response.
    """
    prediction: int
    confidence: float


@app.get("/health")
def health_check():
    if model is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: HeartData):

    # If model is not loaded, return service unavailable
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:

        # Convert input data to DataFrame (required by sklearn models)
        input_df = pd.DataFrame([data.model_dump()])
        
        # Perform prediction
        prediction = model.predict(input_df)[0]

        # Safely extract confidence score if model supports probability prediction
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)
            confidence = float(max(probabilities[0]))
        else:
            
            # Fallback confidence for models without predict_proba
            confidence = 1.0  # fallback for models without probability

        return {
            "prediction": int(prediction),
            "confidence": confidence
        }

    except Exception as e:
        # Catch unexpected runtime errors and return HTTP 500
        raise HTTPException(status_code=500, detail=str(e))

# Entry point for running the app locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
