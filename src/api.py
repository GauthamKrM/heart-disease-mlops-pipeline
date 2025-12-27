import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import os

app = FastAPI(title="Heart Disease Prediction API")

# Instrument Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Load model (global variable)
# In production, we might load this from a remote MLflow registry or S3 bucket.
# For this assignmet, we assume the model artifact is available locally or packaged in the image.
# We will look for the latest run in the default experiment if local, or a specific path.
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Load from local mlruns or a specific path set by env var
        # For simplicity in this assignment's docker flow, we might copy the model to a specific dir.
        # Here we attempt to load from the 'models/' directory if it exists (container pattern),
        # otherwise we might fail or look for mlruns (local dev pattern).
        
        model_path = os.getenv("MODEL_PATH", "models/model") 
        # Check if directory exists, else maybe we are running locally and have runs
        if os.path.isdir(model_path):
            print(f"Loading model from {model_path}...")
            model = mlflow.pyfunc.load_model(model_path)
        else:
            print(f"Model path {model_path} not found. Attempting to load from latest mlruns...")
            # This is a bit hacky for production but fine for assignment dev flow
            # To make this robust, we should explicitly pass the model URI.
            # fallback: try to find a run. Since we can't easily search runs without tracking uri setup,
            # we will rely on the user/docker to place the model in 'models/model'.
            pass
    except Exception as e:
        print(f"Error loading model: {e}")

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

@app.get("/health")
def health_check():
    if model is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: HeartData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_df)
        # Some models return probability too, but pyfunc usually returns just predict. 
        # If the underlying model supports predict_proba, accessing it via pyfunc is tricky 
        # without unwrapping. We will stick to class prediction for the basic assignment.
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
