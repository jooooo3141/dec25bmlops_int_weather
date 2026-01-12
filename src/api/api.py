"""
FastAPI for rain prediction in Australia

Endpoints:
- POST /train: Train the model
- POST /predict: Make predictions

Usage:
    python src/api/main.py
"""

from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import subprocess
from pathlib import Path

app = FastAPI(title="Rain Prediction API")

MODEL_PATH = Path("models/xgboost_model.pkl")
model = None


# Load model when API starts
@app.on_event("startup")
async def load_model():
    """Load model at startup"""
    global model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {MODEL_PATH}")
    except:
        print("No model found. Please train first.")


# Endpoint 1: Train model
@app.post("/train")
async def train():
    """
    Train the XGBoost model
    
    Runs: python src/models/train_model.py
    """
    global model
    
    try:
        print("Starting training...")
        
        # Run training script
        result = subprocess.run(
            ["python", "src/models/train_model.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception("Training failed")
        
        # Reload model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        return {
            "status": "success",
            "message": "Model trained successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 2: Make prediction
@app.post("/predict")
async def predict(data: dict):
    """
    Make a prediction
    
    Input: Dictionary with 110 features
    Output: Prediction (Rain/No Rain) and probability
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return {
            "prediction": int(prediction),
            "label": "Rain" if prediction == 1 else "No Rain",
            "probability": float(probability)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Endpoint Health Check
@app.get("/health")
async def health():
    """Check if API is working"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }



# Endpoint API information
@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Rain Prediction API",
        "endpoints": {
            "train": "POST /train",
            "predict": "POST /predict",
            "health": "GET /health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
