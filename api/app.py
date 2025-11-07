"""
FastAPI Application for Trading Signal Prediction
Serves predictions from best model in MLflow Model Registry
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Deep Trading MLOps API",
    description="CNN-based trading signal prediction API",
    version="1.0.0"
)

# Global variables for model and config
MODEL = None
MODEL_INFO = {}
CONFIG = {}


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    n_bars: int = Field(
        default=256,
        description="Number of bars (window size) to use for prediction"
    )
    ticker: str = Field(
        default="QQQ",
        description="Ticker symbol"
    )
    path_to_features: str = Field(
        default="data/processed/features.parquet",
        description="Path to features file"
    )


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    signal: str
    probs: Dict[str, float]
    used_model: Dict[str, str]
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    model_loaded: bool
    model_info: Optional[Dict] = None


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_from_registry(model_name: str, stage: str = "Staging"):
    """
    Load model from MLflow Model Registry.

    Args:
        model_name: Name of registered model
        stage: Stage to load (default: Staging)

    Returns:
        Tuple of (model, model_info)
    """
    try:
        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Loading model from: {model_uri}")

        model = mlflow.keras.load_model(model_uri)

        # Get model version info
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])

        if versions:
            version_info = versions[0]
            model_info = {
                'name': model_name,
                'version': version_info.version,
                'stage': stage,
                'run_id': version_info.run_id
            }
        else:
            model_info = {
                'name': model_name,
                'version': 'unknown',
                'stage': stage,
                'run_id': 'unknown'
            }

        logger.info(f"Model loaded: {model_info}")
        return model, model_info

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model and config on startup."""
    global MODEL, MODEL_INFO, CONFIG

    logger.info("Starting up API...")

    # Load config
    CONFIG = load_config()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(CONFIG['mlflow']['tracking_uri'])

    # Load model
    model_name = CONFIG['mlflow']['model_name']

    try:
        MODEL, MODEL_INFO = load_model_from_registry(model_name, stage="Staging")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")
        logger.warning("API will start but predictions will fail until model is loaded")

    logger.info("API startup complete")


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint."""
    return {
        "message": "Deep Trading MLOps API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    model_loaded = MODEL is not None

    response = {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }

    if model_loaded:
        response["model_info"] = MODEL_INFO

    return response


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict trading signal for given features.

    Args:
        request: PredictionRequest with n_bars, ticker, path_to_features

    Returns:
        PredictionResponse with signal, probabilities, model info
    """
    global MODEL, MODEL_INFO, CONFIG

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health endpoint."
        )

    try:
        # Load features
        features_path = Path(request.path_to_features)

        if not features_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Features file not found: {request.path_to_features}"
            )

        features_df = pd.read_parquet(features_path)

        # Get last n_bars
        if len(features_df) < request.n_bars:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data. Need {request.n_bars} bars, got {len(features_df)}"
            )

        last_bars = features_df.iloc[-request.n_bars:].values

        # Reshape for model input: (1, n_bars, n_features)
        X = last_bars.reshape(1, request.n_bars, -1)

        # Predict
        y_pred_probs = MODEL.predict(X, verbose=0)[0]  # Shape: (n_classes,)

        # Get class with highest probability
        class_idx = int(np.argmax(y_pred_probs))
        class_names = CONFIG['labeling']['class_names']
        signal = class_names[class_idx]

        # Create probability dict
        probs = {
            class_name: float(prob)
            for class_name, prob in zip(class_names, y_pred_probs)
        }

        # Create response
        response = PredictionResponse(
            signal=signal,
            probs=probs,
            used_model=MODEL_INFO,
            timestamp=datetime.utcnow().isoformat()
        )

        logger.info(f"Prediction: {signal} (probs: {probs})")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return {
        "model_info": MODEL_INFO,
        "config": {
            "window_size": CONFIG['model']['W'],
            "horizon": CONFIG['labeling']['H'],
            "tau": CONFIG['labeling']['tau'],
            "class_names": CONFIG['labeling']['class_names']
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
