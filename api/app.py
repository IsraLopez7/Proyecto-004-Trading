# api/app.py
"""
API FastAPI para predicción de señales
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import mlflow
import numpy as np
import pandas as pd
import pickle
import yaml
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear app
app = FastAPI(
    title="CNN Trading Signal API",
    description="API para predicción de señales de trading usando CNN",
    version="1.0.0"
)

# Modelos de datos
class PredictionRequest(BaseModel):
    n_bars: int = 1
    
class PredictionResponse(BaseModel):
    signal: str
    probabilities: Dict[str, float]
    confidence: float
    metadata: Dict[str, any]
    timestamp: str

# Variables globales
MODEL = None
CONFIG = None
METADATA = None

@app.on_event("startup")
async def startup_event():
    """Carga modelo y configuración al iniciar"""
    global MODEL, CONFIG, METADATA
    
    # Cargar configuración
    with open('config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    # Cargar modelo desde MLflow
    try:
        client = mlflow.tracking.MlflowClient()
        model_name = CONFIG['mlflow']['model_name']
        
        # Obtener última versión en Staging
        model_version = client.get_latest_versions(
            model_name,
            stages=["Staging"]
        )[0]
        
        # Cargar modelo
        model_uri = f"models:/{model_name}/{model_version.version}"
        MODEL = mlflow.keras.load_model(model_uri)
        
        logger.info(f"Modelo cargado: {model_name} v{model_version.version}")
        
        # Cargar metadata
        with open('data/processed/window_metadata.pkl', 'rb') as f:
            METADATA = pickle.load(f)
            
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "CNN Trading Signal API",
        "status": "running",
        "model_loaded": MODEL is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Genera predicción de señal de trading
    
    Args:
        request: PredictionRequest con n_bars
        
    Returns:
        PredictionResponse con señal y probabilidades
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Cargar datos más recientes
        df = pd.read_parquet('data/processed/features.parquet')
        
        # Preparar ventana
        feature_cols = METADATA['feature_columns']
        window_size = METADATA['window_size']
        
        # Tomar última ventana disponible
        if len(df) < window_size:
            raise HTTPException(
                status_code=400,
                detail=f"Datos insuficientes. Se requieren {window_size} barras"
            )
        
        # Crear ventana
        window = df.iloc[-window_size:][feature_cols].values
        X = np.array([window])
        
        # Predicción
        probs = MODEL.predict(X)[0]
        signal_idx = np.argmax(probs)
        
        # Mapear a nombres
        signal_map = {0: 'hold', 1: 'long', 2: 'short'}
        signal = signal_map[signal_idx]
        
        # Crear respuesta
        response = PredictionResponse(
            signal=signal,
            probabilities={
                'hold': float(probs[0]),
                'long': float(probs[1]),
                'short': float(probs[2])
            },
            confidence=float(np.max(probs)),
            metadata={
                'model_name': CONFIG['mlflow']['model_name'],
                'window_size': window_size,
                'n_features': len(feature_cols),
                'last_date': df.iloc[-1]['date'].strftime('%Y-%m-%d')
            },
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Información del modelo actual"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    return {
        "model_name": CONFIG['mlflow']['model_name'],
        "window_size": CONFIG['window_size'],
        "horizon": CONFIG['horizon'],
        "threshold": CONFIG['threshold'],
        "n_features": METADATA['n_features'],
        "feature_columns": METADATA['feature_columns'][:10],  # Primeros 10
        "model_config": CONFIG['model']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)