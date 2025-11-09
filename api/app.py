# api/app.py
"""
API FastAPI para inferencia en tiempo real.
Endpoints:
  - GET /health: health check
  - POST /predict: predicción de señal usando últimas n_bars
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
import pickle
import yaml
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="CNN Trading Signal API", version="1.0")

# Variables globales para modelo y scaler
MODEL = None
SCALER = None
CONFIG = None


def load_config(config_path='../config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_scaler():
    """Carga modelo desde Model Registry y scaler."""
    global MODEL, SCALER, CONFIG
    
    CONFIG = load_config()
    
    # Cargar modelo desde Staging
    model_name = CONFIG['mlflow']['model_name']
    model_uri = f"models:/{model_name}/Staging"
    
    logger.info(f"Cargando modelo desde {model_uri}")
    MODEL = mlflow.keras.load_model(model_uri)
    logger.info("Modelo cargado exitosamente")
    
    # Cargar scaler
    with open('../data/processed/scaler.pkl', 'rb') as f:
        SCALER = pickle.load(f)
    logger.info("Scaler cargado exitosamente")


@app.on_event("startup")
async def startup_event():
    """Carga modelo al iniciar la API."""
    load_model_and_scaler()


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": MODEL is not None}


class PredictionRequest(BaseModel):
    """Request body para predicción."""
    n_bars: int = 256  # Número de barras (igual a W)


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Realiza predicción de señal de trading.
    
    Usa las últimas n_bars de features.parquet.
    
    Returns:
        signal: 0 (long), 1 (hold), 2 (short)
        probabilities: dict con prob de cada clase
        metadata: info del modelo
    """
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Cargar últimas n_bars de features
        df = pd.read_parquet('../data/processed/features.parquet')
        
        n_bars = request.n_bars
        if len(df) < n_bars:
            raise HTTPException(
                status_code=400, 
                detail=f"No hay suficientes datos. Disponibles: {len(df)}, requeridos: {n_bars}"
            )
        
        # Últimas n_bars
        df_recent = df.tail(n_bars)
        
        # Extraer features (excluir OHLCV y label)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'label']
        feature_cols = [col for col in df_recent.columns if col not in exclude_cols]
        
        X = df_recent[feature_cols].values
        
        # Normalizar con scaler pre-entrenado
        X_norm = SCALER.transform(X)
        
        # Reshape para CNN: (1, n_bars, n_features)
        X_input = X_norm.reshape(1, n_bars, -1)
        
        # Predicción
        probs = MODEL.predict(X_input, verbose=0)[0]
        signal = int(np.argmax(probs))
        
        # Mapeo de señales
        signal_names = {0: 'long', 1: 'hold', 2: 'short'}
        
        # Response
        response = {
            "signal": signal,
            "signal_name": signal_names[signal],
            "probabilities": {
                "long": float(probs[0]),
                "hold": float(probs[1]),
                "short": float(probs[2])
            },
            "metadata": {
                "model_name": CONFIG['mlflow']['model_name'],
                "n_bars_used": n_bars,
                "last_date": str(df_recent.index[-1]),
                "n_features": len(feature_cols)
            }
        }
        
        logger.info(f"Predicción: {signal_names[signal]} (probs: {probs})")
        
        return response
    
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    config = load_config()
    uvicorn.run(app, host=config['api']['host'], port=config['api']['port'])