# src/infer.py
"""
Inferencia offline: genera señales de trading desde el modelo en Staging.
Útil para backtest sin usar API.
"""

import numpy as np
import pandas as pd
import yaml
import logging
import mlflow
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_from_registry(model_name, stage='Staging'):
    """Carga modelo desde Model Registry."""
    model_uri = f"models:/{model_name}/{stage}"
    logger.info(f"Cargando modelo desde {model_uri}")
    model = mlflow.keras.load_model(model_uri)
    return model


def generate_signals(model, X, W):
    """
    Genera señales {0:long, 1:hold, 2:short} para todo el conjunto X.
    
    Returns:
        signals: array de señales
        probabilities: array de probabilidades (n_samples, 3)
    """
    # Crear ventanas
    X_windows = []
    for i in range(len(X) - W + 1):
        X_windows.append(X[i:i+W])
    
    X_windows = np.array(X_windows)
    
    # Predicción
    probs = model.predict(X_windows, verbose=0)
    signals = np.argmax(probs, axis=1)
    
    logger.info(f"Señales generadas: {len(signals)}")
    
    return signals, probs


def main():
    config = load_config()
    
    # Cargar modelo
    model_name = config['mlflow']['model_name']
    model = load_model_from_registry(model_name, stage='Staging')
    
    # Cargar datos de validación (para ejemplo)
    X_val = np.load('data/processed/X_val.npy')
    y_val = np.load('data/processed/y_val.npy')
    
    W = config['windows']['W']
    
    # Generar señales (ya viene en ventanas, así que simplemente predecimos)
    probs = model.predict(X_val, verbose=0)
    signals = np.argmax(probs, axis=1)
    
    # Guardar señales
    signals_df = pd.DataFrame({
        'signal': signals,
        'prob_long': probs[:, 0],
        'prob_hold': probs[:, 1],
        'prob_short': probs[:, 2],
        'true_label': y_val
    })
    
    signals_df.to_csv('results/signals_val.csv', index=False)
    logger.info("Señales guardadas en results/signals_val.csv")
    
    # Estadísticas
    label_names = {0: 'long', 1: 'hold', 2: 'short'}
    for i in range(3):
        count = (signals == i).sum()
        pct = (count / len(signals)) * 100
        logger.info(f"{label_names[i]}: {count} ({pct:.2f}%)")
    
    logger.info("✅ infer.py completado")


if __name__ == "__main__":
    main()