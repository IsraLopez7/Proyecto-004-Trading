# src/windows.py
"""
Crea tensores 3D (samples, W, n_features) para entrenar la CNN.
W = tamaño de ventana (ej: 256 días).
Cada muestra es una "foto" de W días consecutivos.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from sklearn.preprocessing import StandardScaler
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_windows(data, labels, W):
    """
    Crea ventanas deslizantes de tamaño W.
    
    Args:
        data: array (n_samples, n_features)
        labels: array (n_samples,)
        W: tamaño de ventana
    
    Returns:
        X: (n_windows, W, n_features)
        y: (n_windows,)
    """
    X, y = [], []
    
    for i in range(len(data) - W + 1):
        X.append(data[i:i+W])
        y.append(labels[i+W-1])  # Etiqueta del último día de la ventana
    
    return np.array(X), np.array(y)


def normalize_data(train_data, test_data, val_data):
    """
    Normaliza datos: fit en train, transform en test/val.
    
    Returns:
        train_norm, test_norm, val_norm, scaler
    """
    scaler = StandardScaler()
    
    # Fit solo en train
    train_norm = scaler.fit_transform(train_data)
    
    # Transform en test y val
    test_norm = scaler.transform(test_data)
    val_norm = scaler.transform(val_data)
    
    logger.info(f"Normalización: mean={scaler.mean_[:5]}, std={scaler.scale_[:5]}")
    
    return train_norm, test_norm, val_norm, scaler


def main():
    """Flujo principal de creación de ventanas."""
    config = load_config()
    
    # Cargar features + labels
    df = pd.read_parquet(config['data']['processed_path'])
    logger.info(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
    
    # Separar features de labels
    label_col = 'label'
    feature_cols = [col for col in df.columns if col not in ['label', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    X = df[feature_cols].values
    y = df[label_col].values
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Shape inicial: X={X.shape}, y={y.shape}")
    
    # Cargar info de splits
    splits_info = pd.read_csv('data/processed/splits_info.csv', index_col=0)
    train_size = splits_info.loc['train', 'size']
    test_size = splits_info.loc['test', 'size']
    
    # Dividir cronológicamente
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_test = X[train_size:train_size+test_size]
    y_test = y[train_size:train_size+test_size]
    
    X_val = X[train_size+test_size:]
    y_val = y[train_size+test_size:]
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}, Val: {X_val.shape}")
    
    # Normalizar (fit en train)
    X_train_norm, X_test_norm, X_val_norm, scaler = normalize_data(X_train, X_test, X_val)
    
    # Guardar scaler para inferencia
    with open('data/processed/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Scaler guardado en data/processed/scaler.pkl")
    
    # Crear ventanas
    W = config['windows']['W']
    logger.info(f"Creando ventanas de tamaño W={W}")
    
    X_train_w, y_train_w = create_windows(X_train_norm, y_train, W)
    X_test_w, y_test_w = create_windows(X_test_norm, y_test, W)
    X_val_w, y_val_w = create_windows(X_val_norm, y_val, W)
    
    logger.info(f"Ventanas creadas:")
    logger.info(f"  Train: {X_train_w.shape}, {y_train_w.shape}")
    logger.info(f"  Test:  {X_test_w.shape}, {y_test_w.shape}")
    logger.info(f"  Val:   {X_val_w.shape}, {y_val_w.shape}")
    
    # Guardar tensores
    np.save('data/processed/X_train.npy', X_train_w)
    np.save('data/processed/y_train.npy', y_train_w)
    np.save('data/processed/X_test.npy', X_test_w)
    np.save('data/processed/y_test.npy', y_test_w)
    np.save('data/processed/X_val.npy', X_val_w)
    np.save('data/processed/y_val.npy', y_val_w)
    
    logger.info("Tensores guardados en data/processed/")
    logger.info("✅ windows.py completado exitosamente")


if __name__ == "__main__":
    main()