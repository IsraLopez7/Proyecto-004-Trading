# src/data_loader.py
"""
Carga datos crudos de QQQ, limpia valores faltantes, 
y genera splits cronológicos (60/20/20) sin look-ahead.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    """Carga archivo de configuración YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_raw_data(csv_path):
    """
    Carga CSV con columnas: Date, Open, High, Low, Close, Volume.
    Retorna DataFrame indexado por fecha.
    """
    logger.info(f"Cargando datos desde {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()  # Asegurar orden cronológico
    logger.info(f"Datos cargados: {len(df)} filas desde {df.index[0]} hasta {df.index[-1]}")
    return df


def clean_data(df):
    """
    Manejo de valores faltantes con forward-fill.
    Remueve filas con NaN al inicio si existen.
    """
    logger.info(f"Valores faltantes antes de limpieza: {df.isnull().sum().sum()}")
    
    # Forward fill para valores faltantes
    df = df.fillna(method='ffill')
    
    # Eliminar filas con NaN restantes (inicio del dataset)
    df = df.dropna()
    
    logger.info(f"Valores faltantes después de limpieza: {df.isnull().sum().sum()}")
    logger.info(f"Filas finales: {len(df)}")
    
    return df


def create_splits(df, train_pct=0.60, test_pct=0.20, val_pct=0.20):
    """
    Divide datos cronológicamente: 60% train, 20% test, 20% val.
    Retorna índices de corte.
    """
    n = len(df)
    train_end = int(n * train_pct)
    test_end = train_end + int(n * test_pct)
    
    train_idx = df.index[:train_end]
    test_idx = df.index[train_end:test_end]
    val_idx = df.index[test_end:]
    
    logger.info(f"Train: {len(train_idx)} filas ({train_idx[0]} a {train_idx[-1]})")
    logger.info(f"Test:  {len(test_idx)} filas ({test_idx[0]} a {test_idx[-1]})")
    logger.info(f"Val:   {len(val_idx)} filas ({val_idx[0]} a {val_idx[-1]})")
    
    # Guardar splits en archivo para referencia
    splits_info = {
        'train': {'start': str(train_idx[0]), 'end': str(train_idx[-1]), 'size': len(train_idx)},
        'test': {'start': str(test_idx[0]), 'end': str(test_idx[-1]), 'size': len(test_idx)},
        'val': {'start': str(val_idx[0]), 'end': str(val_idx[-1]), 'size': len(val_idx)}
    }
    
    splits_df = pd.DataFrame(splits_info).T
    splits_df.to_csv('data/processed/splits_info.csv')
    logger.info("Información de splits guardada en data/processed/splits_info.csv")
    
    return train_idx, test_idx, val_idx


def main():
    """Flujo principal de carga y limpieza de datos."""
    config = load_config()
    
    # Cargar datos crudos
    raw_path = config['data']['raw_path']
    df = load_raw_data(raw_path)
    
    # Limpieza
    df = clean_data(df)
    
    # Verificar que tenemos columnas OHLCV
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en el CSV: {missing_cols}")
    
    # Crear splits cronológicos
    train_pct = config['splits']['train']
    test_pct = config['splits']['test']
    val_pct = config['splits']['val']
    
    create_splits(df, train_pct, test_pct, val_pct)
    
    # Guardar datos limpios (para siguiente paso)
    output_path = 'data/processed/clean_data.parquet'
    df.to_parquet(output_path)
    logger.info(f"Datos limpios guardados en {output_path}")
    
    logger.info("✅ data_loader.py completado exitosamente")


if __name__ == "__main__":
    main()