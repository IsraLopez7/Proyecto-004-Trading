# src/data_loader.py
"""
Carga y limpieza de datos de QQQ con splits cronológicos
"""
import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Carga configuración desde config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def download_data(symbol, start_date, end_date):
    """
    Descarga datos OHLCV de Yahoo Finance
    
    Args:
        symbol: Ticker del activo
        start_date: Fecha inicial
        end_date: Fecha final
    
    Returns:
        DataFrame con OHLCV
    """
    logger.info(f"Descargando datos de {symbol} desde {start_date} hasta {end_date}")
    
    # Descargar datos
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    # Renombrar columnas para consistencia
    data = data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Adj Close': 'adj_close'
    })
    
    # Reset index para tener Date como columna
    data = data.reset_index()
    data = data.rename(columns={'Date': 'date'})
    
    return data

def clean_data(df):
    """
    Limpieza de datos: manejo de faltantes y valores anómalos
    
    Args:
        df: DataFrame crudo
    
    Returns:
        DataFrame limpio
    """
    logger.info(f"Limpiando datos. Shape inicial: {df.shape}")
    
    # Eliminar filas con valores nulos
    df = df.dropna()
    
    # Forward fill para cualquier gap menor
    df = df.fillna(method='ffill')
    
    # Verificar datos anómalos (volumen = 0)
    df = df[df['volume'] > 0]
    
    # Ordenar por fecha
    df = df.sort_values('date')
    
    logger.info(f"Shape final después de limpieza: {df.shape}")
    logger.info(f"Rango de fechas: {df['date'].min()} a {df['date'].max()}")
    logger.info(f"Total de días de trading: {len(df)}")
    
    return df

def create_splits(df, train_pct=0.6, test_pct=0.2):
    """
    Crea splits cronológicos sin look-ahead bias
    
    Args:
        df: DataFrame completo
        train_pct: Porcentaje para entrenamiento
        test_pct: Porcentaje para test
    
    Returns:
        Diccionario con splits e índices
    """
    n_samples = len(df)
    
    # Calcular índices de corte
    train_end = int(n_samples * train_pct)
    test_end = int(n_samples * (train_pct + test_pct))
    
    # Crear splits
    train_data = df.iloc[:train_end].copy()
    test_data = df.iloc[train_end:test_end].copy()
    val_data = df.iloc[test_end:].copy()
    
    logger.info(f"Splits creados:")
    logger.info(f"  Train: {len(train_data)} samples ({train_data['date'].min()} a {train_data['date'].max()})")
    logger.info(f"  Test:  {len(test_data)} samples ({test_data['date'].min()} a {test_data['date'].max()})")
    logger.info(f"  Val:   {len(val_data)} samples ({val_data['date'].min()} a {val_data['date'].max()})")
    
    return {
        'train': train_data,
        'test': test_data,
        'val': val_data,
        'train_idx': (0, train_end),
        'test_idx': (train_end, test_end),
        'val_idx': (test_end, n_samples)
    }

def main():
    """Pipeline principal de carga de datos"""
    # Cargar configuración
    config = load_config()
    
    # Crear directorios
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Descargar datos
    df = download_data(
        config['asset'],
        config['data_start'],
        config['data_end']
    )
    
    # Limpiar datos
    df = clean_data(df)
    
    # Guardar datos crudos
    raw_path = f"data/raw/{config['asset']}.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Datos guardados en {raw_path}")
    
    # Crear y guardar splits
    splits = create_splits(
        df,
        config['train_split'],
        config['test_split']
    )
    
    # Guardar información de splits
    splits_info = {
        'train_idx': splits['train_idx'],
        'test_idx': splits['test_idx'],
        'val_idx': splits['val_idx']
    }
    
    import pickle
    with open('data/processed/splits.pkl', 'wb') as f:
        pickle.dump(splits_info, f)
    
    logger.info("Pipeline de datos completado exitosamente")

if __name__ == "__main__":
    main()