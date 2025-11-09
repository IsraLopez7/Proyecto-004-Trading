# src/features.py
"""
Calcula 20+ features técnicos sobre datos OHLCV:
- Momentum: SMA, EMA, RSI, MACD, ROC, Stochastic, CCI
- Volatilidad: ATR, Bollinger Bands (% b, bandwidth), volatilidad rolling
- Volumen: OBV, MFI, z-score de volumen
- Otros: ADX, lags de rendimientos
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_sma(df, windows):
    """Simple Moving Average para múltiples ventanas."""
    for w in windows:
        df[f'SMA_{w}'] = df['Close'].rolling(window=w).mean()
    return df


def calculate_ema(df, windows):
    """Exponential Moving Average."""
    for w in windows:
        df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
    return df


def calculate_rsi(df, window=14):
    """Relative Strength Index."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    """MACD (Moving Average Convergence Divergence)."""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df


def calculate_atr(df, window=14):
    """Average True Range (volatilidad)."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=window).mean()
    return df


def calculate_bollinger_bands(df, window=20, num_std=2):
    """Bollinger Bands: % b y bandwidth."""
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    df['BB_upper'] = sma + (std * num_std)
    df['BB_lower'] = sma - (std * num_std)
    df['BB_pct_b'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['BB_bandwidth'] = (df['BB_upper'] - df['BB_lower']) / sma
    return df


def calculate_stochastic(df, window=14):
    """Stochastic Oscillator."""
    low_min = df['Low'].rolling(window=window).min()
    high_max = df['High'].rolling(window=window).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    return df


def calculate_adx(df, window=14):
    """Average Directional Index (tendencia)."""
    # Simplificado: ADX aproximado
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    
    pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    atr = df['ATR'] if 'ATR' in df.columns else df['Close'].diff().abs().rolling(window).mean()
    
    pos_di = 100 * (pos_dm.rolling(window).mean() / atr)
    neg_di = 100 * (neg_dm.rolling(window).mean() / atr)
    
    dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
    df['ADX'] = dx.rolling(window).mean()
    return df


def calculate_cci(df, window=20):
    """Commodity Channel Index."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    df['CCI'] = (tp - sma_tp) / (0.015 * mad)
    return df


def calculate_roc(df, window=12):
    """Rate of Change."""
    df['ROC'] = ((df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)) * 100
    return df


def calculate_obv(df):
    """On-Balance Volume."""
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    return df


def calculate_mfi(df, window=14):
    """Money Flow Index (volumen + precio)."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    
    positive_mf = mf.where(tp > tp.shift(), 0).rolling(window).sum()
    negative_mf = mf.where(tp < tp.shift(), 0).rolling(window).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    df['MFI'] = mfi
    return df


def calculate_returns_lags(df, lags=[1, 2, 5, 10]):
    """Rendimientos pasados (lags)."""
    returns = df['Close'].pct_change()
    for lag in lags:
        df[f'Return_lag_{lag}'] = returns.shift(lag)
    return df


def calculate_volatility_rolling(df, windows=[10, 20]):
    """Volatilidad rolling (desviación estándar de rendimientos)."""
    returns = df['Close'].pct_change()
    for w in windows:
        df[f'Volatility_{w}'] = returns.rolling(window=w).std()
    return df


def calculate_volume_zscore(df, window=20):
    """Z-score del volumen (normalizado)."""
    mean_vol = df['Volume'].rolling(window=window).mean()
    std_vol = df['Volume'].rolling(window=window).std()
    df['Volume_zscore'] = (df['Volume'] - mean_vol) / std_vol
    return df


def calculate_all_features(df, config):
    """Calcula todas las features configuradas."""
    logger.info("Calculando features técnicos...")
    
    # Momentum
    df = calculate_sma(df, config['features']['sma_windows'])
    df = calculate_ema(df, config['features']['ema_windows'])
    df = calculate_rsi(df, config['features']['rsi_window'])
    df = calculate_macd(df, config['features']['macd_fast'], 
                       config['features']['macd_slow'], 
                       config['features']['macd_signal'])
    df = calculate_roc(df, config['features']['roc_window'])
    df = calculate_stochastic(df, config['features']['stoch_window'])
    df = calculate_cci(df, config['features']['cci_window'])
    
    # Volatilidad
    df = calculate_atr(df, config['features']['atr_window'])
    df = calculate_bollinger_bands(df, config['features']['bollinger_window'], 
                                   config['features']['bollinger_std'])
    df = calculate_volatility_rolling(df, windows=[10, 20])
    
    # Volumen
    if config['features']['obv']:
        df = calculate_obv(df)
    df = calculate_mfi(df, config['features']['mfi_window'])
    df = calculate_volume_zscore(df)
    
    # Tendencia
    df = calculate_adx(df, config['features']['adx_window'])
    
    # Lags de rendimientos
    df = calculate_returns_lags(df, lags=[1, 2, 5, 10])
    
    # Eliminar filas con NaN generados por ventanas
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f"Filas removidas por NaN en features: {initial_rows - len(df)}")
    
    return df


def main():
    """Flujo principal de feature engineering."""
    config = load_config()
    
    # Cargar datos limpios
    df = pd.read_parquet('data/processed/clean_data.parquet')
    logger.info(f"Datos cargados: {len(df)} filas")
    
    # Calcular features
    df = calculate_all_features(df, config)
    
    # Contar features creados
    original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in df.columns if col not in original_cols]
    logger.info(f"Total de features calculados: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols}")
    
    # Guardar
    output_path = config['data']['processed_path']
    df.to_parquet(output_path)
    logger.info(f"Features guardados en {output_path}")
    
    # Guardar lista de features para referencia
    with open('data/processed/feature_names.txt', 'w') as f:
        for feat in feature_cols:
            f.write(f"{feat}\n")
    
    logger.info("✅ features.py completado exitosamente")


if __name__ == "__main__":
    main()