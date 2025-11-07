"""
Feature Engineering Module
Creates 25+ technical indicators from OHLCV data
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ====================
# MOMENTUM INDICATORS
# ====================

def calculate_sma(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Calculate Simple Moving Averages."""
    for window in windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
    return df


def calculate_ema(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Calculate Exponential Moving Averages."""
    for window in windows:
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return df


def calculate_roc(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Calculate Rate of Change."""
    df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) /
                           df['Close'].shift(period) * 100)
    return df


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()

    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3
) -> pd.DataFrame:
    """Calculate Stochastic Oscillator."""
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()

    df[f'Stochastic_K_{k_period}'] = 100 * (
        (df['Close'] - low_min) / (high_max - low_min)
    )
    df[f'Stochastic_D_{d_period}'] = df[f'Stochastic_K_{k_period}'].rolling(
        window=d_period
    ).mean()
    return df


# ====================
# VOLATILITY INDICATORS
# ====================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Average True Range."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f'ATR_{period}'] = true_range.rolling(window=period).mean()
    return df


def calculate_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2
) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()

    df[f'BB_Upper_{period}'] = sma + (std * num_std)
    df[f'BB_Lower_{period}'] = sma - (std * num_std)
    df[f'BB_Middle_{period}'] = sma

    # Bandwidth
    df[f'BB_Bandwidth_{period}'] = (df[f'BB_Upper_{period}'] -
                                    df[f'BB_Lower_{period}']) / df[f'BB_Middle_{period}']

    # %B (position within bands)
    df[f'BB_PctB_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / (
        df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']
    )

    # Drop intermediate bands, keep only derived features
    df = df.drop(columns=[f'BB_Upper_{period}', f'BB_Lower_{period}', f'BB_Middle_{period}'])
    return df


def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate rolling volatility."""
    df[f'Volatility_{window}'] = df['Close'].pct_change().rolling(
        window=window
    ).std() * np.sqrt(252)  # Annualized
    return df


def calculate_true_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate True Range and High-Low Range."""
    df['High_Low_Range'] = df['High'] - df['Low']

    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['True_Range'] = pd.concat(
        [df['High_Low_Range'], high_close, low_close],
        axis=1
    ).max(axis=1)
    return df


def calculate_volatility_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio of short-term to long-term volatility."""
    vol_short = df['Close'].pct_change().rolling(window=10).std()
    vol_long = df['Close'].pct_change().rolling(window=50).std()
    df['Volatility_Ratio'] = vol_short / vol_long
    return df


# ====================
# VOLUME INDICATORS
# ====================

def calculate_volume_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Calculate volume-based features."""
    # Volume SMA
    df[f'Volume_SMA_{period}'] = df['Volume'].rolling(window=period).mean()

    # Volume Z-Score
    vol_mean = df['Volume'].rolling(window=period).mean()
    vol_std = df['Volume'].rolling(window=period).std()
    df['Volume_ZScore'] = (df['Volume'] - vol_mean) / vol_std

    return df


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate On-Balance Volume."""
    obv = np.where(df['Close'] > df['Close'].shift(1),
                   df['Volume'],
                   np.where(df['Close'] < df['Close'].shift(1),
                           -df['Volume'],
                           0))
    df['OBV'] = pd.Series(obv, index=df.index).cumsum()
    return df


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Money Flow Index."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    df[f'MFI_{period}'] = mfi
    return df


def calculate_volume_roc(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """Calculate Volume Rate of Change."""
    df[f'Volume_ROC_{period}'] = ((df['Volume'] - df['Volume'].shift(period)) /
                                  df['Volume'].shift(period) * 100)
    return df


# ====================
# LAG FEATURES
# ====================

def calculate_returns(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """Calculate return features with different lags."""
    for lag in lags:
        df[f'Return_{lag}d'] = df['Close'].pct_change(lag)

    # Log returns
    df['Log_Return_1d'] = np.log(df['Close'] / df['Close'].shift(1))
    return df


# ====================
# MAIN FEATURE PIPELINE
# ====================

def create_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Create all technical features.

    Args:
        df: DataFrame with OHLCV data
        config: Configuration dictionary

    Returns:
        DataFrame with features
    """
    logger.info("Creating features...")

    # Momentum indicators
    df = calculate_sma(df, config['features']['sma_windows'])
    df = calculate_ema(df, config['features']['ema_windows'])
    df = calculate_rsi(df, config['features']['rsi_period'])
    df = calculate_roc(df, config['features']['roc_period'])
    df = calculate_macd(
        df,
        config['features']['macd_fast'],
        config['features']['macd_slow'],
        config['features']['macd_signal']
    )
    df = calculate_stochastic(
        df,
        config['features']['stochastic_k'],
        config['features']['stochastic_d']
    )

    # Volatility indicators
    df = calculate_atr(df, config['features']['atr_period'])
    df = calculate_bollinger_bands(
        df,
        config['features']['bb_period'],
        config['features']['bb_std']
    )
    df = calculate_volatility(df, config['features']['volatility_window'])
    df = calculate_true_range_features(df)
    df = calculate_volatility_ratio(df)

    # Volume indicators
    df = calculate_volume_features(df, config['features']['volume_sma_period'])
    df = calculate_obv(df)
    df = calculate_mfi(df, config['features']['mfi_period'])
    df = calculate_volume_roc(df, config['features']['volume_roc_period'])

    # Return/lag features
    df = calculate_returns(df, config['features']['return_lags'])

    # Drop OHLCV columns (we only need features)
    df = df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    # Drop NaN rows (from rolling windows)
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)

    logger.info(f"Dropped {initial_rows - final_rows} rows due to NaN from rolling windows")
    logger.info(f"Created {len(df.columns)} features: {list(df.columns)}")

    return df


def normalize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple:
    """
    Normalize features using StandardScaler fit on train set only.

    Args:
        train_df: Training features
        val_df: Validation features
        test_df: Test features

    Returns:
        Tuple of (normalized_train, normalized_val, normalized_test, scaler)
    """
    logger.info("Normalizing features (fit on train only)...")

    scaler = StandardScaler()

    # Fit only on train
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    # Convert back to DataFrames
    train_norm = pd.DataFrame(train_scaled, columns=train_df.columns, index=train_df.index)
    val_norm = pd.DataFrame(val_scaled, columns=val_df.columns, index=val_df.index)
    test_norm = pd.DataFrame(test_scaled, columns=test_df.columns, index=test_df.index)

    logger.info("Normalization complete (mean=0, std=1 based on train set)")

    return train_norm, val_norm, test_norm, scaler


def save_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler,
    output_dir: str = "data/processed"
):
    """Save features and scaler."""
    import joblib

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_path / "features_train.parquet")
    val_df.to_parquet(output_path / "features_val.parquet")
    test_df.to_parquet(output_path / "features_test.parquet")

    # Save combined features
    features_all = pd.concat([train_df, val_df, test_df])
    features_all.to_parquet(output_path / "features.parquet")

    # Save scaler
    joblib.dump(scaler, output_path / "scaler.pkl")

    logger.info(f"Saved features and scaler to {output_dir}/")


def main():
    """Main execution function."""
    logger.info("Starting feature engineering pipeline...")

    # Load config
    config = load_config()

    # Load cleaned data
    data_dir = Path("data/processed")
    train_df = pd.read_parquet(data_dir / "data_train.parquet")
    val_df = pd.read_parquet(data_dir / "data_val.parquet")
    test_df = pd.read_parquet(data_dir / "data_test.parquet")

    # Create features for each split
    logger.info("Creating features for train set...")
    train_features = create_features(train_df.copy(), config)

    logger.info("Creating features for validation set...")
    val_features = create_features(val_df.copy(), config)

    logger.info("Creating features for test set...")
    test_features = create_features(test_df.copy(), config)

    # Align indices (some rows may have been dropped due to NaN)
    common_train_idx = train_features.index
    common_val_idx = val_features.index
    common_test_idx = test_features.index

    logger.info(f"Final sizes: Train={len(common_train_idx)}, "
                f"Val={len(common_val_idx)}, Test={len(common_test_idx)}")

    # Normalize features
    train_norm, val_norm, test_norm, scaler = normalize_features(
        train_features, val_features, test_features
    )

    # Save
    save_features(train_norm, val_norm, test_norm, scaler)

    logger.info("Feature engineering complete!")
    logger.info(f"\nFeature names ({len(train_norm.columns)} total):")
    for i, col in enumerate(train_norm.columns, 1):
        logger.info(f"  {i}. {col}")


if __name__ == "__main__":
    main()
