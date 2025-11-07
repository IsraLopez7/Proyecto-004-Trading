"""
Data Loader Module
Loads, cleans, and splits OHLCV data chronologically
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

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


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load CSV file with OHLCV data.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with Date index and OHLCV columns
    """
    logger.info(f"Loading data from {csv_path}")

    df = pd.read_csv(csv_path)

    # Try to find date column (case-insensitive)
    date_cols = [col for col in df.columns if col.lower() in ['date', 'datetime', 'timestamp']]

    if date_cols:
        df['Date'] = pd.to_datetime(df[date_cols[0]])
        df = df.drop(columns=date_cols)
    else:
        # Assume first column is date
        df['Date'] = pd.to_datetime(df.iloc[:, 0])
        df = df.drop(columns=df.columns[0])

    # Set date as index
    df = df.set_index('Date')
    df = df.sort_index()

    # Standardize column names
    df.columns = df.columns.str.capitalize()

    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
    logger.info(f"Columns: {list(df.columns)}")

    return df[required_cols]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OHLCV data: handle missing values, outliers, and invalid data.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")

    initial_rows = len(df)

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]

    # Remove rows with all NaN
    df = df.dropna(how='all')

    # Forward fill missing values (limited to 5 days)
    df = df.fillna(method='ffill', limit=5)

    # Drop remaining NaN
    df = df.dropna()

    # Remove invalid data (negative prices, zero volume)
    df = df[(df['Close'] > 0) & (df['Open'] > 0) &
            (df['High'] > 0) & (df['Low'] > 0)]

    # Remove rows where High < Low (invalid)
    df = df[df['High'] >= df['Low']]

    # Remove extreme outliers (price changes > 50% in one day - likely data errors)
    df['pct_change'] = df['Close'].pct_change()
    df = df[np.abs(df['pct_change']) < 0.5]
    df = df.drop(columns=['pct_change'])

    final_rows = len(df)
    removed = initial_rows - final_rows

    logger.info(f"Removed {removed} rows ({removed/initial_rows*100:.2f}%)")
    logger.info(f"Clean data: {final_rows} rows")

    return df


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create chronological train/val/test splits.

    Args:
        df: Input DataFrame
        train_ratio: Fraction for training (default: 0.6)
        val_ratio: Fraction for validation (default: 0.2)
        test_ratio: Fraction for testing (default: 0.2)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), \
        "Ratios must sum to 1.0"

    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()

    logger.info("=" * 60)
    logger.info("CHRONOLOGICAL SPLITS (no look-ahead bias):")
    logger.info(f"Train: {len(train_df)} rows ({len(train_df)/n*100:.1f}%) | "
                f"{train_df.index.min()} to {train_df.index.max()}")
    logger.info(f"Val:   {len(val_df)} rows ({len(val_df)/n*100:.1f}%) | "
                f"{val_df.index.min()} to {val_df.index.max()}")
    logger.info(f"Test:  {len(test_df)} rows ({len(test_df)/n*100:.1f}%) | "
                f"{test_df.index.min()} to {test_df.index.max()}")
    logger.info("=" * 60)

    return train_df, val_df, test_df


def save_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/processed"
):
    """Save processed data to parquet files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_path / "data_train.parquet")
    val_df.to_parquet(output_path / "data_val.parquet")
    test_df.to_parquet(output_path / "data_test.parquet")

    # Also save combined clean data
    df_clean = pd.concat([train_df, val_df, test_df])
    df_clean.to_parquet(output_path / "data_clean.parquet")

    logger.info(f"Saved processed data to {output_dir}/")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Load and clean OHLCV data")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/raw/QQQ.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load data
    df = load_csv(args.csv)

    # Clean data
    df = clean_data(df)

    # Create splits
    train_df, val_df, test_df = create_splits(
        df,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )

    # Save data
    save_data(train_df, val_df, test_df, args.output)

    logger.info("Data loading and cleaning complete!")

    # Print summary statistics
    logger.info("\nSummary Statistics (Train Set):")
    logger.info(f"\n{train_df.describe()}")


if __name__ == "__main__":
    main()
