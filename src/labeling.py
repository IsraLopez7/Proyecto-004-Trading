"""
Labeling Module
Creates multiclass labels {long, short, hold} based on future returns
"""

import logging
from pathlib import Path

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


def calculate_future_returns(
    df: pd.DataFrame,
    H: int = 5
) -> pd.DataFrame:
    """
    Calculate future returns over H days.

    Args:
        df: DataFrame with price data (must have 'Close' column)
        H: Prediction horizon in days

    Returns:
        DataFrame with future_return column
    """
    # Get close prices from original data
    data_dir = Path("data/processed")

    # Load original clean data to get prices
    data_clean = pd.read_parquet(data_dir / "data_clean.parquet")

    # Align with features dataframe
    prices = data_clean.loc[df.index, 'Close']

    # Calculate future return
    future_return = prices.pct_change(periods=H).shift(-H)

    df['future_return'] = future_return

    return df


def create_labels(
    df: pd.DataFrame,
    tau: float = 0.005
) -> pd.DataFrame:
    """
    Create multiclass labels based on future returns.

    Labels:
        0 = long  (future_return > +tau)
        1 = short (future_return < -tau)
        2 = hold  (else)

    Args:
        df: DataFrame with future_return column
        tau: Threshold (e.g., 0.005 = 0.5%)

    Returns:
        DataFrame with label column
    """
    conditions = [
        df['future_return'] > tau,   # long
        df['future_return'] < -tau,  # short
    ]
    choices = [0, 1]  # long, short
    default = 2       # hold

    df['label'] = np.select(conditions, choices, default=default)

    return df


def save_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/processed"
):
    """Save labels to parquet files."""
    output_path = Path(output_dir)

    # Save only labels and future returns
    cols_to_save = ['label', 'future_return']

    train_df[cols_to_save].to_parquet(output_path / "labels_train.parquet")
    val_df[cols_to_save].to_parquet(output_path / "labels_val.parquet")
    test_df[cols_to_save].to_parquet(output_path / "labels_test.parquet")

    # Combined labels
    labels_all = pd.concat([
        train_df[cols_to_save],
        val_df[cols_to_save],
        test_df[cols_to_save]
    ])
    labels_all.to_parquet(output_path / "labels.parquet")

    logger.info(f"Saved labels to {output_dir}/")


def print_label_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: list
):
    """Print label distribution statistics."""
    logger.info("=" * 60)
    logger.info("LABEL DISTRIBUTION:")
    logger.info("=" * 60)

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = df['label'].value_counts().sort_index()
        total = len(df)

        logger.info(f"\n{name} Set ({total} samples):")
        for label, count in counts.items():
            class_name = class_names[label]
            pct = count / total * 100
            logger.info(f"  {class_name:5s} (label={label}): {count:5d} ({pct:5.1f}%)")

    logger.info("=" * 60)


def main():
    """Main execution function."""
    logger.info("Starting labeling pipeline...")

    # Load config
    config = load_config()
    H = config['labeling']['H']
    tau = config['labeling']['tau']
    class_names = config['labeling']['class_names']

    logger.info(f"Labeling params: H={H} days, tau={tau*100:.2f}%")

    # Load features
    data_dir = Path("data/processed")
    train_features = pd.read_parquet(data_dir / "features_train.parquet")
    val_features = pd.read_parquet(data_dir / "features_val.parquet")
    test_features = pd.read_parquet(data_dir / "features_test.parquet")

    # Calculate future returns
    logger.info("Calculating future returns...")
    train_df = calculate_future_returns(train_features.copy(), H)
    val_df = calculate_future_returns(val_features.copy(), H)
    test_df = calculate_future_returns(test_features.copy(), H)

    # Create labels
    logger.info("Creating labels...")
    train_df = create_labels(train_df, tau)
    val_df = create_labels(val_df, tau)
    test_df = create_labels(test_df, tau)

    # Remove rows with NaN labels (last H rows)
    train_df = train_df.dropna(subset=['label', 'future_return'])
    val_df = val_df.dropna(subset=['label', 'future_return'])
    test_df = test_df.dropna(subset=['label', 'future_return'])

    # Print distribution
    print_label_distribution(train_df, val_df, test_df, class_names)

    # Check for class imbalance
    train_counts = train_df['label'].value_counts()
    imbalance_ratio = train_counts.max() / train_counts.min()

    if imbalance_ratio > 3:
        logger.warning(f"Class imbalance detected: ratio={imbalance_ratio:.1f}")
        logger.warning("Consider using class_weight in model training")

    # Save labels
    save_labels(train_df, val_df, test_df)

    logger.info("Labeling complete!")


if __name__ == "__main__":
    main()
