"""
Windows Module
Creates sliding windows for CNN from time series features
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


def create_sequences(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    window_size: int = 256
) -> tuple:
    """
    Create sliding windows from time series data.

    Args:
        features: DataFrame with features (n_samples, n_features)
        labels: DataFrame with labels
        window_size: Length of sliding window (W)

    Returns:
        Tuple of (X, y) where:
            X: array of shape (n_samples - W + 1, W, n_features)
            y: array of shape (n_samples - W + 1,)
    """
    # Align features and labels on index
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    n_samples = len(features)
    n_features = features.shape[1]

    if n_samples < window_size:
        raise ValueError(
            f"Not enough samples ({n_samples}) for window size {window_size}"
        )

    # Convert to numpy
    features_array = features.values
    labels_array = labels['label'].values

    # Create sequences
    X = []
    y = []

    for i in range(n_samples - window_size + 1):
        # Extract window
        window = features_array[i:i + window_size]

        # Label corresponds to the last timestep in window
        label = labels_array[i + window_size - 1]

        X.append(window)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Created {len(X)} sequences of shape {X.shape}")

    return X, y


def save_windows(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "data/processed"
):
    """Save windows as compressed numpy arrays."""
    output_path = Path(output_dir)

    np.savez_compressed(
        output_path / "windows_train.npz",
        X=X_train,
        y=y_train
    )
    np.savez_compressed(
        output_path / "windows_val.npz",
        X=X_val,
        y=y_val
    )
    np.savez_compressed(
        output_path / "windows_test.npz",
        X=X_test,
        y=y_test
    )

    logger.info(f"Saved windows to {output_dir}/")


def print_shapes_and_stats(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list
):
    """Print shapes and statistics."""
    logger.info("=" * 60)
    logger.info("WINDOW SHAPES:")
    logger.info("=" * 60)

    logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Val:   X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Test:  X={X_test.shape}, y={y_test.shape}")

    logger.info("\nLabel distribution in windows:")
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        logger.info(f"\n{name}:")
        for label, count in zip(unique, counts):
            class_name = class_names[int(label)]
            pct = count / total * 100
            logger.info(f"  {class_name:5s} (label={label}): {count:5d} ({pct:5.1f}%)")

    logger.info("=" * 60)


def main():
    """Main execution function."""
    logger.info("Starting window creation pipeline...")

    # Load config
    config = load_config()
    W = config['model']['W']
    class_names = config['labeling']['class_names']

    logger.info(f"Window size: W={W}")

    # Load features and labels
    data_dir = Path("data/processed")

    train_features = pd.read_parquet(data_dir / "features_train.parquet")
    val_features = pd.read_parquet(data_dir / "features_val.parquet")
    test_features = pd.read_parquet(data_dir / "features_test.parquet")

    train_labels = pd.read_parquet(data_dir / "labels_train.parquet")
    val_labels = pd.read_parquet(data_dir / "labels_val.parquet")
    test_labels = pd.read_parquet(data_dir / "labels_test.parquet")

    # Create windows
    logger.info("Creating windows for train set...")
    X_train, y_train = create_sequences(train_features, train_labels, W)

    logger.info("Creating windows for validation set...")
    X_val, y_val = create_sequences(val_features, val_labels, W)

    logger.info("Creating windows for test set...")
    X_test, y_test = create_sequences(test_features, test_labels, W)

    # Print statistics
    print_shapes_and_stats(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        class_names
    )

    # Save windows
    save_windows(X_train, y_train, X_val, y_val, X_test, y_test)

    logger.info("Window creation complete!")


if __name__ == "__main__":
    main()
