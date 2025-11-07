"""
CNN Model Module
Defines 1D CNN architecture for time series classification
"""

import logging
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_cnn_model(
    input_shape: tuple,
    n_classes: int = 3,
    filters: List[int] = [64, 128, 256],
    kernel_size: int = 3,
    pool_size: int = 2,
    dropout: float = 0.3,
    dense_units: List[int] = [128, 64],
    learning_rate: float = 0.001,
    seed: int = 42
) -> keras.Model:
    """
    Build 1D CNN model for time series classification.

    Architecture:
        - Multiple Conv1D + MaxPooling blocks
        - Global Average Pooling
        - Dense layers with dropout
        - Softmax output

    Args:
        input_shape: (window_size, n_features)
        n_classes: Number of output classes (default: 3 for long/short/hold)
        filters: List of filter sizes for Conv1D layers
        kernel_size: Kernel size for convolutions
        pool_size: Pool size for MaxPooling
        dropout: Dropout rate
        dense_units: List of units for dense layers
        learning_rate: Learning rate for Adam optimizer
        seed: Random seed for reproducibility

    Returns:
        Compiled Keras model
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = keras.Sequential(name="CNN_Trading_Model")

    # Input layer
    model.add(layers.Input(shape=input_shape, name="input"))

    # Convolutional blocks
    for i, n_filters in enumerate(filters):
        model.add(layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            name=f'conv1d_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
        model.add(layers.MaxPooling1D(
            pool_size=pool_size,
            name=f'max_pool_{i+1}'
        ))
        model.add(layers.Dropout(dropout, name=f'dropout_conv_{i+1}'))

    # Global pooling
    model.add(layers.GlobalAveragePooling1D(name='global_avg_pool'))

    # Dense layers
    for i, units in enumerate(dense_units):
        model.add(layers.Dense(
            units=units,
            activation='relu',
            name=f'dense_{i+1}'
        ))
        model.add(layers.Dropout(dropout, name=f'dropout_dense_{i+1}'))

    # Output layer
    model.add(layers.Dense(
        units=n_classes,
        activation='softmax',
        name='output'
    ))

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("Model built successfully")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Output classes: {n_classes}")

    return model


def compute_class_weights(y: np.ndarray) -> dict:
    """
    Compute class weights to handle imbalanced datasets.

    Args:
        y: Array of labels

    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )

    class_weights = {i: w for i, w in zip(classes, weights)}

    logger.info("Class weights computed:")
    for cls, weight in class_weights.items():
        logger.info(f"  Class {cls}: {weight:.3f}")

    return class_weights


def get_callbacks(
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-5
) -> List:
    """
    Create training callbacks.

    Args:
        early_stopping_patience: Patience for early stopping
        reduce_lr_patience: Patience for reducing learning rate
        reduce_lr_factor: Factor by which to reduce LR
        min_lr: Minimum learning rate

    Returns:
        List of callbacks
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        )
    ]

    return callbacks


def print_model_summary(model: keras.Model):
    """Print model architecture summary."""
    logger.info("\n" + "=" * 60)
    logger.info("MODEL ARCHITECTURE:")
    logger.info("=" * 60)

    # Use string buffer to capture summary
    from io import StringIO
    import sys

    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()

    model.summary()

    sys.stdout = old_stdout
    summary_str = buffer.getvalue()

    logger.info(summary_str)
    logger.info("=" * 60)


if __name__ == "__main__":
    # Test model building
    logger.info("Testing CNN model building...")

    # Mock input shape
    window_size = 256
    n_features = 25
    input_shape = (window_size, n_features)

    # Build model
    model = build_cnn_model(
        input_shape=input_shape,
        n_classes=3,
        filters=[64, 128, 256],
        kernel_size=3,
        pool_size=2,
        dropout=0.3,
        dense_units=[128, 64],
        learning_rate=0.001
    )

    # Print summary
    print_model_summary(model)

    # Test with random data
    X_test = np.random.randn(10, window_size, n_features)
    y_pred = model.predict(X_test, verbose=0)

    logger.info(f"\nTest prediction shape: {y_pred.shape}")
    logger.info(f"Test prediction (first sample): {y_pred[0]}")
    logger.info("Model test passed!")
