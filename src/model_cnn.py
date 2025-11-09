# src/model_cnn.py
"""
Define arquitectura CNN simple para clasificación de señales de trading.
Entrada: (W, n_features) - secuencia de features
Salida: probabilidades {long, hold, short}
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_cnn_model(input_shape, n_classes=3, config=None):
    """
    Construye CNN para clasificación de señales.
    
    Args:
        input_shape: (W, n_features)
        n_classes: número de clases (3: long/hold/short)
        config: dict con arquitectura (filters, kernels, etc.)
    
    Returns:
        modelo Keras compilado
    """
    if config is None:
        config = load_config()['model']
    
    logger.info(f"Construyendo CNN con input_shape={input_shape}")
    
    model = keras.Sequential(name='CNN_Trading_Signal')
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # Convolutional blocks
    filters = config['filters']
    kernel_sizes = config['kernel_sizes']
    pool_sizes = config['pool_sizes']
    
    for i, (f, k, p) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
        model.add(layers.Conv1D(filters=f, kernel_size=k, activation='relu', 
                                padding='same', name=f'conv1d_{i+1}'))
        model.add(layers.MaxPooling1D(pool_size=p, name=f'pool_{i+1}'))
        model.add(layers.Dropout(config['dropout'], name=f'dropout_{i+1}'))
    
    # Flatten
    model.add(layers.Flatten(name='flatten'))
    
    # Dense layers
    dense_units = config['dense_units']
    for i, units in enumerate(dense_units):
        model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(layers.Dropout(config['dropout'], name=f'dropout_dense_{i+1}'))
    
    # Output layer (softmax para multiclase)
    model.add(layers.Dense(n_classes, activation='softmax', name='output'))
    
    logger.info(f"Modelo creado con {model.count_params()} parámetros")
    
    return model


def compile_model(model, learning_rate=0.001):
    """Compila modelo con optimizer y loss."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Modelo compilado con lr={learning_rate}")
    
    return model


if __name__ == "__main__":
    # Test de construcción
    config = load_config()
    W = config['windows']['W']
    n_features = 40  # Ejemplo
    
    model = build_cnn_model(input_shape=(W, n_features))
    model = compile_model(model)
    
    model.summary()
    
    logger.info("✅ model_cnn.py test completado")