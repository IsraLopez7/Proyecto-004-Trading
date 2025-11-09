# src/model_cnn.py
"""
Definición del modelo CNN para predicción de señales
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_cnn_model(input_shape, n_classes=3, config=None):
    """
    Crea modelo CNN para clasificación de series temporales
    
    Args:
        input_shape: Tupla (window_size, n_features)
        n_classes: Número de clases (3: hold, long, short)
        config: Diccionario de configuración
        
    Returns:
        Modelo Keras compilado
    """
    if config is None:
        config = {
            'filters': 64,
            'kernel_size': 3,
            'dropout': 0.3,
            'learning_rate': 0.001
        }
    
    model = keras.Sequential([
        # Primera capa convolucional
        layers.Conv1D(
            filters=config['filters'],
            kernel_size=config['kernel_size'],
            activation='relu',
            padding='same',
            input_shape=input_shape
        ),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(config['dropout']),
        
        # Segunda capa convolucional
        layers.Conv1D(
            filters=config['filters'] * 2,
            kernel_size=config['kernel_size'],
            activation='relu',
            padding='same'
        ),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(config['dropout']),
        
        # Tercera capa convolucional
        layers.Conv1D(
            filters=config['filters'],
            kernel_size=config['kernel_size'],
            activation='relu',
            padding='same'
        ),
        layers.GlobalAveragePooling1D(),
        
        # Capas densas
        layers.Dense(128, activation='relu'),
        layers.Dropout(config['dropout']),
        layers.Dense(64, activation='relu'),
        layers.Dropout(config['dropout']),
        
        # Capa de salida
        layers.Dense(n_classes, activation='softmax')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_callbacks(patience=10, min_delta=0.001):
    """
    Crea callbacks para entrenamiento
    
    Args:
        patience: Paciencia para early stopping
        min_delta: Cambio mínimo para mejora
        
    Returns:
        Lista de callbacks
    """
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks

def calculate_metrics(y_true, y_pred, class_names=['hold', 'long', 'short']):
    """
    Calcula métricas de clasificación
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        class_names: Nombres de las clases
        
    Returns:
        Diccionario con métricas
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    # Métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # F1 por clase
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Report completo
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'f1_hold': f1_per_class[0],
        'f1_long': f1_per_class[1],
        'f1_short': f1_per_class[2],
        'classification_report': report
    }
    
    return metrics