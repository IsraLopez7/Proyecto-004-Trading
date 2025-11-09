# src/train.py
"""
Entrena CNN con MLflow tracking.
Loggea: params, metrics, artifacts (plots, model), y registra en Model Registry.
"""

import numpy as np
import yaml
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mlflow
import mlflow.keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from model_cnn import build_cnn_model, compile_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Seed para reproducibilidad
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data():
    """Carga tensores preprocesados."""
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    X_val = np.load('data/processed/X_val.npy')
    y_val = np.load('data/processed/y_val.npy')
    
    logger.info(f"Datos cargados: Train={X_train.shape}, Test={X_test.shape}, Val={X_val.shape}")
    
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def calculate_class_weights(y_train):
    """Calcula class weights para manejar desbalanceo."""
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {int(c): w for c, w in zip(classes, weights)}
    
    logger.info(f"Class weights: {class_weights}")
    return class_weights


def plot_training_history(history, save_path='results/training_history.png'):
    """Grafica pérdida y accuracy de train/val."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráfica de entrenamiento guardada en {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    """Grafica matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['long', 'hold', 'short'],
                yticklabels=['long', 'hold', 'short'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Validation Set)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Matriz de confusión guardada en {save_path}")


def train_model(config):
    """Flujo completo de entrenamiento con MLflow."""
    
    # Configurar MLflow
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # Cargar datos
        (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_data()
        
        # Class weights
        class_weights = calculate_class_weights(y_train)
        
        # Construir modelo
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_cnn_model(input_shape, n_classes=3, config=config['model'])
        model = compile_model(model, learning_rate=config['training']['learning_rate'])
        
        # Loggear parámetros
        params = {
            'W': config['windows']['W'],
            'H': config['labeling']['H'],
            'tau': config['labeling']['tau'],
            'n_features': X_train.shape[2],
            'batch_size': config['training']['batch_size'],
            'epochs': config['training']['epochs'],
            'learning_rate': config['training']['learning_rate'],
            'filters': config['model']['filters'],
            'kernel_sizes': config['model']['kernel_sizes'],
            'dropout': config['model']['dropout'],
            'dense_units': config['model']['dense_units'],
            'class_weights': str(class_weights),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
        mlflow.log_params(params)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config['callbacks']['early_stopping']['patience'],
                restore_best_weights=config['callbacks']['early_stopping']['restore_best_weights'],
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=config['callbacks']['reduce_lr']['factor'],
                patience=config['callbacks']['reduce_lr']['patience'],
                min_lr=config['callbacks']['reduce_lr']['min_lr'],
                verbose=1
            )
        ]
        
        # Entrenar
        logger.info("Iniciando entrenamiento...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=config['training']['batch_size'],
            epochs=config['training']['epochs'],
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Predicciones en validación
        y_val_pred_probs = model.predict(X_val)
        y_val_pred = np.argmax(y_val_pred_probs, axis=1)
        
        # Métricas en validación
        val_accuracy = np.mean(y_val_pred == y_val)
        val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
        val_f1_per_class = f1_score(y_val, y_val_pred, average=None)
        
        # Classification report
        report = classification_report(y_val, y_val_pred, 
                                      target_names=['long', 'hold', 'short'],
                                      digits=4)
        logger.info(f"\nClassification Report (Validation):\n{report}")
        
        # Loggear métricas
        metrics = {
            'val_accuracy': val_accuracy,
            'val_f1_macro': val_f1_macro,
            'val_f1_long': val_f1_per_class[0],
            'val_f1_hold': val_f1_per_class[1],
            'val_f1_short': val_f1_per_class[2],
            'train_loss_final': history.history['loss'][-1],
            'val_loss_final': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss'])
        }
        mlflow.log_metrics(metrics)
        
        # Guardar plots
        Path('results').mkdir(exist_ok=True)
        plot_training_history(history)
        plot_confusion_matrix(y_val, y_val_pred)
        
        # Loggear artifacts
        mlflow.log_artifact('results/training_history.png')
        mlflow.log_artifact('results/confusion_matrix.png')
        
        # Guardar y registrar modelo
        model_path = 'results/model.h5'
        model.save(model_path)
        
        # Loggear modelo en MLflow
        mlflow.keras.log_model(model, "model")
        
        # Registrar en Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        model_name = config['mlflow']['model_name']
        
        mlflow.register_model(model_uri, model_name)
        logger.info(f"Modelo registrado en Model Registry: {model_name}")
        
        logger.info(f"✅ Entrenamiento completado. Macro-F1: {val_f1_macro:.4f}")
        logger.info(f"Run ID: {run.info.run_id}")


def main():
    config = load_config()
    train_model(config)


if __name__ == "__main__":
    main()