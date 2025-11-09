# src/train.py
"""
Entrenamiento del modelo CNN con MLflow tracking
"""
import numpy as np
import pandas as pd
import yaml
import pickle
import logging
import mlflow
import mlflow.keras
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from model_cnn import create_cnn_model, get_callbacks, calculate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

class ModelTrainer:
    """Clase para entrenar y registrar modelos con MLflow"""
    
    def __init__(self, config):
        self.config = config
        self.model_config = config['model']
        
        # Configurar MLflow
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
    def load_data(self):
        """Carga datos preprocesados"""
        X_train = np.load('data/processed/X_train.npy')
        y_train = np.load('data/processed/y_train.npy')
        X_test = np.load('data/processed/X_test.npy')
        y_test = np.load('data/processed/y_test.npy')
        X_val = np.load('data/processed/X_val.npy')
        y_val = np.load('data/processed/y_val.npy')
        
        with open('data/processed/class_weights.pkl', 'rb') as f:
            class_weights = pickle.load(f)
        
        logger.info(f"Datos cargados:")
        logger.info(f"  X_train shape: {X_train.shape}")
        logger.info(f"  X_test shape: {X_test.shape}")
        logger.info(f"  X_val shape: {X_val.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_val': X_val,
            'y_val': y_val,
            'class_weights': class_weights
        }
    
    def plot_training_history(self, history):
        """Genera gráficos de entrenamiento"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Train')
        axes[0].plot(history.history['val_loss'], label='Validation')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(history.history['accuracy'], label='Train')
        axes[1].plot(history.history['val_accuracy'], label='Validation')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=100)
        plt.close()
        
        return 'training_history.png'
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Genera matriz de confusión"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=['Hold', 'Long', 'Short'],
            yticklabels=['Hold', 'Long', 'Short'],
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100)
        plt.close()
        
        return 'confusion_matrix.png'
    
    def train(self):
        """Pipeline principal de entrenamiento"""
        # Cargar datos
        data = self.load_data()
        
        # Crear modelo
        input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
        
        with mlflow.start_run() as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            
            # Log parámetros
            mlflow.log_param("window_size", self.config['window_size'])
            mlflow.log_param("horizon", self.config['horizon'])
            mlflow.log_param("threshold", self.config['threshold'])
            mlflow.log_param("filters", self.model_config['filters'])
            mlflow.log_param("kernel_size", self.model_config['kernel_size'])
            mlflow.log_param("dropout", self.model_config['dropout'])
            mlflow.log_param("learning_rate", self.model_config['learning_rate'])
            mlflow.log_param("batch_size", self.model_config['batch_size'])
            mlflow.log_param("epochs", self.model_config['epochs'])
            
            # Log class weights
            for class_id, weight in data['class_weights'].items():
                mlflow.log_param(f"class_weight_{class_id}", weight)
            
            # Crear y entrenar modelo
            model = create_cnn_model(input_shape, n_classes=3, config=self.model_config)
            
            # Callbacks
            callbacks = get_callbacks(patience=self.model_config['patience'])
            
            # Entrenar
            logger.info("Iniciando entrenamiento...")
            history = model.fit(
                data['X_train'], data['y_train'],
                validation_data=(data['X_val'], data['y_val']),
                epochs=self.model_config['epochs'],
                batch_size=self.model_config['batch_size'],
                class_weight=data['class_weights'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Predicciones en validación
            y_pred_val = model.predict(data['X_val']).argmax(axis=1)
            
            # Calcular métricas
            metrics = calculate_metrics(data['y_val'], y_pred_val)
            
            # Log métricas
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("macro_f1", metrics['macro_f1'])
            mlflow.log_metric("weighted_f1", metrics['weighted_f1'])
            mlflow.log_metric("f1_hold", metrics['f1_hold'])
            mlflow.log_metric("f1_long", metrics['f1_long'])
            mlflow.log_metric("f1_short", metrics['f1_short'])
            mlflow.log_metric("final_epoch", len(history.history['loss']))
            
            # Generar y log gráficos
            history_plot = self.plot_training_history(history)
            mlflow.log_artifact(history_plot)
            
            cm_plot = self.plot_confusion_matrix(data['y_val'], y_pred_val)
            mlflow.log_artifact(cm_plot)
            
            # Guardar modelo
            model_path = f"models/cnn_model_{run.info.run_id}"
            model.save(model_path)
            
            # Registrar modelo en MLflow
            mlflow.keras.log_model(
                model,
                "model",
                registered_model_name=self.config['mlflow']['model_name']
            )
            
            logger.info(f"Entrenamiento completado")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
            logger.info(f"  Run ID: {run.info.run_id}")
            
            return run.info.run_id, metrics

def main():
    """Ejecutar entrenamiento"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = ModelTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()