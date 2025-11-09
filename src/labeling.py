# src/labeling.py
"""
Creación de etiquetas multiclase: long, short, hold
"""
import pandas as pd
import numpy as np
import yaml
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Labeler:
    """Clase para crear etiquetas de trading"""
    
    def __init__(self, config):
        self.config = config
        self.horizon = config['horizon']
        self.threshold = config['threshold']
        
    def calculate_future_returns(self, df):
        """
        Calcula retornos futuros a H días
        
        Args:
            df: DataFrame con precios
            
        Returns:
            Series con retornos futuros
        """
        future_prices = df['close'].shift(-self.horizon)
        future_returns = (future_prices - df['close']) / df['close']
        return future_returns
    
    def create_labels(self, returns):
        """
        Crea etiquetas multiclase basadas en umbral
        
        Args:
            returns: Serie de retornos futuros
            
        Returns:
            Series con etiquetas (0: hold, 1: long, 2: short)
        """
        labels = pd.Series(index=returns.index, dtype=int)
        
        # Long: retorno > +threshold
        labels[returns > self.threshold] = 1
        
        # Short: retorno < -threshold
        labels[returns < -self.threshold] = 2
        
        # Hold: resto
        labels[(returns >= -self.threshold) & (returns <= self.threshold)] = 0
        
        return labels
    
    def calculate_class_weights(self, labels, splits):
        """
        Calcula pesos de clase para balancear el dataset
        Solo usa datos de entrenamiento
        
        Args:
            labels: Series con etiquetas
            splits: Diccionario con índices de splits
            
        Returns:
            Diccionario con pesos por clase
        """
        # Usar solo datos de train
        train_start, train_end = splits['train_idx']
        train_labels = labels.iloc[train_start:train_end]
        
        # Contar clases
        class_counts = train_labels.value_counts()
        total_samples = len(train_labels)
        n_classes = len(class_counts)
        
        # Calcular pesos inversamente proporcionales a frecuencia
        class_weights = {}
        for class_id in class_counts.index:
            class_weights[class_id] = total_samples / (n_classes * class_counts[class_id])
        
        logger.info(f"Distribución de clases en train:")
        logger.info(f"  Hold (0):  {class_counts.get(0, 0)} samples ({class_counts.get(0, 0)/total_samples*100:.1f}%)")
        logger.info(f"  Long (1):  {class_counts.get(1, 0)} samples ({class_counts.get(1, 0)/total_samples*100:.1f}%)")
        logger.info(f"  Short (2): {class_counts.get(2, 0)} samples ({class_counts.get(2, 0)/total_samples*100:.1f}%)")
        logger.info(f"Class weights: {class_weights}")
        
        return class_weights
    
    def process_labels(self):
        """Pipeline principal de etiquetado"""
        # Cargar datos con features
        df = pd.read_parquet('data/processed/features.parquet')
        
        # Cargar splits
        with open('data/processed/splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        
        logger.info(f"Creando etiquetas con horizon={self.horizon}, threshold={self.threshold}")
        
        # Calcular retornos futuros
        future_returns = self.calculate_future_returns(df)
        
        # Crear etiquetas
        labels = self.create_labels(future_returns)
        
        # Agregar a dataframe
        df['future_return'] = future_returns
        df['label'] = labels
        
        # Eliminar últimas H filas (no tienen etiqueta válida)
        df = df[:-self.horizon].copy()
        
        # Calcular class weights
        class_weights = self.calculate_class_weights(labels, splits)
        
        # Guardar datos etiquetados
        df.to_parquet('data/processed/labeled_features.parquet', index=False)
        
        # Guardar class weights
        with open('data/processed/class_weights.pkl', 'wb') as f:
            pickle.dump(class_weights, f)
        
        logger.info(f"Etiquetas creadas y guardadas")
        logger.info(f"Shape final: {df.shape}")
        
        return df, class_weights

def main():
    """Ejecutar etiquetado"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    labeler = Labeler(config)
    labeler.process_labels()

if __name__ == "__main__":
    main()