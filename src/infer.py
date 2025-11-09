# src/infer.py
"""
Inferencia offline para generar señales de trading
"""
import numpy as np
import pandas as pd
import mlflow
import yaml
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalGenerator:
    """Generador de señales usando modelo en Staging"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Carga modelo desde MLflow Registry (Staging)"""
        client = mlflow.tracking.MlflowClient()
        
        # Obtener modelo en Staging
        model_name = self.config['mlflow']['model_name']
        model_version = client.get_latest_versions(
            model_name,
            stages=["Staging"]
        )[0]
        
        # Cargar modelo
        model_uri = f"models:/{model_name}/{model_version.version}"
        self.model = mlflow.keras.load_model(model_uri)
        
        logger.info(f"Modelo cargado: {model_name} v{model_version.version}")
        
    def generate_signals(self, X):
        """
        Genera señales de trading
        
        Args:
            X: Array de features (n_samples, window_size, n_features)
            
        Returns:
            DataFrame con señales y probabilidades
        """
        # Predicciones
        probs = self.model.predict(X)
        signals = probs.argmax(axis=1)
        
        # Mapear a nombres
        signal_names = {0: 'hold', 1: 'long', 2: 'short'}
        
        # Crear DataFrame
        results = pd.DataFrame({
            'signal': [signal_names[s] for s in signals],
            'signal_num': signals,
            'prob_hold': probs[:, 0],
            'prob_long': probs[:, 1],
            'prob_short': probs[:, 2],
            'confidence': probs.max(axis=1)
        })
        
        return results
    
    def infer_latest(self, n_bars=1):
        """
        Infiere señal para las últimas n_bars
        
        Args:
            n_bars: Número de barras a predecir
            
        Returns:
            DataFrame con señales
        """
        # Cargar datos
        df = pd.read_parquet('data/processed/labeled_features.parquet')
        
        # Cargar metadata
        with open('data/processed/window_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        feature_cols = metadata['feature_columns']
        window_size = metadata['window_size']
        
        # Preparar ventanas para las últimas n_bars
        X_list = []
        for i in range(n_bars):
            start_idx = len(df) - window_size - i
            end_idx = len(df) - i
            window = df.iloc[start_idx:end_idx][feature_cols].values
            X_list.append(window)
        
        X = np.array(X_list)
        
        # Generar señales
        signals = self.generate_signals(X)
        
        # Agregar fechas
        signals['date'] = df.iloc[-n_bars:]['date'].values
        
        return signals

def main():
    """Generar señales para backtest"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    generator = SignalGenerator(config)
    
    # Cargar datos de test
    X_test = np.load('data/processed/X_test.npy')
    
    # Generar señales
    signals = generator.generate_signals(X_test)
    
    # Guardar señales
    signals.to_csv('data/processed/test_signals.csv', index=False)
    
    logger.info(f"Señales generadas: {len(signals)}")
    logger.info(f"Distribución: \n{signals['signal'].value_counts()}")

if __name__ == "__main__":
    main()