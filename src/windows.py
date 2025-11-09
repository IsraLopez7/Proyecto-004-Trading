# src/windows.py
"""
Creación de ventanas temporales para CNN
"""
import pandas as pd
import numpy as np
import yaml
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WindowGenerator:
    """Generador de ventanas temporales para series de tiempo"""
    
    def __init__(self, config):
        self.config = config
        self.window_size = config['window_size']
        
    def create_windows(self, data, features, labels):
        """
        Crea ventanas deslizantes de tamaño W
        
        Args:
            data: DataFrame completo
            features: Lista de columnas de features
            labels: Array de etiquetas
            
        Returns:
            X: Array 3D (n_samples, window_size, n_features)
            y: Array 1D de etiquetas
            indices: Índices originales para tracking
        """
        X_list = []
        y_list = []
        indices = []
        
        # Convertir features a array
        feature_data = data[features].values
        
        # Crear ventanas deslizantes
        for i in range(self.window_size, len(data)):
            # Ventana de features
            window = feature_data[i-self.window_size:i]
            
            # Etiqueta correspondiente
            label = labels[i]
            
            # Solo agregar si la etiqueta es válida
            if not np.isnan(label):
                X_list.append(window)
                y_list.append(label)
                indices.append(i)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Ventanas creadas: X shape={X.shape}, y shape={y.shape}")
        
        return X, y, indices
    
    def split_windows(self, X, y, indices, splits):
        """
        Divide ventanas según splits predefinidos
        
        Args:
            X: Array de features
            y: Array de etiquetas
            indices: Índices originales
            splits: Diccionario con índices de splits
            
        Returns:
            Diccionario con X_train, y_train, etc.
        """
        # Ajustar índices por el window_size
        train_start, train_end = splits['train_idx']
        test_start, test_end = splits['test_idx']
        val_start, val_end = splits['val_idx']
        
        # Encontrar índices en el array de ventanas
        train_mask = (indices >= train_start + self.window_size) & (indices < train_end)
        test_mask = (indices >= test_start) & (indices < test_end)
        val_mask = (indices >= val_start) & (indices < val_end)
        
        result = {
            'X_train': X[train_mask],
            'y_train': y[train_mask],
            'X_test': X[test_mask],
            'y_test': y[test_mask],
            'X_val': X[val_mask],
            'y_val': y[val_mask],
            'train_indices': np.array(indices)[train_mask],
            'test_indices': np.array(indices)[test_mask],
            'val_indices': np.array(indices)[val_mask]
        }
        
        logger.info(f"Splits de ventanas:")
        logger.info(f"  Train: {result['X_train'].shape[0]} ventanas")
        logger.info(f"  Test:  {result['X_test'].shape[0]} ventanas")
        logger.info(f"  Val:   {result['X_val'].shape[0]} ventanas")
        
        return result
    
    def process_windows(self):
        """Pipeline principal de creación de ventanas"""
        # Cargar datos etiquetados
        df = pd.read_parquet('data/processed/labeled_features.parquet')
        
        # Cargar lista de features y splits
        with open('data/processed/feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        
        with open('data/processed/splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        
        logger.info(f"Creando ventanas de tamaño {self.window_size}")
        
        # Crear ventanas
        X, y, indices = self.create_windows(
            df,
            feature_cols,
            df['label'].values
        )
        
        # Dividir en train/test/val
        data_splits = self.split_windows(X, y, indices, splits)
        
        # Guardar arrays procesados
        np.save('data/processed/X_train.npy', data_splits['X_train'])
        np.save('data/processed/y_train.npy', data_splits['y_train'])
        np.save('data/processed/X_test.npy', data_splits['X_test'])
        np.save('data/processed/y_test.npy', data_splits['y_test'])
        np.save('data/processed/X_val.npy', data_splits['X_val'])
        np.save('data/processed/y_val.npy', data_splits['y_val'])
        
        # Guardar metadatos
        metadata = {
            'window_size': self.window_size,
            'n_features': X.shape[2],
            'feature_columns': feature_cols,
            'indices': {
                'train': data_splits['train_indices'],
                'test': data_splits['test_indices'],
                'val': data_splits['val_indices']
            }
        }
        
        with open('data/processed/window_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info("Ventanas creadas y guardadas exitosamente")
        
        return data_splits

def main():
    """Ejecutar generación de ventanas"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    generator = WindowGenerator(config)
    generator.process_windows()

if __name__ == "__main__":
    main()