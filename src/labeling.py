# src/labeling.py
"""
Genera etiquetas multiclase {long:0, hold:1, short:2} basadas en:
- Horizonte H (días hacia adelante)
- Umbral τ (threshold para decidir si es señal clara)

Reglas:
  return_{t→t+H} > +τ  ⇒ long (0)
  return_{t→t+H} < −τ  ⇒ short (2)
  otro caso            ⇒ hold (1)
"""

import pandas as pd
import numpy as np
import yaml
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_labels(df, H=5, tau=0.005):
    """
    Calcula etiquetas basadas en rendimiento futuro.
    
    Args:
        df: DataFrame con columna 'Close'
        H: Horizonte (días hacia adelante)
        tau: Umbral (ej: 0.005 = 0.5%)
    
    Returns:
        Series con etiquetas {0: long, 1: hold, 2: short}
    """
    # Rendimiento futuro de H días
    future_return = df['Close'].shift(-H) / df['Close'] - 1
    
    # Clasificación
    labels = pd.Series(index=df.index, dtype=int)
    labels[future_return > tau] = 0   # Long
    labels[future_return < -tau] = 2  # Short
    labels[(future_return >= -tau) & (future_return <= tau)] = 1  # Hold
    
    return labels


def analyze_class_distribution(labels):
    """Analiza distribución de clases y calcula class weights."""
    counts = Counter(labels.dropna())
    total = sum(counts.values())
    
    logger.info("Distribución de clases:")
    label_names = {0: 'long', 1: 'hold', 2: 'short'}
    for label, count in sorted(counts.items()):
        pct = (count / total) * 100
        logger.info(f"  {label_names[label]} ({label}): {count} ({pct:.2f}%)")
    
    # Calcular class weights: inversamente proporcional a frecuencia
    n_classes = len(counts)
    class_weights = {}
    for label in range(n_classes):
        if label in counts:
            class_weights[label] = total / (n_classes * counts[label])
        else:
            class_weights[label] = 1.0
    
    logger.info(f"Class weights calculados: {class_weights}")
    
    return class_weights


def main():
    """Flujo principal de generación de etiquetas."""
    config = load_config()
    
    # Cargar features
    df = pd.read_parquet(config['data']['processed_path'])
    logger.info(f"Features cargados: {len(df)} filas")
    
    # Parámetros de labeling
    H = config['labeling']['H']
    tau = config['labeling']['tau']
    logger.info(f"Parámetros: H={H}, τ={tau} ({tau*100:.2f}%)")
    
    # Crear etiquetas
    df['label'] = create_labels(df, H=H, tau=tau)
    
    # Analizar distribución
    class_weights = analyze_class_distribution(df['label'])
    
    # Guardar class weights en archivo
    weights_df = pd.DataFrame(class_weights.items(), columns=['class', 'weight'])
    weights_df.to_csv('data/processed/class_weights.csv', index=False)
    logger.info("Class weights guardados en data/processed/class_weights.csv")
    
    # Remover últimas H filas (no tienen etiqueta válida)
    df = df.iloc[:-H]
    logger.info(f"Filas después de remover últimas {H}: {len(df)}")
    
    # Guardar features + labels
    df.to_parquet(config['data']['processed_path'])
    logger.info(f"Features con labels guardados en {config['data']['processed_path']}")
    
    logger.info("✅ labeling.py completado exitosamente")


if __name__ == "__main__":
    main()