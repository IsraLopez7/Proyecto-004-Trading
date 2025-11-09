# src/drift_utils.py
"""
Utilidades para detección de drift usando Kolmogorov-Smirnov test.
Compara distribuciones de features entre train/test/val.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_feature_data():
    """Carga features de train, test y val."""
    # Cargar datos
    df = pd.read_parquet('data/processed/features.parquet')
    
    # Cargar splits info
    splits_info = pd.read_csv('data/processed/splits_info.csv', index_col=0)
    train_size = int(splits_info.loc['train', 'size'])
    test_size = int(splits_info.loc['test', 'size'])
    
    # Dividir
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:train_size+test_size]
    df_val = df.iloc[train_size+test_size:]
    
    logger.info(f"Train: {len(df_train)}, Test: {len(df_test)}, Val: {len(df_val)}")
    
    return df_train, df_test, df_val


def calculate_ks_drift(df_reference, df_current, feature_cols):
    """
    Calcula KS-test p-value para cada feature entre dos datasets.
    
    Args:
        df_reference: DataFrame de referencia (ej: train)
        df_current: DataFrame actual (ej: test o val)
        feature_cols: lista de features a analizar
    
    Returns:
        DataFrame con feature, statistic, p_value, drift_detected
    """
    results = []
    
    for feature in feature_cols:
        ref_data = df_reference[feature].dropna().values
        curr_data = df_current[feature].dropna().values
        
        # KS test
        statistic, p_value = ks_2samp(ref_data, curr_data)
        
        # Drift detectado si p < 0.05
        drift_detected = p_value < 0.05
        
        results.append({
            'feature': feature,
            'ks_statistic': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')
    
    return results_df


def get_top_drift_features(drift_df, top_n=5):
    """Obtiene top N features con mayor drift (menor p-value)."""
    top_drift = drift_df.nsmallest(top_n, 'p_value')
    return top_drift


def interpret_drift(drift_df):
    """Genera interpretación básica del drift."""
    n_drift = (drift_df['drift_detected'] == True).sum()
    pct_drift = (n_drift / len(drift_df)) * 100
    
    interpretation = f"""
    📊 ANÁLISIS DE DRIFT
    
    Total de features analizados: {len(drift_df)}
    Features con drift detectado (p < 0.05): {n_drift} ({pct_drift:.1f}%)
    
    Interpretación:
    - Si >30% de features tienen drift → Alta probabilidad de degradación del modelo
    - Si 10-30% tienen drift → Drift moderado, monitorear de cerca
    - Si <10% tienen drift → Drift bajo, modelo estable
    
    Top 5 features con mayor drift:
    """
    
    top_5 = drift_df.nsmallest(5, 'p_value')
    for _, row in top_5.iterrows():
        interpretation += f"\n  • {row['feature']}: p-value={row['p_value']:.6f}"
    
    return interpretation


if __name__ == "__main__":
    # Test de funcionalidades
    df_train, df_test, df_val = load_feature_data()
    
    # Features a analizar (excluir OHLCV y label)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'label']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]
    
    # Drift train vs test
    logger.info("Calculando drift: train vs test")
    drift_test = calculate_ks_drift(df_train, df_test, feature_cols)
    
    # Drift train vs val
    logger.info("Calculando drift: train vs val")
    drift_val = calculate_ks_drift(df_train, df_val, feature_cols)
    
    # Interpretación
    print(interpret_drift(drift_test))
    
    logger.info("✅ drift_utils.py test completado")