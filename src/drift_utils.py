# src/drift_utils.py
"""
Utilidades para detección de drift con KS-test
"""
import pandas as pd
import numpy as np
from scipy import stats
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetector:
    """Detector de drift usando Kolmogorov-Smirnov test"""
    
    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level
        
    def ks_test(self, reference_data, current_data):
        """
        Aplica KS-test entre dos distribuciones
        
        Args:
            reference_data: Datos de referencia (train)
            current_data: Datos actuales (test/val)
            
        Returns:
            statistic: KS statistic
            p_value: p-value del test
            drift_detected: True si p_value < significance_level
        """
        statistic, p_value = stats.ks_2samp(reference_data, current_data)
        drift_detected = p_value < self.significance_level
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected
        }
    
    def detect_feature_drift(self, df, feature_cols, train_idx, test_idx, val_idx):
        """
        Detecta drift para cada feature
        
        Args:
            df: DataFrame completo
            feature_cols: Lista de columnas de features
            train_idx: Índices de train
            test_idx: Índices de test
            val_idx: Índices de validación
            
        Returns:
            DataFrame con resultados de drift
        """
        results = []
        
        # Extraer datos por split
        train_start, train_end = train_idx
        test_start, test_end = test_idx
        val_start, val_end = val_idx
        
        train_data = df.iloc[train_start:train_end]
        test_data = df.iloc[test_start:test_end]
        val_data = df.iloc[val_start:val_end]
        
        for feature in feature_cols:
            # Train vs Test
            train_test_result = self.ks_test(
                train_data[feature].dropna(),
                test_data[feature].dropna()
            )
            
            # Train vs Val
            train_val_result = self.ks_test(
                train_data[feature].dropna(),
                val_data[feature].dropna()
            )
            
            results.append({
                'feature': feature,
                'train_test_statistic': train_test_result['statistic'],
                'train_test_pvalue': train_test_result['p_value'],
                'train_test_drift': train_test_result['drift_detected'],
                'train_val_statistic': train_val_result['statistic'],
                'train_val_pvalue': train_val_result['p_value'],
                'train_val_drift': train_val_result['drift_detected']
            })
        
        drift_df = pd.DataFrame(results)
        
        # Ordenar por p-value más bajo (mayor drift)
        drift_df = drift_df.sort_values('train_test_pvalue')
        
        return drift_df
    
    def get_top_drifted_features(self, drift_df, n=5):
        """
        Obtiene top N features con mayor drift
        
        Args:
            drift_df: DataFrame con resultados de drift
            n: Número de features a retornar
            
        Returns:
            DataFrame con top N features
        """
        # Filtrar features con drift detectado
        drifted = drift_df[
            (drift_df['train_test_drift']) | 
            (drift_df['train_val_drift'])
        ]
        
        return drifted.head(n)
    
    def interpret_drift(self, drift_df):
        """
        Genera interpretación del drift detectado
        
        Args:
            drift_df: DataFrame con resultados de drift
            
        Returns:
            String con interpretación
        """
        n_features = len(drift_df)
        n_drift_test = drift_df['train_test_drift'].sum()
        n_drift_val = drift_df['train_val_drift'].sum()
        
        interpretation = f"""
        === Análisis de Data Drift ===
        
        Total de features analizados: {n_features}
        
        Drift detectado (Train vs Test): {n_drift_test} features ({n_drift_test/n_features*100:.1f}%)
        Drift detectado (Train vs Val): {n_drift_val} features ({n_drift_val/n_features*100:.1f}%)
        
        Interpretación:
        """
        
        if n_drift_test > n_features * 0.3:
            interpretation += """
        ⚠️ ALTO DRIFT: Más del 30% de features muestran drift significativo.
        El modelo podría no generalizar bien a datos nuevos.
        Considerar re-entrenar con datos más recientes.
        """
        elif n_drift_test > n_features * 0.1:
            interpretation += """
        ⚡ DRIFT MODERADO: Entre 10-30% de features muestran drift.
        Monitorear performance del modelo de cerca.
        """
        else:
            interpretation += """
        ✅ DRIFT BAJO: Menos del 10% de features muestran drift.
        El modelo debería mantener su performance.
        """
        
        return interpretation
    
    def run_drift_analysis(self):
        """Pipeline completo de análisis de drift"""
        # Cargar datos
        df = pd.read_parquet('data/processed/features.parquet')
        
        # Cargar metadata
        with open('data/processed/feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        
        with open('data/processed/splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        
        logger.info("Detectando drift en features...")
        
        # Detectar drift
        drift_results = self.detect_feature_drift(
            df,
            feature_cols,
            splits['train_idx'],
            splits['test_idx'],
            splits['val_idx']
        )
        
        # Top features con drift
        top_drift = self.get_top_drifted_features(drift_results)
        
        # Interpretación
        interpretation = self.interpret_drift(drift_results)
        
        # Guardar resultados
        drift_results.to_csv('data/processed/drift_analysis.csv', index=False)
        
        logger.info(interpretation)
        
        return drift_results, top_drift, interpretation

def main():
    """Ejecutar análisis de drift"""
    detector = DriftDetector()
    detector.run_drift_analysis()

if __name__ == "__main__":
    main()