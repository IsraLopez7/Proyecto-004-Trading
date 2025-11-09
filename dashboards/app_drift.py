# dashboards/app_drift.py
"""
Dashboard Streamlit para visualización de drift.
Muestra KS-test p-values y Top-5 features con mayor drift.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from src.drift_utils import (
    load_feature_data,
    calculate_ks_drift,
    get_top_drift_features,
    interpret_drift
)

st.set_page_config(page_title="Data Drift Monitor", layout="wide")

st.title("📊 Data Drift Monitoring Dashboard")
st.markdown("Monitoreo de drift usando Kolmogorov-Smirnov test")

# Cargar datos
with st.spinner("Cargando datos..."):
    df_train, df_test, df_val = load_feature_data()

# Features a analizar
exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'label']
feature_cols = [col for col in df_train.columns if col not in exclude_cols]

st.success(f"✅ Datos cargados: {len(feature_cols)} features analizados")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Train vs Test", "📉 Train vs Val", "🔍 Interpretación"])

with tab1:
    st.header("Drift: Train vs Test")
    
    drift_test = calculate_ks_drift(df_train, df_test, feature_cols)
    
    # Métricas
    col1, col2, col3 = st.columns(3)
    with col1:
        n_drift = (drift_test['drift_detected'] == True).sum()
        st.metric("Features con drift", n_drift)
    with col2:
        pct_drift = (n_drift / len(drift_test)) * 100
        st.metric("% de drift", f"{pct_drift:.1f}%")
    with col3:
        min_pvalue = drift_test['p_value'].min()
        st.metric("p-value mínimo", f"{min_pvalue:.6f}")
    
    # Tabla completa
    st.subheader("Resultados completos")
    st.dataframe(drift_test, use_container_width=True)
    
    # Top 5
    st.subheader("Top 5 features con mayor drift")
    top_5 = get_top_drift_features(drift_test, top_n=5)
    
    for _, row in top_5.iterrows():
        with st.expander(f"🔴 {row['feature']} (p-value: {row['p_value']:.6f})"):
            st.write(f"**KS Statistic:** {row['ks_statistic']:.4f}")
            st.write(f"**Drift detectado:** {'Sí' if row['drift_detected'] else 'No'}")
            
            # Mini histograma comparativo
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 3))
            
            ax.hist(df_train[row['feature']].dropna(), bins=30, alpha=0.5, label='Train', density=True)
            ax.hist(df_test[row['feature']].dropna(), bins=30, alpha=0.5, label='Test', density=True)
            ax.legend()
            ax.set_xlabel(row['feature'])
            ax.set_ylabel('Densidad')
            ax.set_title(f"Distribución: {row['feature']}")
            
            st.pyplot(fig)

with tab2:
    st.header("Drift: Train vs Validation")
    
    drift_val = calculate_ks_drift(df_train, df_val, feature_cols)
    
    # Métricas
    col1, col2, col3 = st.columns(3)
    with col1:
        n_drift_val = (drift_val['drift_detected'] == True).sum()
        st.metric("Features con drift", n_drift_val)
    with col2:
        pct_drift_val = (n_drift_val / len(drift_val)) * 100
        st.metric("% de drift", f"{pct_drift_val:.1f}%")
    with col3:
        min_pvalue_val = drift_val['p_value'].min()
        st.metric("p-value mínimo", f"{min_pvalue_val:.6f}")
    
    # Tabla
    st.subheader("Resultados completos")
    st.dataframe(drift_val, use_container_width=True)
    
    # Top 5
    st.subheader("Top 5 features con mayor drift")
    top_5_val = get_top_drift_features(drift_val, top_n=5)
    
    for _, row in top_5_val.iterrows():
        with st.expander(f"🔴 {row['feature']} (p-value: {row['p_value']:.6f})"):
            st.write(f"**KS Statistic:** {row['ks_statistic']:.4f}")
            st.write(f"**Drift detectado:** {'Sí' if row['drift_detected'] else 'No'}")

with tab3:
    st.header("🧠 Interpretación del Drift")
    
    st.markdown("### Train vs Test")
    interpretation_test = interpret_drift(drift_test)
    st.text(interpretation_test)
    
    st.markdown("---")
    
    st.markdown("### Train vs Validation")
    interpretation_val = interpret_drift(drift_val)
    st.text(interpretation_val)
    
    st.markdown("---")
    
    st.info("""
    **¿Qué hacer si hay drift alto?**
    
    1. **Reentrenar el modelo** con datos más recientes
    2. **Revisar features** que cambiaron más
    3. **Ajustar ventanas** de features (ej: SMA más cortas)
    4. **Feature selection** para remover features inestables
    5. **Monitoreo continuo** del performance en producción
    """)

st.markdown("---")
st.caption("Dashboard creado con Streamlit | Deep Learning Trading MLOps")