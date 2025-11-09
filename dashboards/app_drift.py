# dashboards/app_drift.py
"""
Dashboard Streamlit para monitoreo de drift
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Data Drift Monitor",
    page_icon="📊",
    layout="wide"
)

st.title("🔍 Monitor de Data Drift")
st.markdown("---")

# Cargar datos de drift
@st.cache_data
def load_drift_data():
    """Carga resultados de análisis de drift"""
    if Path('data/processed/drift_analysis.csv').exists():
        return pd.read_csv('data/processed/drift_analysis.csv')
    else:
        return None

# Cargar datos
drift_df = load_drift_data()

if drift_df is None:
    st.error("❌ No se encontraron datos de drift. Ejecuta primero: `python src/drift_utils.py`")
else:
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    total_features = len(drift_df)
    drift_test = drift_df['train_test_drift'].sum()
    drift_val = drift_df['train_val_drift'].sum()
    avg_pvalue = drift_df['train_test_pvalue'].mean()
    
    with col1:
        st.metric("Total Features", total_features)
    
    with col2:
        st.metric(
            "Drift Train→Test",
            f"{drift_test}",
            f"{drift_test/total_features*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Drift Train→Val",
            f"{drift_val}",
            f"{drift_val/total_features*100:.1f}%"
        )
    
    with col4:
        st.metric("Avg P-Value", f"{avg_pvalue:.4f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Tabla de Drift",
        "🔝 Top-5 Features",
        "📈 Visualización",
        "💡 Interpretación"
    ])
    
    with tab1:
        st.subheader("Tabla Completa de Drift por Feature")
        
        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            show_drift_only = st.checkbox("Mostrar solo features con drift", value=False)
        
        with col2:
            p_value_threshold = st.slider(
                "Umbral p-value",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01
            )
        
        # Filtrar datos
        display_df = drift_df.copy()
        if show_drift_only:
            display_df = display_df[
                (display_df['train_test_drift']) |
                (display_df['train_val_drift'])
            ]
        
        # Formatear tabla
        display_df['train_test_pvalue'] = display_df['train_test_pvalue'].apply(lambda x: f"{x:.6f}")
        display_df['train_val_pvalue'] = display_df['train_val_pvalue'].apply(lambda x: f"{x:.6f}")
        display_df['train_test_statistic'] = display_df['train_test_statistic'].apply(lambda x: f"{x:.4f}")
        display_df['train_val_statistic'] = display_df['train_val_statistic'].apply(lambda x: f"{x:.4f}")
        
        # Renombrar columnas para display
        display_df = display_df.rename(columns={
            'feature': 'Feature',
            'train_test_statistic': 'KS Stat (Test)',
            'train_test_pvalue': 'P-Value (Test)',
            'train_test_drift': 'Drift Test?',
            'train_val_statistic': 'KS Stat (Val)',
            'train_val_pvalue': 'P-Value (Val)',
            'train_val_drift': 'Drift Val?'
        })
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    with tab2:
        st.subheader("🚨 Top-5 Features con Mayor Drift")
        
        # Obtener top 5 con menor p-value
        top5 = drift_df.nsmallest(5, 'train_test_pvalue')
        
        for idx, row in top5.iterrows():
            with st.expander(f"📊 {row['feature']}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "KS Statistic",
                        f"{row['train_test_statistic']:.4f}"
                    )
                
                with col2:
                    st.metric(
                        "P-Value",
                        f"{row['train_test_pvalue']:.6f}"
                    )
                
                with col3:
                    drift_status = "🔴 DRIFT" if row['train_test_drift'] else "🟢 OK"
                    st.metric("Status", drift_status)
                
                # Interpretación
                if row['train_test_drift']:
                    st.warning(
                        f"⚠️ La distribución de **{row['feature']}** ha cambiado "
                        f"significativamente entre train y test (p-value < 0.05)"
                    )
    
    with tab3:
        st.subheader("Visualización de Drift")
        
        # Gráfico de barras con p-values
        fig1 = px.bar(
            drift_df.head(20),
            x='feature',
            y='train_test_pvalue',
            title='P-Values por Feature (Train vs Test)',
            labels={'train_test_pvalue': 'P-Value', 'feature': 'Feature'},
            color='train_test_drift',
            color_discrete_map={True: 'red', False: 'green'}
        )
        
        # Agregar línea de umbral
        fig1.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="red",
            annotation_text="Umbral (p=0.05)"
        )
        
        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Scatter plot: KS Statistic vs P-Value
        fig2 = px.scatter(
            drift_df,
            x='train_test_