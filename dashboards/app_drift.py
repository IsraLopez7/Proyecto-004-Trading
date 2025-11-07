"""
Streamlit Dashboard for Data Drift Analysis
Visualizes distribution changes and drift detection
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from src.drift_utils import (
    calculate_distribution_stats,
    detect_drift_per_feature,
    get_top_drifted_features,
    interpret_drift,
    compare_distributions
)

# Page config
st.set_page_config(
    page_title="Data Drift Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_config():
    """Load configuration."""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)


@st.cache_data
def load_features():
    """Load feature data."""
    data_dir = Path("data/processed")

    train_df = pd.read_parquet(data_dir / "features_train.parquet")
    val_df = pd.read_parquet(data_dir / "features_val.parquet")
    test_df = pd.read_parquet(data_dir / "features_test.parquet")

    return train_df, val_df, test_df


@st.cache_data
def compute_drift(train_df, test_df, threshold):
    """Compute drift detection."""
    return detect_drift_per_feature(train_df, test_df, threshold)


def plot_distribution_comparison(train_df, val_df, test_df, feature):
    """Plot distribution comparison for a feature."""
    distributions = compare_distributions(train_df, val_df, test_df, feature)

    fig = go.Figure()

    for split_name, values in distributions.items():
        fig.add_trace(go.Histogram(
            x=values,
            name=split_name.capitalize(),
            opacity=0.6,
            nbinsx=30
        ))

    fig.update_layout(
        title=f"Distribution of {feature}",
        xaxis_title=feature,
        yaxis_title="Frequency",
        barmode='overlay',
        height=400,
        template="plotly_white"
    )

    return fig


def plot_drift_scores(drift_results, top_n=10):
    """Plot drift scores for top features."""
    top_features = drift_results.nsmallest(top_n, 'p_value')

    fig = go.Figure()

    colors = ['red' if d else 'green' for d in top_features['drift_detected']]

    fig.add_trace(go.Bar(
        x=top_features['p_value'],
        y=top_features['feature'],
        orientation='h',
        marker=dict(color=colors),
        text=top_features['p_value'].round(4),
        textposition='auto'
    ))

    fig.add_vline(
        x=0.05,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold (p=0.05)"
    )

    fig.update_layout(
        title=f"Top {top_n} Features by Drift (KS-test p-value)",
        xaxis_title="p-value",
        yaxis_title="Feature",
        height=500,
        template="plotly_white"
    )

    return fig


def main():
    """Main Streamlit app."""
    st.title("📊 Data Drift Dashboard")
    st.markdown("**Monitoring distribution changes across train/val/test splits**")

    # Sidebar
    st.sidebar.header("Configuration")

    # Load config
    try:
        config = load_config()
        threshold = st.sidebar.slider(
            "Drift Detection Threshold (p-value)",
            min_value=0.01,
            max_value=0.10,
            value=config['drift']['ks_test_threshold'],
            step=0.01,
            help="Lower values = stricter drift detection"
        )
        top_n = st.sidebar.slider(
            "Number of Top Features",
            min_value=3,
            max_value=15,
            value=config['drift']['top_n_features'],
            step=1
        )
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return

    # Load data
    try:
        with st.spinner("Loading feature data..."):
            train_df, val_df, test_df = load_features()

        st.sidebar.success(f"✅ Data loaded")
        st.sidebar.metric("Features", len(train_df.columns))
        st.sidebar.metric("Train samples", len(train_df))
        st.sidebar.metric("Val samples", len(val_df))
        st.sidebar.metric("Test samples", len(test_df))

    except Exception as e:
        st.error(f"Error loading features: {e}")
        st.info("Please run the feature engineering pipeline first: `python -m src.features`")
        return

    # Compute drift
    with st.spinner("Detecting drift..."):
        drift_results = compute_drift(train_df, test_df, threshold)
        top_drifted = get_top_drifted_features(drift_results, top_n)
        interpretations = interpret_drift(drift_results)

    # Summary metrics
    st.header("📈 Drift Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Features",
            len(drift_results)
        )

    with col2:
        n_drifted = drift_results['drift_detected'].sum()
        st.metric(
            "Features with Drift",
            n_drifted,
            delta=f"{n_drifted/len(drift_results)*100:.1f}%"
        )

    with col3:
        avg_p_value = drift_results['p_value'].mean()
        st.metric(
            "Avg p-value",
            f"{avg_p_value:.4f}"
        )

    with col4:
        max_ks_stat = drift_results['ks_statistic'].max()
        st.metric(
            "Max KS Statistic",
            f"{max_ks_stat:.4f}"
        )

    # Interpretations
    st.header("🔍 Interpretations")
    for interp in interpretations:
        if interp.startswith("⚠️"):
            st.warning(interp)
        elif interp.startswith("ℹ️"):
            st.info(interp)
        else:
            st.write(interp)

    # Drift scores plot
    st.header("📊 Top Features by Drift")
    fig_scores = plot_drift_scores(drift_results, top_n=top_n)
    st.plotly_chart(fig_scores, use_container_width=True)

    # Drift table
    st.header("📋 Drift Detection Results")

    # Add color styling
    def highlight_drift(row):
        if row['drift_detected']:
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)

    styled_df = drift_results.style.apply(highlight_drift, axis=1)

    st.dataframe(
        styled_df.format({
            'ks_statistic': '{:.4f}',
            'p_value': '{:.4f}'
        }),
        use_container_width=True,
        height=400
    )

    # Download drift results
    csv = drift_results.to_csv(index=False)
    st.download_button(
        label="📥 Download Drift Results (CSV)",
        data=csv,
        file_name="drift_results.csv",
        mime="text/csv"
    )

    # Feature-level analysis
    st.header("🔬 Feature-Level Distribution Analysis")

    selected_feature = st.selectbox(
        "Select feature to analyze",
        options=list(train_df.columns),
        index=0
    )

    if selected_feature:
        # Get drift info for this feature
        feature_drift = drift_results[drift_results['feature'] == selected_feature].iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("KS Statistic", f"{feature_drift['ks_statistic']:.4f}")

        with col2:
            drift_status = "🔴 Drift Detected" if feature_drift['drift_detected'] else "🟢 No Drift"
            st.metric("p-value", f"{feature_drift['p_value']:.4f}", delta=drift_status)

        # Plot distribution
        fig_dist = plot_distribution_comparison(train_df, val_df, test_df, selected_feature)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Statistics table
        stats_df = calculate_distribution_stats(train_df, val_df, test_df)
        feature_stats = stats_df[stats_df['feature'] == selected_feature].T
        feature_stats.columns = ['Value']
        st.dataframe(feature_stats.style.format("{:.4f}"), use_container_width=True)

    # Top drifted features detail
    st.header(f"🎯 Top {top_n} Drifted Features - Detailed View")

    for idx, row in top_drifted.iterrows():
        with st.expander(f"{row['feature']} (p-value: {row['p_value']:.4f})"):
            fig = plot_distribution_comparison(train_df, val_df, test_df, row['feature'])
            st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Deep Trading MLOps** | Data Drift Monitoring | "
        "Built with Streamlit 📊"
    )


if __name__ == "__main__":
    main()
