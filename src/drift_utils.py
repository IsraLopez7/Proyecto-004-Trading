"""
Data Drift Utilities
Tools for detecting and analyzing distribution drift using KS-test
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def kolmogorov_smirnov_test(
    reference: np.ndarray,
    current: np.ndarray
) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test.

    Args:
        reference: Reference distribution (e.g., train)
        current: Current distribution (e.g., test)

    Returns:
        Tuple of (statistic, p_value)
    """
    statistic, p_value = ks_2samp(reference, current)
    return statistic, p_value


def detect_drift_per_feature(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Detect drift for each feature using KS-test.

    Args:
        train_df: Training features
        test_df: Test features
        threshold: p-value threshold for drift detection

    Returns:
        DataFrame with drift detection results
    """
    results = []

    for col in train_df.columns:
        if col not in test_df.columns:
            continue

        train_values = train_df[col].dropna().values
        test_values = test_df[col].dropna().values

        statistic, p_value = kolmogorov_smirnov_test(train_values, test_values)

        drift_detected = p_value < threshold

        results.append({
            'feature': col,
            'ks_statistic': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')

    return results_df


def get_top_drifted_features(
    drift_results: pd.DataFrame,
    top_n: int = 5
) -> pd.DataFrame:
    """Get top N features with highest drift."""
    return drift_results.nsmallest(top_n, 'p_value')


def interpret_drift(drift_results: pd.DataFrame) -> List[str]:
    """
    Generate interpretations for detected drift.

    Args:
        drift_results: DataFrame with drift detection results

    Returns:
        List of interpretation strings
    """
    interpretations = []

    drifted = drift_results[drift_results['drift_detected']]

    if len(drifted) == 0:
        interpretations.append("No significant drift detected across features.")
        return interpretations

    interpretations.append(f"Drift detected in {len(drifted)} out of {len(drift_results)} features.")

    # Top drifted features
    top_5 = get_top_drifted_features(drift_results, top_n=5)

    interpretations.append("\nTop 5 features with highest drift:")
    for idx, row in top_5.iterrows():
        interpretations.append(
            f"  - {row['feature']}: p-value={row['p_value']:.4f}, "
            f"KS-statistic={row['ks_statistic']:.4f}"
        )

    # Recommendations
    if len(drifted) > len(drift_results) * 0.3:
        interpretations.append(
            "\n⚠️  Warning: More than 30% of features show drift. "
            "Consider retraining the model with recent data."
        )
    elif len(drifted) > 0:
        interpretations.append(
            "\nℹ️  Some drift detected. Monitor model performance closely."
        )

    return interpretations


def compare_distributions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature: str
) -> Dict[str, np.ndarray]:
    """
    Extract distributions for a specific feature across splits.

    Args:
        train_df: Training features
        val_df: Validation features
        test_df: Test features
        feature: Feature name

    Returns:
        Dictionary with distributions per split
    """
    return {
        'train': train_df[feature].dropna().values,
        'val': val_df[feature].dropna().values,
        'test': test_df[feature].dropna().values
    }


def calculate_distribution_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate distribution statistics for all features.

    Args:
        train_df: Training features
        val_df: Validation features
        test_df: Test features

    Returns:
        DataFrame with statistics per feature
    """
    stats = []

    for col in train_df.columns:
        stats.append({
            'feature': col,
            'train_mean': train_df[col].mean(),
            'train_std': train_df[col].std(),
            'val_mean': val_df[col].mean(),
            'val_std': val_df[col].std(),
            'test_mean': test_df[col].mean(),
            'test_std': test_df[col].std(),
            'mean_shift_train_test': abs(train_df[col].mean() - test_df[col].mean()),
            'std_shift_train_test': abs(train_df[col].std() - test_df[col].std())
        })

    return pd.DataFrame(stats)


if __name__ == "__main__":
    # Test drift detection
    logger.info("Testing drift detection utilities...")

    from pathlib import Path

    # Load features
    data_dir = Path("data/processed")

    if (data_dir / "features_train.parquet").exists():
        train_df = pd.read_parquet(data_dir / "features_train.parquet")
        val_df = pd.read_parquet(data_dir / "features_val.parquet")
        test_df = pd.read_parquet(data_dir / "features_test.parquet")

        # Detect drift
        drift_results = detect_drift_per_feature(train_df, test_df)

        logger.info("\nDrift Detection Results:")
        logger.info(f"\n{drift_results}")

        # Interpretations
        interpretations = interpret_drift(drift_results)

        logger.info("\nInterpretations:")
        for interp in interpretations:
            logger.info(interp)

        # Distribution stats
        stats = calculate_distribution_stats(train_df, val_df, test_df)

        logger.info("\nDistribution Statistics (Top 5 by mean shift):")
        logger.info(f"\n{stats.nlargest(5, 'mean_shift_train_test')}")

        logger.info("\nDrift detection test complete!")
    else:
        logger.warning("Features not found. Run feature engineering first.")
