"""
Inference Module
Generates predictions using best model from MLflow for backtesting
"""

import logging
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_from_registry(model_name: str, stage: str = "Staging"):
    """
    Load model from MLflow Model Registry.

    Args:
        model_name: Name of registered model
        stage: Stage to load (default: Staging)

    Returns:
        Loaded model
    """
    model_uri = f"models:/{model_name}/{stage}"
    logger.info(f"Loading model from: {model_uri}")

    model = mlflow.keras.load_model(model_uri)

    logger.info("Model loaded successfully")
    return model


def generate_predictions(
    model,
    X: np.ndarray,
    class_names: list
) -> pd.DataFrame:
    """
    Generate predictions with model.

    Args:
        model: Trained model
        X: Input features
        class_names: List of class names

    Returns:
        DataFrame with predictions and probabilities
    """
    logger.info(f"Generating predictions for {len(X)} samples...")

    # Predict
    y_pred_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Create DataFrame
    results = pd.DataFrame({
        'prediction': y_pred,
        'signal': [class_names[i] for i in y_pred]
    })

    # Add probabilities
    for i, class_name in enumerate(class_names):
        results[f'prob_{class_name}'] = y_pred_probs[:, i]

    logger.info("Predictions generated")
    logger.info(f"\nSignal distribution:")
    signal_counts = results['signal'].value_counts()
    for signal, count in signal_counts.items():
        logger.info(f"  {signal}: {count} ({count/len(results)*100:.1f}%)")

    return results


def main():
    """Main execution function."""
    logger.info("Starting inference pipeline...")

    # Load config
    config = load_config()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

    # Load model
    model_name = config['mlflow']['model_name']
    model = load_model_from_registry(model_name, stage="Staging")

    # Load test windows
    data_dir = Path("data/processed")
    test_data = np.load(data_dir / "windows_test.npz")
    X_test = test_data['X']
    y_test = test_data['y']

    logger.info(f"Test set: {X_test.shape}")

    # Generate predictions
    class_names = config['labeling']['class_names']
    predictions = generate_predictions(model, X_test, class_names)

    # Add true labels for comparison
    predictions['true_label'] = y_test
    predictions['true_signal'] = [class_names[i] for i in y_test]

    # Save predictions
    output_path = data_dir / "signals_test.parquet"
    predictions.to_parquet(output_path)

    logger.info(f"Predictions saved to {output_path}")

    # Calculate accuracy
    accuracy = (predictions['prediction'] == predictions['true_label']).mean()
    logger.info(f"\nTest Accuracy: {accuracy:.4f}")

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
