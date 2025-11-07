"""
Training Module
Trains CNN model with MLflow tracking
"""

import logging
from pathlib import Path
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import seaborn as sns

from src.model_cnn import (
    build_cnn_model,
    compute_class_weights,
    get_callbacks,
    print_model_summary
)

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


def load_windows(data_dir: str = "data/processed") -> tuple:
    """Load windowed data."""
    data_path = Path(data_dir)

    train_data = np.load(data_path / "windows_train.npz")
    val_data = np.load(data_path / "windows_val.npz")
    test_data = np.load(data_path / "windows_test.npz")

    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    X_test, y_test = test_data['X'], test_data['y']

    logger.info(f"Loaded windows: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix (Validation Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Confusion matrix saved to {save_path}")


def plot_training_history(history, save_path: str):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Training curves saved to {save_path}")


def main():
    """Main training function with MLflow tracking."""
    logger.info("Starting training pipeline with MLflow tracking...")

    # Load config
    config = load_config()

    # MLflow setup
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_windows()

    input_shape = (X_train.shape[1], X_train.shape[2])  # (W, n_features)
    n_classes = len(np.unique(y_train))

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Number of classes: {n_classes}")

    # Compute class weights
    class_weights = compute_class_weights(y_train) if config['model']['class_weights'] is None \
        else config['model']['class_weights']

    # Start MLflow run
    with mlflow.start_run():
        logger.info(f"MLflow run started: {mlflow.active_run().info.run_id}")

        # Log parameters
        params = {
            'window_size': config['model']['W'],
            'horizon': config['labeling']['H'],
            'tau': config['labeling']['tau'],
            'batch_size': config['model']['batch_size'],
            'epochs': config['model']['epochs'],
            'learning_rate': config['model']['learning_rate'],
            'filters': str(config['model']['filters']),
            'kernel_size': config['model']['kernel_size'],
            'pool_size': config['model']['pool_size'],
            'dropout': config['model']['dropout'],
            'dense_units': str(config['model']['dense_units']),
            'seed': config['model']['seed'],
            'class_weights': str(class_weights),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'n_features': input_shape[1]
        }

        mlflow.log_params(params)

        # Build model
        logger.info("Building CNN model...")
        model = build_cnn_model(
            input_shape=input_shape,
            n_classes=n_classes,
            filters=config['model']['filters'],
            kernel_size=config['model']['kernel_size'],
            pool_size=config['model']['pool_size'],
            dropout=config['model']['dropout'],
            dense_units=config['model']['dense_units'],
            learning_rate=config['model']['learning_rate'],
            seed=config['model']['seed']
        )

        print_model_summary(model)

        # Log model summary as artifact
        if config['mlflow']['log_model_summary']:
            from io import StringIO
            import sys

            old_stdout = sys.stdout
            sys.stdout = buffer = StringIO()
            model.summary()
            sys.stdout = old_stdout
            summary_str = buffer.getvalue()

            with open("/tmp/model_summary.txt", "w") as f:
                f.write(summary_str)
            mlflow.log_artifact("/tmp/model_summary.txt", artifact_path="model")

        # Get callbacks
        callbacks = get_callbacks(
            early_stopping_patience=config['model']['early_stopping_patience'],
            reduce_lr_patience=config['model']['reduce_lr_patience'],
            reduce_lr_factor=config['model']['reduce_lr_factor'],
            min_lr=config['model']['min_lr']
        )

        # Train model
        logger.info("Training model...")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config['model']['epochs'],
            batch_size=config['model']['batch_size'],
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        y_val_pred_probs = model.predict(X_val, verbose=0)
        y_val_pred = np.argmax(y_val_pred_probs, axis=1)

        # Metrics
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
        val_f1_per_class = f1_score(y_val, y_val_pred, average=None)

        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation F1 (macro): {val_f1_macro:.4f}")

        # Log metrics
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_f1_macro", val_f1_macro)

        class_names = config['labeling']['class_names']
        for i, class_name in enumerate(class_names):
            mlflow.log_metric(f"val_f1_{class_name}", val_f1_per_class[i])

        # Classification report
        logger.info("\nClassification Report (Validation):")
        report = classification_report(
            y_val,
            y_val_pred,
            target_names=class_names
        )
        logger.info(f"\n{report}")

        # Save classification report
        with open("/tmp/classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("/tmp/classification_report.txt", artifact_path="metrics")

        # Plot and log confusion matrix
        if config['mlflow']['log_confusion_matrix']:
            cm_path = "/tmp/confusion_matrix.png"
            plot_confusion_matrix(y_val, y_val_pred, class_names, cm_path)
            mlflow.log_artifact(cm_path, artifact_path="metrics")

        # Plot and log training history
        if config['mlflow']['log_training_curves']:
            history_path = "/tmp/training_curves.png"
            plot_training_history(history, history_path)
            mlflow.log_artifact(history_path, artifact_path="metrics")

        # Log final epoch metrics
        final_epoch = len(history.history['loss'])
        mlflow.log_metric("final_epoch", final_epoch)
        mlflow.log_metric("final_train_loss", history.history['loss'][-1])
        mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])

        # Log model to MLflow
        logger.info("Logging model to MLflow...")
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name=config['mlflow']['model_name']
        )

        logger.info(f"Model logged to MLflow Model Registry: {config['mlflow']['model_name']}")
        logger.info(f"Run ID: {mlflow.active_run().info.run_id}")

        # Evaluate on test set (for reference, not for model selection)
        logger.info("Evaluating on test set...")
        y_test_pred_probs = model.predict(X_test, verbose=0)
        y_test_pred = np.argmax(y_test_pred_probs, axis=1)

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1_macro = f1_score(y_test, y_test_pred, average='macro')

        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test F1 (macro): {test_f1_macro:.4f}")

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1_macro", test_f1_macro)

    logger.info("Training complete!")
    logger.info(f"View results in MLflow UI: {config['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    main()
