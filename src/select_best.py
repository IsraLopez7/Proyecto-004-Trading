"""
Model Selection Module
Selects best model from MLflow runs based on macro-F1 score
"""

import logging

import mlflow
from mlflow.tracking import MlflowClient
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


def find_best_run(experiment_name: str, metric_name: str = "val_f1_macro") -> dict:
    """
    Find the best run based on a metric.

    Args:
        experiment_name: Name of MLflow experiment
        metric_name: Metric to optimize (default: val_f1_macro)

    Returns:
        Dictionary with best run info
    """
    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Search runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    best_run = runs[0]

    # Extract info
    best_run_info = {
        'run_id': best_run.info.run_id,
        'experiment_id': best_run.info.experiment_id,
        'start_time': best_run.info.start_time,
        'end_time': best_run.info.end_time,
        'status': best_run.info.status,
        'metrics': best_run.data.metrics,
        'params': best_run.data.params
    }

    return best_run_info


def promote_model_to_staging(model_name: str, run_id: str):
    """
    Promote a model version to Staging.

    Args:
        model_name: Name of registered model
        run_id: Run ID of the model to promote
    """
    client = MlflowClient()

    # Get model versions for this run
    versions = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")

    if not versions:
        logger.error(f"No model version found for run_id={run_id}")
        return

    version = versions[0].version

    # Transition to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=True  # Archive previous Staging versions
    )

    logger.info(f"Model {model_name} version {version} promoted to Staging")


def main():
    """Main execution function."""
    logger.info("Selecting best model from MLflow experiments...")

    # Load config
    config = load_config()

    experiment_name = config['mlflow']['experiment_name']
    model_name = config['mlflow']['model_name']

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

    # Find best run
    logger.info(f"Searching for best run in experiment: {experiment_name}")
    best_run = find_best_run(experiment_name, metric_name="val_f1_macro")

    # Print best run info
    logger.info("=" * 60)
    logger.info("BEST MODEL FOUND:")
    logger.info("=" * 60)
    logger.info(f"Run ID: {best_run['run_id']}")
    logger.info(f"Status: {best_run['status']}")

    logger.info("\nMetrics:")
    for metric, value in sorted(best_run['metrics'].items()):
        logger.info(f"  {metric}: {value:.4f}")

    logger.info("\nParameters:")
    for param, value in sorted(best_run['params'].items()):
        logger.info(f"  {param}: {value}")

    logger.info("=" * 60)

    # Promote to Staging
    logger.info(f"\nPromoting model to Staging in Model Registry: {model_name}")
    promote_model_to_staging(model_name, best_run['run_id'])

    logger.info("\nModel selection complete!")
    logger.info(f"Best model (Run ID: {best_run['run_id']}) is now in Staging")
    logger.info(f"Macro F1 Score: {best_run['metrics']['val_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
