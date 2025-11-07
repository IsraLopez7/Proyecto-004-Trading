"""
Script to manually promote a model version to Staging
"""

import argparse
import logging

import mlflow
import yaml
from mlflow.tracking import MlflowClient

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


def promote_model(model_name: str, run_id: str, stage: str = "Staging"):
    """
    Promote a specific model version to a stage.

    Args:
        model_name: Name of registered model
        run_id: Run ID of the model
        stage: Target stage (default: Staging)
    """
    client = MlflowClient()

    # Get model versions for this run
    versions = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")

    if not versions:
        logger.error(f"No model version found for run_id={run_id}")
        return

    version = versions[0].version

    # Transition to target stage
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True
    )

    logger.info(f"Model {model_name} version {version} promoted to {stage}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Promote model to Staging/Production")
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run ID of the model to promote"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Staging",
        choices=["Staging", "Production", "Archived"],
        help="Target stage (default: Staging)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

    model_name = config['mlflow']['model_name']

    logger.info(f"Promoting model {model_name} (run_id={args.run_id}) to {args.stage}")

    promote_model(model_name, args.run_id, args.stage)

    logger.info("Promotion complete!")


if __name__ == "__main__":
    main()
