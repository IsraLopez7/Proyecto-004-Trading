# src/select_best.py
"""
Selecciona el mejor modelo del Model Registry basado en macro-F1 en validación
y lo promociona a "Staging".
"""

import mlflow
from mlflow.tracking import MlflowClient
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_best_run(experiment_name, metric_name='val_f1_macro'):
    """
    Busca el run con mejor métrica en el experimento.
    
    Returns:
        run_id del mejor modelo
    """
    client = MlflowClient()
    
    # Obtener experimento
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' no encontrado")
    
    # Buscar runs ordenados por métrica
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        raise ValueError(f"No se encontraron runs en experimento '{experiment_name}'")
    
    best_run = runs[0]
    best_metric = best_run.data.metrics.get(metric_name)
    
    logger.info(f"Mejor run encontrado:")
    logger.info(f"  Run ID: {best_run.info.run_id}")
    logger.info(f"  {metric_name}: {best_metric:.4f}")
    logger.info(f"  Fecha: {best_run.info.start_time}")
    
    return best_run.info.run_id, best_metric


def promote_to_staging(model_name, run_id):
    """
    Promociona versión del modelo a Staging.
    """
    client = MlflowClient()
    
    # Buscar versión del modelo correspondiente al run_id
    model_versions = client.search_model_versions(f"run_id='{run_id}'")
    
    if len(model_versions) == 0:
        raise ValueError(f"No se encontró versión de modelo para run_id={run_id}")
    
    version = model_versions[0].version
    
    # Transicionar a Staging
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=True  # Archiva versiones anteriores en Staging
    )
    
    logger.info(f"Modelo '{model_name}' versión {version} promocionado a Staging")
    
    return version


def main():
    config = load_config()
    
    experiment_name = config['mlflow']['experiment_name']
    model_name = config['mlflow']['model_name']
    
    logger.info(f"Seleccionando mejor modelo de experimento '{experiment_name}'")
    
    # Obtener mejor run
    best_run_id, best_metric = get_best_run(experiment_name, metric_name='val_f1_macro')
    
    # Promocionar a Staging
    version = promote_to_staging(model_name, best_run_id)
    
    logger.info("✅ Modelo en Staging listo para producción")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Version: {version}")
    logger.info(f"   Run ID: {best_run_id}")
    logger.info(f"   Macro-F1: {best_metric:.4f}")


if __name__ == "__main__":
    main()