# src/select_best.py
"""
Selecciona el mejor modelo por macro-F1 y lo promueve a Staging
"""
import mlflow
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_best_model(experiment_name, model_name, metric='macro_f1'):
    """
    Selecciona el mejor run y lo registra en Staging
    
    Args:
        experiment_name: Nombre del experimento MLflow
        model_name: Nombre del modelo en registry
        metric: Métrica para selección
    """
    # Configurar MLflow
    mlflow.set_experiment(experiment_name)
    
    # Obtener experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    # Buscar todos los runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        logger.error("No se encontraron runs en el experimento")
        return
    
    # Mejor run
    best_run = runs.iloc[0]
    best_run_id = best_run.run_id
    best_metric_value = best_run[f"metrics.{metric}"]
    
    logger.info(f"Mejor run encontrado:")
    logger.info(f"  Run ID: {best_run_id}")
    logger.info(f"  {metric}: {best_metric_value:.4f}")
    
    # Obtener cliente de MLflow
    client = mlflow.tracking.MlflowClient()
    
    # Buscar versión del modelo
    model_versions = client.search_model_versions(
        filter_string=f"run_id='{best_run_id}'"
    )
    
    if len(model_versions) > 0:
        # Promover a Staging
        version = model_versions[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )
        
        logger.info(f"Modelo promovido a Staging:")
        logger.info(f"  Modelo: {model_name}")
        logger.info(f"  Versión: {version}")
        logger.info(f"  Stage: Staging")
    else:
        logger.error(f"No se encontró modelo registrado para run {best_run_id}")
    
    return best_run_id, best_metric_value

def main():
    """Ejecutar selección de mejor modelo"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    select_best_model(
        config['mlflow']['experiment_name'],
        config['mlflow']['model_name'],
        metric='macro_f1'
    )

if __name__ == "__main__":
    main()