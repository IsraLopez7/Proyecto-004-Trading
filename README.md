# Proyecto-004-Trading
Proyecto #4 de trading

# Deep Trading MLOps - CNN Signal Prediction System

Sistema de trading sistemático con deep learning (CNN) para predicción de señales {long, short, hold} con tracking de experimentos en MLflow, API REST y monitoreo de data drift.

## Arquitectura

- **Data Pipeline**: Procesamiento de datos OHLCV con 20+ features de momentum, volatilidad y volumen
- **Modelo**: CNN en TensorFlow/Keras para capturar patrones temporales
- **MLOps**: MLflow para tracking de experimentos y model registry
- **API**: FastAPI para servir predicciones en tiempo real
- **Monitoring**: Dashboard Streamlit para detectar data drift
- **Backtesting**: Motor con costos realistas (comisiones 0.125%, borrow rate 0.25%)

## Requisitos

- Python 3.8+
- 8GB RAM mínimo
- GPU opcional pero recomendada

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env si es necesario

# 4. Crear directorios si no existen
mkdir -p data/{raw,processed}
mkdir -p mlruns
```

## Pipeline de Ejecución

### 1. Preparar datos SPY

```bash
# Descargar datos de SPY (15+ años)
# Colocar archivo CSV en data/raw/SPY.csv con columnas: Date,Open,High,Low,Close,Volume
```

### 2. Lanzar MLflow UI

```bash
# Terminal 1 - MLflow UI
mlflow ui --port 5000
# Acceder en: http://localhost:5000
```

### 3. Ejecutar Pipeline Completo

```bash
# Cargar y limpiar datos
python -m src.data_loader --csv data/raw/SPY.csv --output data/processed/

# Generar features (20+)
python -m src.features --input data/processed/clean_data.parquet --output data/processed/features.parquet

# Generar etiquetas {long, short, hold}
python -m src.labeling --input data/processed/features.parquet --output data/processed/labeled_data.parquet --horizon 5 --threshold 0.005

# Crear ventanas para CNN
python -m src.windows --input data/processed/labeled_data.parquet --output data/processed/windows.npz --window_size 256

# Entrenar modelos con MLflow
python -m src.train --input data/processed/windows.npz --config config.yaml

# Seleccionar mejor modelo (por macro-F1)
python -m src.select_best --metric macro_f1 --registry_name cnn_signal_model

# Generar predicciones para backtest
python -m src.infer --model_name cnn_signal_model --stage Staging --input data/processed/windows.npz --output data/processed/predictions.parquet

# Ejecutar backtesting
python -m src.backtest --predictions data/processed/predictions.parquet --prices data/processed/clean_data.parquet --output reports/
```

### 4. Lanzar API REST

```bash
# Terminal 2 - API
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
# API disponible en: http://localhost:8000
# Documentación: http://localhost:8000/docs
```

### 5. Testing con Postman

#### Endpoint Health Check
```
GET http://localhost:8000/health
```

Respuesta esperada:
```json
{
  "status": "ok",
  "model_info": {
    "name": "cnn_signal_model",
    "version": "1",
    "stage": "Staging"
  }
}
```

#### Endpoint Predicción
```
POST http://localhost:8000/predict
Content-Type: application/json

{
  "n_bars": 256,
  "ticker": "SPY",
  "path_to_features": "data/processed/features.parquet"
}
```

Respuesta esperada:
```json
{
  "signal": "long",
  "probs": {
    "long": 0.65,
    "short": 0.15,
    "hold": 0.20
  },
  "used_model": {
    "name": "cnn_signal_model",
    "version": "1",
    "stage": "Staging"
  },
  "timestamp": "2025-11-07T10:30:45"
}
```

### 6. Dashboard Data Drift

```bash
# Terminal 3 - Streamlit
streamlit run dashboards/app_drift.py
# Dashboard disponible en: http://localhost:8501
```

## Configuración

Editar `config.yaml` para ajustar hiperparámetros:

```yaml
data:
  train_ratio: 0.6
  test_ratio: 0.2
  val_ratio: 0.2

features:
  momentum_windows: [5, 10, 20, 50]
  volatility_windows: [10, 20, 30]
  volume_windows: [5, 10, 20]

labeling:
  horizon: 5  # días
  threshold: 0.005  # 0.5%

model:
  window_size: 256
  batch_size: 256
  epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10
  
backtest:
  commission: 0.00125  # 0.125%
  borrow_rate: 0.0025  # 0.25% anual
  stop_loss: 0.02  # 2%
  take_profit: 0.03  # 3%
  n_shares: 100
```

## Scripts Auxiliares

```bash
# Promover modelo a Production
python scripts/promote_to_production.py --model_name cnn_signal_model --version 1

# Limpiar experimentos antiguos
python scripts/clean_mlruns.py --keep_best 10

# Generar reporte de métricas
python scripts/generate_report.py --output reports/metrics_report.html
```

## Estructura del Proyecto

```
deep-trading-mlops/
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml
├── data/
│   ├── raw/              # CSV originales
│   └── processed/         # Features y ventanas procesadas
├── src/
│   ├── data_loader.py    # Carga y limpieza de datos
│   ├── features.py        # Feature engineering (20+ features)
│   ├── labeling.py        # Generación de labels {long,short,hold}
│   ├── windows.py         # Creación de ventanas para CNN
│   ├── model_cnn.py       # Arquitectura CNN
│   ├── train.py           # Entrenamiento con MLflow
│   ├── select_best.py     # Selección mejor modelo
│   ├── infer.py           # Inferencia offline
│   ├── backtest.py        # Backtesting con costos
│   └── drift_utils.py     # Utilidades para data drift
├── api/
│   └── app.py             # FastAPI endpoints
├── dashboards/
│   └── app_drift.py       # Streamlit dashboard
├── scripts/
│   ├── run_mlflow_ui.sh
│   └── promote_to_production.py
└── mlruns/                # MLflow artifacts (auto-generado)
```

## Métricas Clave

### Modelo
- **Macro F1-Score**: >0.5 (objetivo)
- **Accuracy por clase**: Long, Short, Hold
- **Matriz de confusión**: Visualización en MLflow

### Backtesting
- **Sharpe Ratio**: >1.0 (objetivo)
- **Calmar Ratio**: >0.5 (objetivo)
- **Max Drawdown**: <15%
- **Win Rate**: >50%
- **Comisión total**: 0.125% por trade
- **Borrow cost**: 0.25% anual para shorts

### Data Drift
- **KS-Test p-value**: <0.05 indica drift significativo
- **Top-5 features**: Features con mayor drift
- **Periodo de monitoreo**: Train vs Test vs Validation

## Troubleshooting

### Error: "No module named 'mlflow'"
```bash
pip install mlflow==2.8.0
```

### Error: "CUDA not available"
El modelo funciona en CPU pero es más lento. Para GPU:
```bash
pip install tensorflow-gpu
```

### Error: "Port already in use"
Cambiar puertos en los comandos:
- MLflow: `--port 5001`
- API: `--port 8001`
- Streamlit: `--port 8502`

## Licencia

MIT License - Ver LICENSE para detalles

## Autor

Deep Trading MLOps System v1.0