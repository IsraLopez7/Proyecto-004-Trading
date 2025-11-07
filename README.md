# Deep Trading MLOps with CNN

Sistema completo de trading sistemático usando CNN para predicción de señales {long, short, hold} con seguimiento de experimentos en MLflow, API REST para predicciones y dashboard de data drift.

## 🎯 Características Principales

- **Deep Learning**: CNN con Keras/TensorFlow para secuencias temporales
- **MLOps Completo**: Tracking con MLflow, Model Registry y gestión de modelos
- **Feature Engineering**: 20+ features técnicos de momentum, volatilidad y volumen
- **API REST**: FastAPI para inferencia en producción (testeable con Postman)
- **Data Drift**: Dashboard Streamlit con KS-test y análisis de distribuciones
- **Backtesting Realista**: Con comisiones (0.125%) y borrow rate (0.25% anual)
- **Reproducibilidad**: Seeds fijos, configuración centralizada

## 📁 Estructura del Proyecto

```
deep-trading-mlops/
  README.md
  requirements.txt
  .env.example
  config.yaml                      # Hiperparámetros por defecto
  data/
    raw/                           # CSV original (QQQ.csv)
    processed/                     # features.parquet, labels.parquet, etc.
  src/
    data_loader.py                 # Carga, limpieza, splits cronológicos
    features.py                    # Feature engineering (≥20 features)
    labeling.py                    # Labels {long, short, hold}
    windows.py                     # Creación de ventanas para CNN
    model_cnn.py                   # Arquitectura CNN
    train.py                       # Entrenamiento con MLflow
    select_best.py                 # Selección del mejor modelo
    infer.py                       # Inferencia offline
    backtest.py                    # Backtesting con costos
    drift_utils.py                 # Utilidades de data drift
  api/
    app.py                         # FastAPI con /health y /predict
  dashboards/
    app_drift.py                   # Dashboard de drift en Streamlit
  scripts/
    run_mlflow_ui.sh              # Lanza MLflow UI
    promote_to_staging.py         # Promoción de modelos en registry
    download_data.py              # Descarga datos históricos de QQQ
  notebooks/                       # (Opcional) Exploración
```

## 🔧 Instalación

### 1. Crear entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configurar variables de entorno (opcional)

```bash
cp .env.example .env
# Editar .env si necesitas customizar rutas o parámetros
```

## 🚀 Guía de Uso Completo

### Paso 1: Descargar Datos Históricos de QQQ

```bash
python scripts/download_data.py
```

**Output esperado**: Descarga ~15 años de datos diarios de QQQ y guarda en `data/raw/QQQ.csv`

### Paso 2: Lanzar MLflow UI (terminal separada)

```bash
bash scripts/run_mlflow_ui.sh
# O directamente: mlflow ui --port 5000
```

**Acceder a**: http://localhost:5000

### Paso 3: Pipeline de Datos y Entrenamiento

#### 3.1 Cargar y limpiar datos

```bash
python -m src.data_loader --csv data/raw/QQQ.csv
```

**Output**:
- `data/processed/data_clean.parquet`
- Reporta estadísticas: # filas, fechas, splits 60/20/20

#### 3.2 Generar features (≥20)

```bash
python -m src.features
```

**Output**:
- `data/processed/features.parquet`
- Reporta features creados: SMA, EMA, RSI, MACD, ATR, Bollinger, ADX, etc.

#### 3.3 Etiquetar señales {long, short, hold}

```bash
python -m src.labeling
```

**Output**:
- `data/processed/labels.parquet`
- Reporta distribución de clases (manejar desbalanceo con class_weight)

#### 3.4 Crear ventanas para CNN

```bash
python -m src.windows
```

**Output**:
- `data/processed/windows_train.npz`
- `data/processed/windows_val.npz`
- `data/processed/windows_test.npz`
- Reporta shapes: (N_samples, W, n_features)

#### 3.5 Entrenar modelo con MLflow

```bash
python -m src.train
```

**Output**:
- Modelo registrado en MLflow
- Métricas: accuracy, macro F1, F1 por clase
- Artefactos: confusion matrix, loss curves
- Nombre en Model Registry: `cnn_signal_model`

**Ver en MLflow UI**: http://localhost:5000

#### 3.6 Seleccionar el mejor modelo

```bash
python -m src.select_best
```

**Output**:
- Identifica el run con mejor macro-F1 en validación
- Promociona a "Staging" en Model Registry
- Imprime: Run ID, macro-F1, parámetros

#### 3.7 Generar señales para backtesting

```bash
python -m src.infer
```

**Output**:
- `data/processed/signals_test.parquet`
- Señales predichas para el conjunto de test

#### 3.8 Ejecutar backtesting

```bash
python -m src.backtest
```

**Output**:
- `data/processed/backtest_results.json` (métricas)
- `data/processed/equity_curve.png`
- `data/processed/returns_distribution.png`
- Métricas: Sharpe, Sortino, Calmar, Max DD, Win Rate, #Trades

### Paso 4: API de Predicción (FastAPI)

#### 4.1 Iniciar servidor

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Acceder a**:
- Docs interactivas: http://localhost:8000/docs
- Health check: http://localhost:8000/health

#### 4.2 Test con Postman

**Endpoint**: `POST http://localhost:8000/predict`

**Request Body** (JSON):
```json
{
  "n_bars": 256,
  "ticker": "QQQ",
  "path_to_features": "data/processed/features.parquet"
}
```

**Response Esperada** (JSON):
```json
{
  "signal": "long",
  "probs": {
    "long": 0.67,
    "short": 0.15,
    "hold": 0.18
  },
  "used_model": {
    "name": "cnn_signal_model",
    "version": "2",
    "stage": "Staging"
  },
  "timestamp": "2025-11-07T14:32:15.123456"
}
```

**Endpoint**: `GET http://localhost:8000/health`

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Paso 5: Dashboard de Data Drift (Streamlit)

```bash
streamlit run dashboards/app_drift.py
```

**Acceder a**: http://localhost:8501

**Funcionalidades**:
- Vista temporal de distribuciones por periodo (train/test/val)
- Tabla de KS-test p-values por feature
- Flags "Drift detected" cuando p < 0.05
- Top-5 features con mayor drift
- Interpretación de drift detectado

## 📊 Parámetros por Defecto (config.yaml)

```yaml
data:
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2

features:
  window_short: [5, 10, 20]
  window_long: [50, 100, 200]

labeling:
  H: 5                    # Horizonte de predicción (días)
  tau: 0.005              # Umbral (0.5%)

model:
  W: 256                  # Longitud de ventana para CNN
  batch_size: 256
  epochs: 50
  learning_rate: 0.001
  filters: [64, 128, 256]
  kernel_size: 3
  dropout: 0.3
  seed: 42

backtest:
  commission: 0.00125     # 0.125% por lado
  borrow_rate: 0.0025     # 0.25% anual
  stop_loss: 0.02         # 2%
  take_profit: 0.04       # 4%
```

## 📈 Features Implementados (≥20)

### Momentum (8)
1. SMA_5, SMA_10, SMA_20, SMA_50
2. EMA_12, EMA_26
3. RSI_14
4. ROC_10

### Volatilidad (7)
5. ATR_14
6. BB_bandwidth_20
7. BB_pct_b_20 (Posición relativa en Bollinger Bands)
8. Volatility_20 (rolling std)
9. High_Low_Range
10. True_Range
11. Volatility_Ratio (vol corto / vol largo)

### Volumen (5)
12. Volume_SMA_20
13. Volume_ZScore
14. OBV (On-Balance Volume)
15. MFI_14 (Money Flow Index)
16. Volume_ROC_5

### Otros Indicadores (5)
17. MACD
18. MACD_Signal
19. MACD_Hist
20. Stochastic_K_14
21. Stochastic_D_3

### Lags y Derivados (4)
22. Return_1d
23. Return_5d
24. Return_10d
25. Log_Return_1d

**Total**: 25 features

## 🧪 Métricas de Evaluación

### Modelo
- **Accuracy**: Global
- **Macro F1**: Promedio de F1 por clase (métrica principal)
- **F1 por clase**: long, short, hold
- **Confusion Matrix**: Visualización de predicciones

### Backtesting
- **Total Return**: Retorno acumulado
- **Sharpe Ratio**: Return ajustado por volatilidad
- **Sortino Ratio**: Return ajustado por downside risk
- **Calmar Ratio**: Return/Max Drawdown
- **Max Drawdown**: Pérdida máxima desde peak
- **Win Rate**: % de trades ganadores
- **# Trades**: Total de operaciones

## 🔄 Gestión de Modelos MLflow

### Promoción manual a Production

```bash
python scripts/promote_to_staging.py --run_id <RUN_ID>
```

### Ver modelos registrados

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
models = client.search_registered_models()
for model in models:
    print(f"Name: {model.name}")
    for version in model.latest_versions:
        print(f"  Version: {version.version}, Stage: {version.current_stage}")
```

## 📝 Notas Técnicas

### Splits Cronológicos
- **Train**: 60% primeros datos (fit normalización)
- **Validation**: 20% siguientes (selección de modelo)
- **Test**: 20% últimos (backtesting)
- **Sin look-ahead bias**: Normalización solo con estadísticas de train

### Manejo de Desbalanceo
- Cálculo automático de `class_weight` en training
- Ajuste de umbrales de probabilidad si es necesario
- Oversampling/undersampling opcional (comentado en código)

### Costos de Trading
- **Comisión**: 0.125% por entrada y 0.125% por salida
- **Borrow Rate**: 0.25% anual para posiciones short (prorrateado por días)
- **Slippage**: No implementado (puede agregarse)

### Data Drift
- **KS-test**: Kolmogorov-Smirnov para comparar distribuciones
- **Umbral**: p-value < 0.05 indica drift significativo
- **Acción**: Re-entrenar modelo si drift detectado en features críticos

## 🐛 Troubleshooting

### Error: "No module named 'src'"
```bash
# Asegúrate de estar en el directorio raíz del proyecto
cd /home/user/Proyecto-004-Trading
python -m src.data_loader --csv data/raw/QQQ.csv
```

### MLflow UI no carga
```bash
# Verificar que no haya otro proceso en puerto 5000
lsof -i :5000
# Cambiar puerto si es necesario
mlflow ui --port 5001
```

### API no carga modelo
```bash
# Verificar que existe modelo en registry
python -c "from mlflow.tracking import MlflowClient; print(MlflowClient().search_registered_models())"
# Si no existe, ejecutar train.py y select_best.py
```

### Error de memoria en training
```bash
# Reducir batch_size en config.yaml
# O reducir W (longitud de ventana)
# O usar gradient checkpointing
```

## 📚 Referencias

- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Keras CNN Guide](https://keras.io/examples/vision/)

## 📄 Licencia

MIT License - Ver archivo LICENSE

## 👤 Autor

Deep Trading MLOps Project - Ingeniería de ML para Trading Sistemático

---

## ✅ Checklist de Aceptación

- [x] ≥20 features documentados
- [x] Splits 60/20/20 sin contaminación
- [x] Normalización fit en train y aplicada en test/val
- [x] CNN entrenada con class weights y early stopping
- [x] MLflow registra params+metrics+artefactos
- [x] Modelo en registry `cnn_signal_model`
- [x] Script selección mejor run por macro-F1
- [x] API FastAPI con GET /health y POST /predict
- [x] Backtest con comisión 0.125% y borrow 0.25%
- [x] Streamlit drift dashboard con KS-test
- [x] README con pasos exactos
- [x] Código modular sin menús

**Proyecto listo para producción** ✨
