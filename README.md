# 🚀 Deep Learning Trading con MLOps - Proyecto QQQ

Proyecto completo de trading algorítmico usando Deep Learning (CNN) con buenas prácticas de MLOps.

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Flujo de Ejecución](#flujo-de-ejecución)
- [MLflow UI](#mlflow-ui)
- [API FastAPI](#api-fastapi)
- [Dashboard de Drift](#dashboard-de-drift)
- [Backtest](#backtest)
- [Parámetros Configurables](#parámetros-configurables)

---

## 📖 Descripción

Este proyecto predice señales de trading **{long, short, hold}** para QQQ (NASDAQ-100 ETF) usando:

- **CNN (Convolutional Neural Network)** entrenada sobre ventanas de 256 días con 20+ features técnicos
- **MLflow** para tracking de experimentos y Model Registry
- **FastAPI** para deployment como API REST
- **Streamlit** para monitoreo de data drift
- **Backtest realista** con costos de comisión (0.125%) y borrow (0.25% anual)

**Métricas clave**: Macro-F1, Sharpe, Sortino, Calmar, Max Drawdown

---

## 🛠️ Instalación

### Requisitos

- Python 3.8+
- pip

### Pasos
```bash
# 1. Clonar repositorio (o crear directorio)
mkdir deep-trading-mlops && cd deep-trading-mlops

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Crear estructura de directorios
mkdir -p data/{raw,processed} results mlruns

# 5. Descargar datos de QQQ (última línea descarga ~15 años)
python -c "import yfinance as yf; yf.download('QQQ', start='2009-01-01', end='2024-12-31').to_csv('data/raw/QQQ_daily.csv')"
```

---

## 📁 Estructura del Proyecto
```
deep-trading-mlops/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/                # QQQ_daily.csv
│   └── processed/          # features.parquet, splits, scaler, tensores
├── src/
│   ├── data_loader.py      # Carga y limpieza
│   ├── features.py         # 20+ features técnicos
│   ├── labeling.py         # Etiquetas {long, hold, short}
│   ├── windows.py          # Tensores 3D para CNN
│   ├── model_cnn.py        # Arquitectura CNN
│   ├── train.py            # Entrenamiento + MLflow
│   ├── select_best.py      # Selección de mejor modelo
│   ├── infer.py            # Inferencia offline
│   ├── backtest.py         # Backtest con costos
│   └── drift_utils.py      # KS-test para drift
├── api/
│   └── app.py              # FastAPI /health y /predict
├── dashboards/
│   └── app_drift.py        # Streamlit drift monitor
├── scripts/
│   └── run_mlflow_ui.sh    # Script para MLflow UI
└── results/                # Plots, reports, modelos
```

---

## 🎯 Flujo de Ejecución

### Paso 1: Preparación de Datos
```bash
# Limpieza y splits 60/20/20
python src/data_loader.py

# Calcula 20+ features técnicos
python src/features.py

# Genera etiquetas {long, hold, short}
python src/labeling.py

# Crea ventanas y tensores 3D
python src/windows.py
```

**Output esperado**:
- `data/processed/features.parquet` (features + labels)
- `data/processed/X_train.npy`, `y_train.npy`, etc.
- `data/processed/scaler.pkl`

---

### Paso 2: Entrenamiento con MLflow
```bash
# Terminal 1: Iniciar MLflow UI
bash scripts/run_mlflow_ui.sh
# Abre http://localhost:5000

# Terminal 2: Entrenar modelo
python src/train.py
```

**Durante el entrenamiento**:
- MLflow loggea: params (W, H, τ, lr, etc.), métricas (accuracy, macro-F1), plots
- Modelo se guarda en Model Registry como `cnn_signal_model`

**Métricas clave**:
- `val_f1_macro`: F1 promedio entre las 3 clases (métrica de selección)
- `val_accuracy`: Exactitud en validación
- `val_f1_long`, `val_f1_hold`, `val_f1_short`: F1 por clase

---

### Paso 3: Selección del Mejor Modelo
```bash
python src/select_best.py
```

**Qué hace**:
1. Busca el run con mejor `val_f1_macro` en el experimento
2. Promociona esa versión del modelo a **Staging** en Model Registry
3. Archiva versiones anteriores

**Output**:
```
✅ Modelo en Staging listo para producción
   Model: cnn_signal_model
   Version: 2
   Macro-F1: 0.4823
```

---

### Paso 4: Desplegar API
```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints**:
- `GET /health`: Health check
- `POST /predict`: Predicción de señal

**Probar con Postman** (ver sección [API FastAPI](#api-fastapi))

---

### Paso 5: Dashboard de Drift
```bash
streamlit run dashboards/app_drift.py
```

**Abre**: http://localhost:8501

**Visualizaciones**:
- KS-test p-values por feature (train vs test, train vs val)
- Top-5 features con mayor drift
- Histogramas comparativos

---

### Paso 6: Backtest
```bash
# Primero genera señales (opcional si ya se corrió)
python src/infer.py

# Ejecuta backtest
python src/backtest.py
```

**Output**:
- `results/backtest_report.txt`: Métricas completas
- `results/equity_curve.png`: Curva de equity
- `results/returns_distribution.png`: Distribución de retornos por trade
- `results/trades.csv`: Detalle de cada trade

**Métricas**:
- Retorno Total
- Sharpe Ratio (anualizado)
- Sortino Ratio
- Calmar Ratio
- Max Drawdown
- Win Rate
- Número de trades

---

## 🌐 MLflow UI
```bash
# Terminal dedicada
bash scripts/run_mlflow_ui.sh
```

**Acceso**: http://localhost:5000

**Funcionalidades**:
- Ver todos los experimentos y runs
- Comparar métricas entre runs
- Visualizar artifacts (plots, confusion matrix)
- Model Registry: versiones del modelo, transiciones (None → Staging → Production)

---

## 🔌 API FastAPI

### Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

**Respuesta**:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### 2. Predict

**Request (Postman o curl)**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "n_bars": 256
  }'
```

**Payload JSON** (para Postman):
```json
{
  "n_bars": 256
}
```

**Respuesta ejemplo**:
```json
{
  "signal": 0,
  "signal_name": "long",
  "probabilities": {
    "long": 0.6234,
    "hold": 0.2891,
    "short": 0.0875
  },
  "metadata": {
    "model_name": "cnn_signal_model",
    "n_bars_used": 256,
    "last_date": "2024-12-15",
    "n_features": 42
  }
}
```

**Interpretación**:
- `signal`: 0=long, 1=hold, 2=short
- `probabilities`: Confianza del modelo en cada clase
- `metadata`: Info del modelo y datos usados

---

## 📊 Dashboard de Drift
```bash
streamlit run dashboards/app_drift.py
```

**Tabs**:

1. **Train vs Test**: Compara distribución de features entre train y test
2. **Train vs Val**: Compara train con validación
3. **Interpretación**: Guía de acción según nivel de drift

**KS-test**:
- `p-value < 0.05` → Drift detectado (distribución cambió significativamente)
- Features con drift alto → Candidatos a revisar o remover

**Top-5 features con mayor drift**: Visualización con histogramas comparativos

---

## 💹 Backtest
```bash
python src/backtest.py
```

### Lógica del Backtest

1. **Señales**: Usa predicciones del modelo en validación
2. **Entry/Exit**: 
   - Long: entrada cuando `signal=0`, salida cuando `signal≠0` o SL/TP
   - Short: entrada cuando `signal=2`, salida cuando `signal≠2` o SL/TP
3. **Costos**:
   - **Comisión**: 0.125% por lado (0.25% total por round-trip)
   - **Borrow cost**: 0.25% anual prorrateado por días en cortos
4. **Stop Loss / Take Profit**: Configurable en `config.yaml`

### Métricas Calculadas

- **Retorno Total**: (Equity final / Equity inicial) - 1
- **Sharpe Ratio**: (Retorno promedio / Desv. estándar) × √252
- **Sortino Ratio**: (Retorno promedio / Desv. de retornos negativos) × √252
- **Calmar Ratio**: Retorno total / |Max Drawdown|
- **Max Drawdown**: Caída máxima desde un pico de equity
- **Win Rate**: % de trades ganadores

---

## ⚙️ Parámetros Configurables

Edita `config.yaml` para ajustar:
```yaml
labeling:
  H: 5              # Días hacia adelante para etiquetas
  tau: 0.005        # Umbral (0.5%) para long/short

windows:
  W: 256            # Tamaño de ventana (secuencia)

model:
  filters: [64, 128, 64]
  kernel_sizes: [5, 3, 3]
  dropout: 0.3

training:
  batch_size: 256
  epochs: 50
  learning_rate: 0.001

backtest:
  commission: 0.00125
  borrow_rate: 0.0025
  stop_loss_pct: 0.02
  take_profit_pct: 0.03
```

**Tip**: Experimenta con diferentes valores de `H` y `τ` para ver cómo afecta el balance de clases.

---

## ✅ Checklist de Aceptación

- [x] ≥20 features documentadas (ver `feature_names.txt`)
- [x] Splits 60/20/20 sin look-ahead (cronológicos)
- [x] Normalización: fit en train, aplicado a test/val
- [x] CNN con class_weights + early stopping
- [x] MLflow: params, métricas, artefactos y registry `cnn_signal_model`
- [x] Mejor run por macro-F1 en Staging
- [x] API `/health` y `/predict` funcional
- [x] Backtest con 0.125% comisión y 0.25% borrow anual
- [x] Streamlit drift con KS-test y Top-5 features
- [x] README con comandos exactos

---

## 📚 Glosario Breve

| Término | Definición |
|---------|-----------|
| **OHLCV** | Open, High, Low, Close, Volume (datos de velas) |
| **Feature** | Variable derivada de datos crudos (ej: RSI, SMA) |
| **Window (W)** | Secuencia de días que el modelo analiza (ej: 256) |
| **Horizon (H)** | Días hacia adelante para calcular etiquetas (ej: 5) |
| **Threshold (τ)** | Umbral para decidir long/short (ej: 0.5%) |
| **Label** | Etiqueta de clase {0:long, 1:hold, 2:short} |
| **CNN** | Convolutional Neural Network (red neuronal convolucional) |
| **Macro-F1** | F1-score promediado entre todas las clases |
| **Class Weights** | Pesos para balancear clases desbalanceadas |
| **Drift** | Cambio en la distribución estadística de features |
| **KS-test** | Kolmogorov-Smirnov test (detecta cambios en distribuciones) |
| **MLflow** | Plataforma para tracking y gestión de modelos ML |
| **Model Registry** | Repositorio centralizado de versiones de modelos |
| **Staging** | Etapa en Model Registry (modelo validado, listo para producción) |
| **Backtest** | Simulación histórica de estrategia de trading |
| **Sharpe Ratio** | Retorno ajustado por riesgo (volatilidad total) |
| **Sortino Ratio** | Retorno ajustado por riesgo negativo (downside) |
| **Max Drawdown** | Caída máxima desde un pico de capital |
| **Win Rate** | % de trades ganadores |
| **Stop Loss (SL)** | Salida automática por pérdida máxima |
| **Take Profit (TP)** | Salida automática por ganancia objetivo |
| **Borrow Cost** | Costo de préstamo de acciones en ventas en corto |

---

## 🎓 Notas Finales

- **Reproducibilidad**: Seeds fijados en `train.py` (seed=42)
- **Datos**: ~15 años de QQQ (ajusta en `config.yaml` si necesitas más/menos)
- **Experimentación**: Corre múltiples entrenamientos cambiando params, luego usa `select_best.py`
- **Producción**: Migra API a Docker/Kubernetes para deployment real

**¡Éxito en tu proyecto!** 🚀

---

**Contacto**: [Tu nombre/email]  
**Licencia**: MIT
```

---

## 📬 Ejemplo de Payload/Respuesta para Postman

### Request (POST /predict)

**URL**: `http://localhost:8000/predict`

**Method**: POST

**Headers**:
```
Content-Type: application/json