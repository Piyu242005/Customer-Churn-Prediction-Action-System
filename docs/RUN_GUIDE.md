# Step-by-Step Guide to Run the MLP Churn Classifier

This guide covers how to set up and run the **Neural Network Churn Classifier** project end-to-end.

---

## 📋 Prerequisites

- **Python 3.10** (or 3.11+)
- **Git** installed
- **Virtual environment** (recommended)
- **Dataset:** `data/Business_Analytics_Dataset_10000_Rows.csv`

---

## 🚀 Quick Start (5 steps)

### Step 1: Clone & Navigate to Project

```bash
git clone <your-repo-url>
cd Neural-Network-Churn-Classifier--MLP
```

### Step 2: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create Database and Train Model

```bash
# Step 4a: Initialize SQLite database
python src/database.py
# ✅ Creates: data/churn_data.db

# Step 4b: Train model (creates MLflow logs)
python src/pipeline.py
# ✅ Creates: artifacts/mlp_churn_classifier_final.pth
#            mlflow.db (MLflow experiment tracking)
#            mlp_training_history.json
```

### Step 5: Launch Services

**Option A: FastAPI + Streamlit (Recommended)**

Open **two separate terminals** in the project root:

**Terminal 1 — FastAPI Backend (port 5000):**
```bash
uvicorn serving.app:app --port 5000
```
- 📍 API runs at: `http://127.0.0.1:5000`
- 📍 API docs at: `http://127.0.0.1:5000/docs`

**Terminal 2 — Streamlit Dashboard (port 8501):**
```bash
streamlit run serving/dashboard.py
```
- 📍 Dashboard opens at: `http://localhost:8501`

---

## 🐳 Running with Docker (Alternative)

### Build and Run

```bash
docker-compose up --build
```

This launches:
- **Streamlit Dashboard** on port `8501`
- **FastAPI Backend** on port `5000`
- **SQLite Database** (mounted volume)

### Stop Services

```bash
docker-compose down
```

---

## 📁 Project Structure

```
Neural-Network-Churn-Classifier--MLP/
├── src/
│   ├── config.py                 # Central config (PATHS, MODEL_CONFIG, etc.)
│   ├── database.py              # SQLAlchemy ORM → creates churn_data.db
│   ├── data_preprocessing.py    # Data cleaning & feature engineering
│   ├── model.py                 # MLP architecture
│   ├── pipeline.py              # Orchestrates training & artifact saving
│   ├── train.py                 # MLflow logging & hyperparameter optimization
│   └── evaluate.py              # Evaluation metrics
│
├── serving/
│   ├── app.py                   # FastAPI endpoints
│   └── dashboard.py             # Streamlit web interface
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── data/
│   ├── Business_Analytics_Dataset_10000_Rows.csv
│   └── churn_data.db            # Created by src/database.py
│
├── artifacts/                   # Generated after training
│   ├── mlp_churn_classifier_final.pth
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── label_encoders.pkl
│
├── docs/
│   ├── RUN_GUIDE.md            # This file
│   ├── QUICKSTART.md
│   ├── DEPLOYMENT.md
│   └── ADVANCED_MODELS.md
│
└── requirements.txt
```

---

## ⚙️ Detailed Breakdown

### 1️⃣ Database Initialization

**Command:**
```bash
python src/database.py
```

**What it does:**
- Reads `data/Business_Analytics_Dataset_10000_Rows.csv`
- Creates SQLite database at `data/churn_data.db`
- Ingests data into `customers` table

**Time:** ~5-10 seconds

---

### 2️⃣ Model Training & Pipeline

**Command:**
```bash
python src/pipeline.py
```

**What it does:**
1. Loads data from `data/churn_data.db`
2. Preprocesses features (encoding, scaling, SMOTE)
3. Trains MLP classifier with Optuna hyperparameter optimization
4. Logs metrics to MLflow (`mlflow.db`)
5. Saves artifacts to `artifacts/` directory:
   - `mlp_churn_classifier_final.pth` (model weights)
   - `scaler.pkl` (feature scaler)
   - `feature_names.pkl` (feature list)
   - `label_encoders.pkl` (categorical encoders)

**Time:** ~3-5 minutes (CPU) / ~1 minute (GPU)

**Output files:**
- `mlp_training_history.json` — Training curves
- `mlflow.db` — Experiment tracking

---

### 3️⃣ FastAPI Server

**Command:**
```bash
uvicorn serving.app:app --port 5000
```

**What it does:**
- Loads trained model from `artifacts/`
- Exposes REST API endpoints:
  - `POST /predict` — Make single predictions
  - `POST /predict_batch` — Batch predictions
  - `GET /health` — Health check
  - `GET /docs` — Swagger documentation

**Example request:**
```bash
curl -X POST "http://127.0.0.1:5000/predict" \
  -H "Content-Type: application/json" \
  -d "{
    \"age\": 35,
    \"tenure_months\": 24,
    \"monthly_bill\": 85.5,
    \"total_charges\": 2052.0,
    \"contract\": \"Month-to-month\",
    \"internet_service\": \"Fiber optic\"
  }"
```

---

### 4️⃣ Streamlit Dashboard

**Command (from project root):**
```bash
streamlit run serving/dashboard.py
```

**What it shows:**
- 📊 Model performance metrics
- 📈 Training history charts
- 🔮 Single/batch prediction interface
- 📉 Feature importance (SHAP)
- 📋 Customer overview stats

**Access at:** `http://localhost:8501`

---

## 🔧 Configuration

Edit `src/config.py` to customize:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 50 | Number of training epochs |
| `LEARNING_RATE` | 0.001 | Optimizer learning rate |
| `HIDDEN_DIMS` | [256, 128, 64] | MLP layer dimensions |
| `DROPOUT_RATE` | 0.3 | Dropout regularization |

**Example:**
```python
# src/config.py
MODEL_CONFIG = {
    'hidden_dims': [512, 256, 128, 64],  # Deeper network
    'dropout_rate': 0.4,
}

TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.0005,
    'epochs': 100,
}
```

Then retrain:
```bash
python src/pipeline.py
```

---

## ✅ Verification

Check that everything is running correctly:

```bash
# 1. Database exists
ls -la data/churn_data.db

# 2. Artifacts created
ls -la artifacts/

# 3. API responds
curl http://127.0.0.1:5000/health
# Expected: {"status": "ok"}

# 4. Dashboard loads
# Open: http://localhost:8501
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: data/churn_data.db` | Run `python src/database.py` first |
| `FileNotFoundError: artifacts/mlp_churn_classifier_final.pth` | Run `python src/pipeline.py` first |
| `Connection refused on port 5000` | Check if another process is using it: `lsof -i :5000` |
| `Streamlit connection error` | Ensure FastAPI is running; check firewall |
| CUDA out of memory | Edit `config.py`: reduce `BATCH_SIZE` to 16 or 8 |
| `pandas.errors.OptionError` | Upgrade pandas: `pip install --upgrade pandas` |

---

## 🎯 Common Workflows

### Workflow 1: Train Once, Serve Forever

```bash
# One-time setup
python src/database.py
python src/pipeline.py

# Then run both servers indefinitely
# Terminal 1:
uvicorn serving.app:app --port 5000

# Terminal 2:
streamlit run serving/dashboard.py
```

### Workflow 2: Retrain with New Hyperparameters

```bash
# Edit config.py

python src/pipeline.py     # Retrains and overwrites artifacts/

# Servers auto-reload the new model
```

### Workflow 3: Batch Predictions from CSV

```python
# Create predict_batch.py
import json
import requests

with open('predictions.csv', 'w') as f:
    f.write('customer_id,churn_probability\n')

with open('customers.csv') as input_file:
    for line in input_file:
        customer = json.loads(line)
        resp = requests.post('http://127.0.0.1:5000/predict', json=customer)
        prob = resp.json()['churn_probability']
        f.write(f"{customer['id']},{prob}\n")
```

---

## 📊 Expected Performance

| Metric | Expected |
|--------|----------|
| Test Accuracy | ~89% |
| F1-Score | ~0.85-0.88 |
| ROC-AUC | ~0.92+ |
| Training Time (CPU) | 3-5 minutes |
| Training Time (GPU) | ~1 minute |
| Prediction Latency | <50ms (single) |

---

## 📚 Next Steps

1. ✅ Complete the quick start above
2. ✅ Explore API endpoints at `http://127.0.0.1:5000/docs`
3. ✅ Try predictions in the Streamlit dashboard
4. ✅ Read `docs/ADVANCED_MODELS.md` for multi-task learning & sequence models
5. ✅ Read `docs/DEPLOYMENT.md` for production deployment

---

## 🤝 Need Help?

- **API Issues:** Check `http://127.0.0.1:5000/docs` for interactive API testing
- **Dashboard Issues:** Check terminal logs for Streamlit errors
- **Database Issues:** Delete `data/churn_data.db` and rerun `python src/database.py`
- **Training Issues:** Check `mlflow.db` or open MLflow UI: `mlflow ui`

---

**Version:** 2026-03-15
**Last Updated:** 2026-03-15
