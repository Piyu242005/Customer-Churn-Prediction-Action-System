

## 🧠 Neural Network Churn Classifier — Project Overview

### What Is It?

This is a **production-ready customer churn prediction system** built by Piyush Ramteke. It uses a custom **Multi-Layer Perceptron (MLP)** neural network in PyTorch to predict whether a customer is likely to stop doing business with a company ("churn"). The model is trained on 10,000 customer records and achieves **89% accuracy**.

---

### 🎯 The Business Problem

**Customer churn** = customers leaving or becoming inactive. This is a huge business challenge because:
- Retaining a customer costs 5–25x less than acquiring a new one
- Lost customers mean lost recurring revenue
- Companies need to know *who* is at risk before they leave, so they can intervene

---

### 🏗️ How the Model Works

The MLP takes **16 engineered features** as input and outputs a churn probability (0 = active, 1 = churned):

```
Input (16 features)
    → Hidden Layer 1: 128 neurons + ReLU + Dropout(30%)
    → Hidden Layer 2: 64 neurons  + ReLU + Dropout(30%)
    → Hidden Layer 3: 32 neurons  + ReLU + Dropout(30%)
    → Output: 1 neuron + Sigmoid → Churn probability
```

**Key design choices:**
- **ReLU activation** for non-linearity and fast training
- **30% Dropout** to prevent overfitting
- **Adam optimizer** with learning rate scheduling
- **Binary Cross-Entropy** as the loss function
- **Early stopping** (patience = 15 epochs) to stop training at the right time

---

### 📊 Features Used

The dataset is transactional, so features are engineered from raw purchase data:

**Numerical:** total orders, total revenue, average revenue, revenue std, total profit, average profit, average discount, total quantity, average quantity, days since last purchase, customer lifetime (days), purchase frequency

**Categorical (encoded):** region, product category, customer segment, payment method

**Churn label** is defined as: inactive for 90+ days, OR in the bottom 30% profit tier, OR fewer than 3 orders with 60+ days of inactivity.

---

### 📈 Performance

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **MLP (this project)** | **89%** | **0.86** | **0.92** |
| XGBoost | 87% | 0.83 | 0.90 |
| Random Forest | 85% | 0.81 | 0.88 |
| Logistic Regression | 78% | 0.73 | 0.82 |

It correctly identifies **829 out of 942 churned customers** (88% recall) with 87% precision.

---

### 🗂️ Project Structure

This is not just a notebook — it's a full ML engineering project with:

- **`Neural_Network_Churn_Classifier.ipynb`** — the full training + analysis notebook
- **`model.py`** — MLP architecture definition
- **`train.py`** — automated training script
- **`evaluate.py`** — metrics and visualizations
- **`app.py`** — Flask REST API for real-time predictions
- **`dashboard.py`** — Streamlit interactive dashboard
- **`pipeline.py`** — end-to-end automation
- **`explainability.py`** — SHAP/LIME model interpretability
- **`Dockerfile` + `docker-compose.yml`** — containerized deployment

---

### 🚀 Deployment Options

1. **Jupyter Notebook** — for learning and experimentation
2. **Flask REST API** — send customer data via HTTP POST, get back a churn probability + risk level
3. **Streamlit Dashboard** — interactive UI for demos and business stakeholders
4. **Docker** — containerized and cloud-ready (AWS, Azure, GCP, Heroku)

---

### 💼 Real-World Use Cases

This kind of system is used in:
- **E-commerce** — identify customers about to stop buying
- **SaaS** — predict subscription cancellations before they happen
- **Telecom** — forecast contract non-renewals
- **Banking** — detect account closure risk

---

### 🌟 What Makes It Stand Out

This goes well beyond a typical academic notebook — it includes modular code, saved model artifacts (`*.pth`, `*.pkl`), a REST API, a visual dashboard, Docker support, and SHAP-based model explainability. It's a strong end-to-end portfolio project demonstrating both ML engineering and MLOps awareness.
