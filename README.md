<!-- HEADER -->
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=200&section=header&text=Customer%20Churn%20Prediction%20System&fontSize=40&fontAlignY=35&fontColor=ffffff&desc=AI-Powered%20Retention%20Intelligence&descAlignY=55&descAlign=50" width="100%"/>

### 📈 Predict Risk. Uncover Reasons. Take Action. 🚀

<br>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](#)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-00FF66.svg?style=for-the-badge&logo=scikit-learn&logoColor=black)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg?style=for-the-badge&logo=github&logoColor=white)](#)

**An end-to-end AI system that predicts customer churn, identifies root causes via SHAP explainability, and recommends actionable retention strategies.**

</div>

---

## 📌 Overview

Customer churn is a silent killer of Annual Recurring Revenue (ARR). Acquiring a new customer costs up to **5x more** than retaining an existing one. 

This project goes beyond traditional machine learning by transforming raw predictions into an **Action System**. It doesn't just predict *who* will churn; it identifies *why* they are leaving and prescribes exactly *what to do* about it.

---

## ✨ System Workflow

<div align="center">
  <img src="https://mermaid.ink/img/pako:eNptkE1qwzAUhT-l6KqDrHbtZGFSSpOWkIQuxE0Qx0mNInwk-XCa0n_v2B90s-iue-45H-iS8KAQk62A29BvDUihz-s4D12rT9KzG8f2A9Z6228sX4tP2Ej3T0_j4y9Yl-yHl4mP9E-6w0w0O-lEDzRYW8yO0ZtD4_lV9a7AOPmP8PI1xUXjv8gBDBcXzMUdQVkITsofmVmAa-1zVRcfuBaa8Q?type=png" alt="Workflow" width="80%">
</div>

---

## 🚀 Key Features

| Feature | Description | Business Impact |
| :--- | :--- | :--- |
| **🎯 Churn Prediction** | XGBoost algorithm outputs continuous churn probability (0–100%). | Anticipate churn before it happens. |
| **🚥 Risk Segmentation** | Categorizes customers into **Low**, **Medium**, and **High** risk. | Triage support ticket priority easily. |
| **🧠 Explainability (SHAP)**| Highlights user-specific factors driving churn (e.g., poor support). | Understand exactly *why* users leave. |
| **💡 Action Recommender** | Maps localized risk factors to retention tasks (e.g., give a discount). | Shift from reactive to **proactive**. |
| **📊 Interactive UI** | Clean, accessible Streamlit dashboard. | Enables non-technical stakeholder use. |

---

## 🛠️ Technology Stack

<div align="center">
  <img src="https://skillicons.dev/icons?i=python,pytorch,scikitlearn,pandas,numpy,git" /><br><br>
  <i><b>Machine Learning:</b> XGBoost, PyTorch, Scikit-learn, SHAP <br> <b>Data & App:</b> Pandas, NumPy, Streamlit</i>
</div>

---

## 📊 Evaluation & Impact

- 📈 **Model Performance**: Selected **XGBoost** over Logistic Regression and MLP for its balance of high predictive accuracy **(~85-90%)** and structural interpretability.
- ⚡ **Throughput**: Effortlessly scores and categorizes **thousands of customer records** in seconds.
- 💼 **Decision Automaton**: Saves hours of manual cohort analysis and reduces human error in assigning promotional offerings.

---

## 💻 Dashboard Preview

<div align="center">
  <img src="Screenshot/Screenshot%202026-03-31%20202422.png" width="48%" alt="App Upload Interface" style="border-radius: 8px;" />
  <img src="Screenshot/Screenshot%202026-03-31%20203546.png" width="48%" alt="Customer Data Preview" style="border-radius: 8px;" />
  <br><br>
  <img src="Screenshot/Screenshot%202026-03-31%20203704.png" width="48%" alt="High-Risk Alerts & Actions" style="border-radius: 8px;" />
  <img src="Screenshot/Screenshot%202026-03-31%20204012.png" width="48%" alt="SHAP Business Analytics" style="border-radius: 8px;" />
</div>

---

## ⚙️ How to Run

Launch the dashboard locally in under two minutes:

```bash
# 1. Clone the repository
git clone https://github.com/Piyu242005/Customer-Churn-Prediction-Action-System.git
cd Customer-Churn-Prediction-Action-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the application
streamlit run app/main.py
```

> **Note:** Open the resulting `localhost` address in your browser, upload the sample dataset located at [`Screenshot/Sample Data To Used/WA_Fn-UseC_-Telco-Customer-Churn.csv`](Screenshot/Sample%20Data%20To%20Used/WA_Fn-UseC_-Telco-Customer-Churn.csv), and click "Predict Churn Risk" to see the predictive engine in action!

---

## 📁 Repository Structure

```text
📦 Customer-Churn-Prediction-Action-System
 ┣ 📂 app
 ┃ ┗ 📜 main.py                # Streamlit UI & Frontend logic
 ┣ 📂 data                     # Raw and processed datasets (ignored)
 ┣ 📂 models                   # Serialized ML artifacts (.pkl)
 ┣ 📂 notebooks                # Jupyter notebooks for Exploratory Data Analysis
 ┣ 📂 src                      # Python scripts (Training, Evaluation, Pipelines)
 ┣ 📜 Makefile                 # Handy terminal aliases
 ┣ 📜 requirements.txt         # Project environment dependencies
 ┗ 📜 README.md                # Project documentation
```

---

## 🎯 Future Roadmap

- [ ] **REST API Module**: Decouple the model via a FastAPI route (`/predict`).
- [ ] **Cloud Deployment**: Host directly via Render, HuggingFace Spaces, or AWS EC2.
- [ ] **A/B Testing Integration**: Measure success rates of "Suggested Actions" on held-out user groups.

---

## 💬 Let's Connect

**Piyush Ramteke**
- 💼 **LinkedIn**: [Connect with me](www.linkedin.com/in/piyu24)
- 💻 **GitHub**: [Piyu242005](https://github.com/Piyu242005)
- 📧 **Email**: [piyu.143247@gmail.com](piyu.143247@gmail.com)

<br>

<div align="center">
  <b>⭐ If you find this repository useful, please consider giving it a star! ⭐</b>
</div>