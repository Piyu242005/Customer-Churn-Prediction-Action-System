#  Customer Churn Prediction & Action System

##  Problem Statement
In a highly competitive business landscape, acquiring a new customer is up to 5 times more expensive than retaining an existing one. High churn rates directly impact annual recurring revenue (ARR). Businesses need a proactive way to not only predict *who* will churn, but understand *why*, and immediately know *what to do* about it.

##  Solution Overview
This project transforms a standard machine learning classifier into an end-to-end **Customer Retention Action System**. It predicts the probability of customer churn, leverages explainable AI (SHAP) to identify the root causes per customer, and prescribes targeted, actionable retention strategies through an interactive web dashboard.

##  Architecture
1. **Data Pipeline:** Preprocesses raw transactional and customer data using Scikit-Learn.
2. **AI Prediction Engine:** Compares Logistic Regression, Random Forest, and XGBoost, utilizing the optimal model for inference.
3. **Explainability Layer:** Uses SHAP (SHapley Additive exPlanations) to extract global and local feature importance.
4. **Business Logic Layer:** Translates raw predictive probabilities (0-100%) into Risk Categories (Low/Medium/High) and generates strict business interventions (e.g., promotional emails, discounts).
5. **UI / Dashboard:** Deployed via Streamlit to allow non-technical stakeholders to upload datasets and access real-time insights.

##  Tech Stack
- **Modeling:** Python, Scikit-Learn, XGBoost, PyTorch (MLP baseline)
- **Data Processing:** Pandas, NumPy
- **Explainability:** SHAP
- **Deployment:** Streamlit

##  Results & Business Impact
- **Model Performance:** Selected **XGBoost**, achieving ~85% Accuracy and 0.70 ROC-AUC, outperforming the baseline Logistic Regression.
- **Scale:** Engineered to process thousands of customer records instantly.
- **Business Value:** Enables automated, data-driven retention strategies, replacing manual cohort analysis and theoretically reducing revenue churn by enabling targeted interventions before customers leave.

##  How to Run

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/yourusername/Customer-Churn-Prediction-Action-System.git
   cd Customer-Churn-Prediction-Action-System
   pip install -r requirements.txt
   ```

2. **Train the ML Models (Optional, artifacts are in `/models`):**
   ```bash
   python src/train_dashboard.py
   ```

3. **Launch the Streamlit Dashboard:**
   ```bash
   streamlit run app/main.py
   ```

4. **Upload Data:** Upload `data/Business_Analytics_Dataset_10000_Rows.csv` in the Streamlit sidebar to see the magic happen!

##  Screenshots
*(Add a screenshot of your Streamlit web app here)*
`![Dashboard Preview](docs/dashboard_preview.png)`

