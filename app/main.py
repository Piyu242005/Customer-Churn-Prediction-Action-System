import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Customer Churn Prediction & Action System", layout="wide")

st.title(" Customer Churn Prediction & Action System")
st.markdown("Upload a customer churn dataset to predict churn risk using **XGBoost**. The system features real-time inference, actionable insights mapped to retention strategies, and **SHAP** explainability to uncover why a customer might churn.")

# Preload Models
@st.cache_resource
def load_models():
    model_path = "models/best_churn_model.pkl"
    scaler_path = "models/scaler.pkl"
    feat_path = "models/feature_names.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feat_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(feat_path)
        return model, scaler, features
    return None, None, None

model, scaler, expected_features = load_models()

required_columns = [
    "tenure", "MonthlyCharges", "TotalCharges", 
    "Contract", "InternetService", "OnlineSecurity", "TechSupport"
]

# --- Sidebar ---
st.sidebar.header("Upload Data")
st.sidebar.markdown("Upload a CSV file containing at minimum the following features:")
for f in required_columns:
    st.sidebar.markdown(f"- `{f}`")

uploaded_file = st.sidebar.file_uploader("Upload customer churn dataset", type=["csv"])

# Define smart business logic for churn actions based on Telco data
def get_action(risk_level, top_driver):
    if risk_level == "High":
        if "Contract" in top_driver:
            return "Offer a discounted long-term contract to lock-in"
        elif "Charges" in top_driver:
            return "Review pricing plan and offer competitive retention rate"
        elif "InternetService" in top_driver or "TechSupport" in top_driver:
            return "Proactively reach out with technical support check-up"
        elif "OnlineSecurity" in top_driver:
            return "Offer 3 months free of premium online security add-on"
        elif "tenure" in top_driver:
            return "Provide targeted onboarding support and loyalty bonuses"
        else:
            return "Assign to dedicated customer success manager for immediate review"
    elif risk_level == "Medium":
        return "Send targeted promotional email and soft engagement"
    else:
        return "No immediate action required"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df.head())

    # Check for missing features instead of model's expected features
    missing_feats = [f for f in required_columns if f not in df.columns]

    if missing_feats:
        st.error(f"Missing required columns in CSV: {', '.join(missing_feats)}")
    elif not model:
        st.error("Model artifacts not found. Please run the model training script `src/train_dashboard.py` first.")
    else:
        if st.button("Predict Real Churn Risk", type="primary"):
            with st.spinner("Running XGBoost inference and SHAP explanations..."):
                
                # Preprocessing
                X_inference = df.copy()
                
                # Convert TotalCharges to numeric
                X_inference['TotalCharges'] = pd.to_numeric(X_inference['TotalCharges'], errors='coerce').fillna(0)
                
                # Encode categorical columns safely
                cat_cols = [c for c in ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport'] if c in X_inference.columns]
                X_inference = pd.get_dummies(X_inference, columns=cat_cols, drop_first=True)
                
                # Align columns with what the model expects perfectly
                if expected_features:
                    for col in expected_features:
                        if col not in X_inference.columns:
                            X_inference[col] = 0
                    X_inference = X_inference[expected_features]
                
                X_scaled = scaler.transform(X_inference)
                
                # Real ML Predictions
                probabilities = model.predict_proba(X_scaled)[:, 1] * 100
                
                # SHAP Explainability
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled)
                
                # Find top driver per customer (feature with highest positive SHAP value push)
                top_driver_indices = np.argmax(shap_values, axis=1)
                top_drivers = [expected_features[idx] if expected_features else str(idx) for idx in top_driver_indices]
                
                # Append Results
                results_df = df.copy()
                results_df["Churn_Prob_%"] = np.round(probabilities, 2)
                results_df["Risk_Level"] = pd.cut(results_df["Churn_Prob_%"], 
                                                bins=[-1, 30, 70, 101], 
                                                labels=["Low", "Medium", "High"])
                results_df["Top_Reason"] = top_drivers
                results_df["Recommended_Action"] = results_df.apply(lambda row: get_action(row["Risk_Level"], row["Top_Reason"]), axis=1)

                st.success("Inference Complete!")

                # Key Business Metrics
                high_risk_count = len(results_df[results_df["Risk_Level"] == "High"])
                med_risk_count = len(results_df[results_df["Risk_Level"] == "Medium"])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Customers Evaluated", len(results_df))
                col2.metric(" High Risk Customers (Urgent)", high_risk_count)
                col3.metric(" Medium Risk Customers", med_risk_count)

                st.divider()
                
                # Display Results
                st.subheader("Actionable Target List")
                
                # Filters
                risk_filter = st.selectbox("Filter Table by Risk Level:", ["All", "High", "Medium", "Low"])
                if risk_filter != "All":
                    display_df = results_df[results_df["Risk_Level"] == risk_filter]
                else:
                    display_df = results_df
                
                def highlight_risk(val):
                    color = "#ff4b4b" if val == "High" else "#ffa32f" if val == "Medium" else "#00cc96"
                    return f"color: {color}; font-weight: bold"

                disp_cols = [c for c in ["customerID", "gender", "Partner"] if c in df.columns] 
                if not disp_cols: disp_cols = list(df.columns[:2])
                disp_cols += ["Churn_Prob_%", "Risk_Level", "Top_Reason", "Recommended_Action"]
                
                st.dataframe(display_df[disp_cols].style.map(highlight_risk, subset=["Risk_Level"]), use_container_width=True)

                st.divider()

                # Business Dashboard + SHAP
                st.subheader("Analytical Dashboard & Explainability")
                dash_col1, dash_col2 = st.columns(2)
                
                with dash_col1:
                    st.markdown("#### Risk Distribution")
                    risk_counts = results_df["Risk_Level"].value_counts().reindex(["High", "Medium", "Low"])
                    st.bar_chart(risk_counts)

                with dash_col2:
                    st.markdown("#### Global Feature Importance (SHAP)")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    shap.summary_plot(shap_values, X_scaled, feature_names=expected_features, show=False, plot_type="bar")
                    plt.tight_layout()
                    st.pyplot(fig)
else:
    st.info("Awaiting CSV file to be uploaded.")
