import streamlit as st
import pandas as pd
import numpy as np
import joblib
# import shap  # Uncomment for SHAP

st.set_page_config(page_title="Customer Churn Prediction & Action System", layout="wide")

st.title("🚀 Customer Churn Prediction & Action System")
st.markdown("Upload customer data to predict churn risk, identify key drivers, and generate actionable retention strategies.")

# --- Sidebar ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Define business logic for churn actions
def get_action(risk_level, top_driver):
    if risk_level == "High":
        if "Charges" in top_driver or "Price" in top_driver:
            return "Offer a 15% discount or retention plan"
        else:
            return "Reach out with a personalized wellness check call"
    elif risk_level == "Medium":
        return "Send targeted promotional email"
    else:
        return "No immediate action required"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df.head())

    if st.button("Predict Churn Risk"):
        with st.spinner("Analyzing customer risk..."):
            # MOCK PREDICTION FOR EXAMPLE (Replace with model: joblib.load('models/xgb_model.pkl'))
            # Generate mock probability since no trained model provides this perfectly yet
            np.random.seed(42)
            probabilities = np.random.uniform(0, 100, size=len(df))
            
            results_df = df.copy()
            results_df['Churn_Prob_%'] = np.round(probabilities, 2)
            results_df['Risk_Level'] = pd.cut(results_df['Churn_Prob_%'], 
                                              bins=[-1, 30, 70, 101], 
                                              labels=["Low", "Medium", "High"])
            results_df['Top_Reason'] = np.random.choice(['High Charges', 'Low Usage', 'Customer Support', 'Competitor Offer'], size=len(df))
            results_df['Recommended_Action'] = results_df.apply(lambda row: get_action(row['Risk_Level'], row['Top_Reason']), axis=1)

            st.success("Analysis Complete!")

            # Quick Metrics
            high_risk_count = len(results_df[results_df['Risk_Level'] == 'High'])
            st.metric("Total Customers Evaluated", len(results_df))
            st.metric("🚨 High Risk Customers", high_risk_count)

            # Display Results
            st.subheader("Actionable Insights")
            
            # Formatting table to highlight high risk
            def highlight_risk(val):
                color = '#ff4b4b' if val == 'High' else '#ffa32f' if val == 'Medium' else '#00cc96'
                return f'color: {color}; font-weight: bold'
            
            disp_cols = [c for c in df.columns[:2]] + ['Churn_Prob_%', 'Risk_Level', 'Top_Reason', 'Recommended_Action']
            st.dataframe(results_df[disp_cols].style.applymap(highlight_risk, subset=['Risk_Level']), use_container_width=True)

            # Plots
            st.subheader("Business Dashboard")
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(results_df['Risk_Level'].value_counts())
            with col2:
                # Mock SHAP plot placeholder
                st.info("SHAP Feature Importance Plot (e.g., Tenure, Monthly Charges) goes here.")

else:
    st.info("Awaiting CSV file to be uploaded.")
