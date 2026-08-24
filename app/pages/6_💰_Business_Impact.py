import streamlit as st
st.set_page_config(page_title="Business Impact",page_icon="💰",layout="wide")
st.title("💰 Business Impact")
st.write("Estimate the potential value of keeping at-risk customers.")
df=st.session_state.get("customer_data")
if df is None: st.info("Load customer data from **📂 Upload Data** first."); st.stop()
try:
 from pathlib import Path
 import sys,joblib,pandas as pd
 ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));model=joblib.load(ROOT/"models/best_churn_model.pkl");scaler=joblib.load(ROOT/"models/scaler.pkl");features=list(joblib.load(ROOT/"models/feature_names.pkl"));from src.retention import roi_estimate
 x=df[features].copy()
 for c in features:x[c]=pd.to_numeric(x[c],errors="coerce")
 x=x.fillna(x.median(numeric_only=True)).fillna(0);risk=model.predict_proba(scaler.transform(x))[:,1]*100
 n=int((risk>=70).sum());revenue=float(df["Revenue"].sum()*risk.mean()/100) if "Revenue" in df else 10000.0
 a,b=st.columns(2);n=st.number_input("Customers targeted",0,max(len(df),1),max(n,1));cost=st.number_input("Cost per customer ($)",0.0,25.0,5.0);save=st.slider("Expected save rate (%)",0,100,30)
 roi=roi_estimate(n,revenue,cost,save);c1,c2,c3,c4=st.columns(4);c1.metric("Value saved",f"${roi['expected_saved']:,.0f}");c2.metric("Cost",f"${roi['intervention_cost']:,.0f}");c3.metric("Net value",f"${roi['net_value']:,.0f}");c4.metric("ROI",f"{roi['roi_percent']:.1f}%")
except Exception as e:st.error(f"Could not calculate business impact: {e}")
