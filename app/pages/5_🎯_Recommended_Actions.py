import streamlit as st
st.set_page_config(page_title="Recommended Actions",page_icon="🎯",layout="wide")
st.title("🎯 Recommended Actions")
st.write("Get simple next steps for customers who may leave.")
df=st.session_state.get("customer_data")
if df is None: st.info("Load customer data from **📂 Upload Data** first."); st.stop()
try:
 from pathlib import Path
 import sys,joblib,pandas as pd,numpy as np
 ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));model=joblib.load(ROOT/"models/best_churn_model.pkl");scaler=joblib.load(ROOT/"models/scaler.pkl");features=list(joblib.load(ROOT/"models/feature_names.pkl"));from src.retention import retention_action,risk_band
 x=df[features].copy()
 for c in features:x[c]=pd.to_numeric(x[c],errors="coerce")
 x=x.fillna(x.median(numeric_only=True)).fillna(0);risk=model.predict_proba(scaler.transform(x))[:,1]*100
 out=df.copy();out["Churn Risk %"]=risk.round(2);out["Risk Level"]=[risk_band(v) for v in risk];out["Recommended Action"]=[retention_action("Customer risk factors",r) for r in out["Risk Level"]]
 cols=[c for c in ["Customer_ID","Order_ID","Churn Risk %","Risk Level","Recommended Action"] if c in out];st.dataframe(out[cols].sort_values("Churn Risk %",ascending=False),use_container_width=True);st.download_button("⬇️ Download action list",out[cols].to_csv(index=False),"retention_actions.csv","text/csv")
except Exception as e:st.error(f"Could not create actions: {e}")
