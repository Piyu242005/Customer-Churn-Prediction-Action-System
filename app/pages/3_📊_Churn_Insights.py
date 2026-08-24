import streamlit as st
import numpy as np
st.set_page_config(page_title="Churn Insights",page_icon="📊",layout="wide")
st.title("📊 Churn Insights")
st.write("See which customers may leave and how much churn risk exists.")
df=st.session_state.get("customer_data")
if df is None: st.info("Load customer data from **📂 Upload Data** first."); st.stop()
# Use the project's existing prediction artifacts.
try:
    from pathlib import Path
    import sys,joblib,pandas as pd
    ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
    from app.data_ingestion import REQUIRED_COLUMNS
    model=joblib.load(ROOT/"models/best_churn_model.pkl");scaler=joblib.load(ROOT/"models/scaler.pkl");features=joblib.load(ROOT/"models/feature_names.pkl")
    x=df[list(features)].copy()
    for c in features:x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.fillna(x.median(numeric_only=True)).fillna(0)
    risk=model.predict_proba(scaler.transform(x))[:,1]*100
    out=df.copy();out["Churn Risk %"]=risk.round(2);out["Risk Level"]=np.select([risk>=70,risk>=40],["High","Medium"],default="Low")
    a,b,c=st.columns(3);a.metric("Customers",len(out));b.metric("High Risk",int((risk>=70).sum()));c.metric("Average Risk",f"{risk.mean():.1f}%")
    l,r=st.columns(2)
    with l:st.subheader("Risk levels");st.bar_chart(out["Risk Level"].value_counts().reindex(["High","Medium","Low"]).fillna(0))
    with r:st.subheader("Risk distribution");st.bar_chart(out["Churn Risk %"].round().value_counts().sort_index())
    st.dataframe(out.sort_values("Churn Risk %",ascending=False).head(50),use_container_width=True)
except Exception as e: st.error(f"Could not calculate churn insights: {e}")
