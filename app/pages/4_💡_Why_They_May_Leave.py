import streamlit as st
st.set_page_config(page_title="Why They May Leave",page_icon="💡",layout="wide")
st.title("💡 Why They May Leave")
st.write("Understand the main factors behind a customer's churn prediction.")
df=st.session_state.get("customer_data")
if df is None: st.info("Load customer data from **📂 Upload Data** first."); st.stop()
try:
 from pathlib import Path
 import sys,joblib,pandas as pd,numpy as np,shap
 ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));model=joblib.load(ROOT/"models/best_churn_model.pkl");scaler=joblib.load(ROOT/"models/scaler.pkl");features=list(joblib.load(ROOT/"models/feature_names.pkl"))
 x=df[features].copy()
 for c in features:x[c]=pd.to_numeric(x[c],errors="coerce")
 x=x.fillna(x.median(numeric_only=True)).fillna(0);xs=scaler.transform(x);sv=shap.TreeExplainer(model).shap_values(xs)
 if isinstance(sv,list):sv=sv[-1]
 sv=np.asarray(sv);sv=sv[:,:,-1] if sv.ndim==3 else sv
 imp=pd.DataFrame({"Factor":features,"Importance":np.abs(sv).mean(axis=0)}).sort_values("Importance",ascending=False)
 st.subheader("Most important factors");st.bar_chart(imp.set_index("Factor"))
 idx=st.selectbox("Choose a customer",df.index,format_func=lambda i:str(df.loc[i].get("Customer_ID",i)))
 local=pd.DataFrame({"Factor":features,"Impact":sv[idx],"Value":x.loc[idx].values}).sort_values("Impact",key=abs,ascending=False)
 st.dataframe(local,use_container_width=True);st.caption("Higher positive impact means the factor is pushing the prediction toward churn.")
except Exception as e:st.error(f"Could not create explanations: {e}")
