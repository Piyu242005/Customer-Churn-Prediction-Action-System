from __future__ import annotations
import sys,io
from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from app.data_ingestion import REQUIRED_COLUMNS,detect_type,read_customer_file,suggest_mapping,validate_and_normalize
st.set_page_config(page_title="Upload Data",page_icon="📂",layout="wide")
st.title("📂 Upload Data")
st.subheader("Upload → Check → Ready")
st.write("Upload your customer data. We support CSV, Excel, and table-based PDF files.")
sample=pd.DataFrame({"Customer_ID":["CUST-001","CUST-002"],"Quantity":[3,1],"Unit_Price":[250,500],"Discount_Rate":[.10,.05],"Revenue":[675,475],"Cost":[450,350],"Profit":[225,125]})
a,b=st.columns(2)
a.download_button("⬇️ Sample CSV",sample.to_csv(index=False),"customer_churn_sample.csv","text/csv")
excel=io.BytesIO()
with pd.ExcelWriter(excel,engine="openpyxl") as w: sample.to_excel(w,index=False,sheet_name="Customers")
b.download_button("⬇️ Sample Excel",excel.getvalue(),"customer_churn_sample.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.caption("Required: Quantity • Unit_Price • Discount_Rate • Revenue • Cost • Profit")
uploaded=st.file_uploader("📤 Choose customer file",type=["csv","xlsx","xls","pdf"])
if uploaded:
    kind=detect_type(uploaded.name);st.success(f"Detected: {uploaded.name} • {kind}");sheet=0
    if kind=="Excel": sheet=st.selectbox("Choose worksheet",pd.ExcelFile(io.BytesIO(uploaded.getvalue())).sheet_names)
    try: df,_=read_customer_file(uploaded,sheet)
    except Exception as e: st.error(f"Could not read this file: {e}");st.stop()
    st.subheader("1. Preview");st.dataframe(df.head(20),use_container_width=True)
    st.subheader("2. Check column names")
    suggestions=suggest_mapping(df.columns);mapping={};cols=st.columns(2)
    for i,target in enumerate(REQUIRED_COLUMNS):
        opts=["— Not mapped —"]+list(df.columns);default=opts.index(suggestions[target]) if target in suggestions else 0
        with cols[i%2]: selected=st.selectbox(target,opts,index=default,key=f"upload_map_{target}")
        if selected!="— Not mapped —":mapping[target]=selected
    normalized,report=validate_and_normalize(df,mapping)
    if normalized is None: st.error("Please map all required fields before continuing.")
    else:
        if report.get("invalid"): st.warning(f"Some values need attention: {report['invalid']}")
        else: st.success("✅ Your data is ready for analysis.")
        st.session_state.customer_data=normalized
        st.session_state.results=None
        st.download_button("⬇️ Download checked data",normalized.to_csv(index=False),"checked_customer_data.csv","text/csv")
        st.info("Now open **👀 Explore Customers**, **📊 Churn Insights**, or another page from the navigation menu.")
else: st.info("👤 Don't have a file? Start the Guest Demo from **Home**.")
