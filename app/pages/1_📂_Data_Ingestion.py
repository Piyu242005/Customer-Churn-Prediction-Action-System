from __future__ import annotations

import sys
from pathlib import Path
import io
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.data_ingestion import REQUIRED_COLUMNS, detect_type, read_customer_file, suggest_mapping, validate_and_normalize

st.set_page_config(page_title="Data Ingestion | Customer Churn", page_icon="📂", layout="wide")
st.markdown("""
<style>
.stApp {background:#080808;color:#f5f5f5}.stSidebar{background:#050505}
h1,h2,h3{color:#fff}.stButton>button,.stDownloadButton>button{background:#e50914;color:#fff;border:0}
[data-testid="stMetric"]{background:#111;border-left:3px solid #e50914;border-radius:10px}
[data-testid="stFileUploader"]{background:#111;border:1px dashed #e50914;border-radius:10px}
</style>
""", unsafe_allow_html=True)

st.title("📂 Customer Data Ingestion")
st.subheader("Upload → Detect → Preview → Map → Validate → Analyze")
st.write("Prepare customer data once. The validated dataset can then be used by the existing Customer 360, Analytics, SHAP, Batch, What-if, ROI, and Monitoring pages.")

sample = pd.DataFrame({
    "Customer_ID": ["CUST-001", "CUST-002"],
    "Quantity": [3, 1], "Unit_Price": [250, 500], "Discount_Rate": [0.10, 0.05],
    "Revenue": [675, 475], "Cost": [450, 350], "Profit": [225, 125]
})

st.download_button("⬇️ Download Sample CSV", sample.to_csv(index=False), "customer_churn_sample.csv", "text/csv")
excel = io.BytesIO()
with pd.ExcelWriter(excel, engine="openpyxl") as writer:
    sample.to_excel(writer, index=False, sheet_name="Customers")
st.download_button("⬇️ Download Sample Excel", excel.getvalue(), "customer_churn_sample.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("### 📤 Upload Customer Data")
st.caption("Supported formats: CSV • XLSX • XLS • PDF")
uploaded = st.file_uploader("Choose customer data file", type=["csv", "xlsx", "xls", "pdf"], label_visibility="collapsed")

if uploaded:
    kind = detect_type(uploaded.name)
    st.success(f"Detected file: **{uploaded.name}**  •  Type: **{kind}**")

    sheet = 0
    if kind == "Excel":
        book = pd.ExcelFile(io.BytesIO(uploaded.getvalue()))
        sheet = st.selectbox("Select worksheet", book.sheet_names)

    try:
        df, _ = read_customer_file(uploaded, sheet)
    except Exception as exc:
        st.error(f"❌ Could not read this file: {exc}")
        st.stop()

    st.markdown("### 1. Data Preview")
    a, b = st.columns(2)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{len(df.columns):,}")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### 2. Column Mapping")
    suggestions = suggest_mapping(df.columns)
    mapping = {}
    cols = st.columns(2)
    for i, target in enumerate(REQUIRED_COLUMNS):
        options = ["— Not mapped —"] + list(df.columns)
        default = options.index(suggestions[target]) if target in suggestions else 0
        with cols[i % 2]:
            selected = st.selectbox(target, options, index=default, key=f"map_{target}")
        if selected != "— Not mapped —":
            mapping[target] = selected

    st.markdown("### 3. Validation")
    normalized, report = validate_and_normalize(df, mapping)
    if normalized is None:
        st.error("❌ Dataset is not ready. Map all required model features above.")
        st.write("Required columns:", ", ".join(REQUIRED_COLUMNS))
        st.write("Detected columns:", ", ".join(map(str, df.columns)))
        st.stop()

    invalid = report.get("invalid", {})
    if invalid:
        st.warning("⚠️ Some values could not be converted to numeric values. Review these columns before analysis.")
        st.json(invalid)
    else:
        st.success("✅ Dataset validated successfully.")

    st.markdown("### 4. Normalized Dataset")
    st.dataframe(normalized.head(20), use_container_width=True)
    st.caption("Required model features: " + " • ".join(REQUIRED_COLUMNS))

    st.markdown("### 🚀 Ready for Analysis")
    st.success("Your dataset is normalized and ready. Use the existing analysis pages to continue.")
    st.download_button("⬇️ Download Normalized Dataset", normalized.to_csv(index=False), "normalized_customer_data.csv", "text/csv")
else:
    st.info("👤 Don't have a file? Use the Guest Demo from the main application.")

st.divider()
st.caption("CSV + Excel + PDF ingestion • validation • column mapping • preview • normalized export")
