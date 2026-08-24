from __future__ import annotations

import sys
import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retention import retention_action, risk_band, roi_estimate


# ============================================================
# PAGE CONFIG + LOGO
# ============================================================

LOGO_PATH = ROOT / "favicon.png"

st.set_page_config(
    page_title="Customer Churn Action System",
    page_icon=str(LOGO_PATH),
    layout="wide",
)


# ============================================================
# RED + BLACK PREMIUM THEME
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background: #080808;
        color: #f5f5f5;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #050505;
        border-right: 1px solid #2b2b2b;
    }

    /* Text */
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }

    p, label {
        color: #dddddd;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #111111;
        border: 1px solid #2b2b2b;
        border-left: 3px solid #e50914;
        padding: 14px;
        border-radius: 10px;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    [data-testid="stMetricLabel"] {
        color: #aaaaaa !important;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background: #e50914 !important;
        color: white !important;
        border: 1px solid #e50914 !important;
        border-radius: 8px !important;
        font-weight: 700;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #b20710 !important;
        border-color: #ff1a25 !important;
    }

    /* Navigation */
    [data-baseweb="tab-list"] {
        background: #111111;
        padding: 5px;
        border-radius: 10px;
    }

    [data-baseweb="tab"] {
        color: #aaaaaa !important;
    }

    [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom-color: #e50914 !important;
    }

    /* Cards */
    [data-testid="stExpander"] {
        background: #111111;
        border: 1px solid #2b2b2b;
        border-radius: 10px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #111111;
        border: 1px dashed #444444;
        border-radius: 10px;
        padding: 10px;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #2b2b2b;
        border-radius: 8px;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        background: #111111;
        border-color: #e50914;
    }

    /* Header logo */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 22px;
        padding: 8px 0;
    }

    .brand-logo {
        width: 64px;
        height: 64px;
        object-fit: contain;
        border-radius: 12px;
    }

    .brand-title {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 14px;
        color: #aaaaaa;
        margin-top: 5px;
    }

    .section-card {
        background: #111111;
        border: 1px solid #2b2b2b;
        border-radius: 12px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BRAND HEADER
# ============================================================

if LOGO_PATH.exists():

    logo_base64 = base64.b64encode(
        LOGO_PATH.read_bytes()
    ).decode("utf-8")

    st.markdown(
        f"""
        <div class="brand-header">

            <img
                class="brand-logo"
                src="data:image/png;base64,{logo_base64}"
                alt="Customer Churn Logo"
            >

            <div>

                <div class="brand-title">
                    Customer Churn Prediction & Action System
                </div>

                <div class="brand-subtitle">
                    Predict Risk → Explain Why → Recommend Action → Measure Business Impact
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.title("Customer Churn Prediction & Action System")


# ============================================================
# MODEL FILES
# ============================================================

MODEL = ROOT / "models/best_churn_model.pkl"
SCALER = ROOT / "models/scaler.pkl"
FEATURE_FILE = ROOT / "models/feature_names.pkl"
REFERENCE = ROOT / "data/Business_Analytics_Dataset_10000_Rows.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_artifacts():

    if not all(
        path.exists()
        for path in [MODEL, SCALER, FEATURE_FILE]
    ):
        return None, None, None

    return (
        joblib.load(MODEL),
        joblib.load(SCALER),
        joblib.load(FEATURE_FILE),
    )


@st.cache_data
def load_reference():

    if REFERENCE.exists():
        return pd.read_csv(REFERENCE)

    return None


# ============================================================
# GUEST DEMO DATA
# ============================================================

@st.cache_data
def make_guest_data(n=250):

    rng = np.random.default_rng(2026)

    data = pd.DataFrame(
        {
            "Order_ID": np.arange(1, n + 1),

            "Customer_ID": [
                f"GUEST-{i:04d}"
                for i in range(1, n + 1)
            ],

            "Quantity": rng.integers(
                1,
                11,
                n,
            ),

            "Unit_Price": np.round(
                rng.uniform(20, 500, n),
                2,
            ),

            "Discount_Rate": np.round(
                rng.uniform(0, 0.30, n),
                2,
            ),

            "Region": rng.choice(
                [
                    "North",
                    "South",
                    "East",
                    "West",
                ],
                n,
            ),

            "Product_Category": rng.choice(
                [
                    "Electronics",
                    "Clothing",
                    "Sports",
                    "Beauty",
                    "Home & Kitchen",
                ],
                n,
            ),
        }
    )

    data["Revenue"] = np.round(
        data["Quantity"]
        * data["Unit_Price"]
        * (1 - data["Discount_Rate"]),
        2,
    )

    data["Cost"] = np.round(
        data["Revenue"]
        * rng.uniform(0.45, 0.82, n),
        2,
    )

    data["Profit"] = np.round(
        data["Revenue"] - data["Cost"],
        2,
    )

    return data


# ============================================================
# LOAD ARTIFACTS
# ============================================================

model, scaler, expected_features = load_artifacts()

if model is None:

    st.error(
        "Model files are missing. "
        "Please make sure the model artifacts exist."
    )

    st.stop()


FEATURES = list(expected_features)

reference = load_reference()


# ============================================================
# SESSION STATE
# ============================================================

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "results" not in st.session_state:
    st.session_state.results = None


# ============================================================
# SIMPLE NAVIGATION
# ============================================================

page = st.radio(
    "",
    [
        "🏠 Home",
        "📂 Upload Data",
        "👀 Explore Customers",
        "📊 Churn Insights",
        "💡 Why They May Leave",
        "🎯 Recommended Actions",
        "💰 Business Impact",
    ],
    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.caption(
        "A simple customer-retention tool. "
        "You don't need technical knowledge to use it."
    )

    st.markdown("### Start in seconds")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
            **👤 Try Guest Demo**

            Explore the complete system using
            safe sample customer data.

            No upload required.
            """
        )

        if st.button(
            "🚀 Start Guest Demo",
            use_container_width=True,
        ):

            st.session_state.dataset = make_guest_data()

            st.session_state.results = None

            st.success(
                "Guest Demo loaded successfully."
            )

            st.rerun()

    with col2:

        st.info(
            """
            **📂 Have customer data?**

            Upload your CSV, Excel, or
            table-based PDF.
            """
        )

        st.markdown(
            "Go to **📂 Upload Data** to begin."
        )

    st.markdown("### What you can do")

    cards = st.columns(4)

    cards[0].metric(
        "📊 Find Risky Customers",
        "See who may leave.",
    )

    cards[1].metric(
        "💡 Understand Why",
        "See the main reasons.",
    )

    cards[2].metric(
        "🎯 Take Action",
        "Get retention suggestions.",
    )

    cards[3].metric(
        "💰 Measure Impact",
        "Estimate potential value.",
    )

    st.info(
        "👤 Guest Demo uses synthetic data only."
    )

    st.stop()


# ============================================================
# UPLOAD DATA
# ============================================================

if page == "📂 Upload Data":

    st.header("📂 Upload Your Customer Data")

    st.write(
        "Upload a **CSV, Excel, or table-based PDF**. "
        "We'll check the file before analysis."
    )

    with st.expander(
        "ℹ️ What should my file contain?",
        expanded=True,
    ):

        st.markdown(
            """
            **Required columns:**

            - `Quantity`
            - `Unit_Price`
            - `Discount_Rate`
            - `Revenue`
            - `Cost`
            - `Profit`

            Optional:

            - `Customer_ID`
            - `Order_ID`
            """
        )

    uploaded = st.file_uploader(
        "📂 Choose a customer file",
        type=[
            "csv",
            "xlsx",
            "xls",
            "pdf",
        ],
    )

    if uploaded:

        try:

            suffix = Path(
                uploaded.name
            ).suffix.lower()

            # ---------------------------
            # CSV
            # ---------------------------

            if suffix == ".csv":

                df = pd.read_csv(uploaded)

            # ---------------------------
            # EXCEL
            # ---------------------------

            elif suffix in [".xlsx", ".xls"]:

                excel_file = pd.ExcelFile(
                    uploaded
                )

                sheets = excel_file.sheet_names

                if len(sheets) > 1:

                    selected_sheet = st.selectbox(
                        "📑 Choose worksheet",
                        sheets,
                    )

                else:

                    selected_sheet = sheets[0]

                df = pd.read_excel(
                    uploaded,
                    sheet_name=selected_sheet,
                )

            # ---------------------------
            # PDF
            # ---------------------------

            elif suffix == ".pdf":

                try:

                    import pdfplumber

                except ImportError:

                    st.error(
                        "PDF support requires "
                        "`pdfplumber` in requirements.txt."
                    )

                    st.stop()

                tables = []

                with pdfplumber.open(
                    uploaded
                ) as pdf:

                    for page_obj in pdf.pages:

                        extracted = (
                            page_obj.extract_tables()
                        )

                        if extracted:

                            tables.extend(
                                extracted
                            )

                if not tables:

                    raise ValueError(
                        "No customer table was found "
                        "inside this PDF."
                    )

                first_table = tables[0]

                if len(first_table) < 2:

                    raise ValueError(
                        "The PDF table does not contain "
                        "enough rows."
                    )

                df = pd.DataFrame(
                    first_table[1:],
                    columns=first_table[0],
                )

            else:

                raise ValueError(
                    "Unsupported file format."
                )

            # ---------------------------
            # VALIDATION
            # ---------------------------

            if df.empty:

                raise ValueError(
                    "The uploaded file is empty."
                )

            missing = [
                feature
                for feature in FEATURES
                if feature not in df.columns
            ]

            if missing:

                st.error(
                    "❌ Missing required columns: "
                    + ", ".join(missing)
                )

                st.info(
                    "Please check the required "
                    "columns above."
                )

            else:

                st.success(
                    f"✅ File ready: "
                    f"{len(df):,} rows × "
                    f"{len(df.columns):,} columns"
                )

                st.subheader(
                    "👀 Preview"
                )

                st.dataframe(
                    df.head(15),
                    use_container_width=True,
                )

                if st.button(
                    "🚀 Use This Data",
                    type="primary",
                ):

                    st.session_state.dataset = df

                    st.session_state.results = None

                    st.success(
                        "Your data is ready for analysis."
                    )

        except Exception as error:

            st.error(
                f"❌ We couldn't read this file: "
                f"{error}"
            )

    # ---------------------------
    # SAMPLE CSV
    # ---------------------------

    sample_data = (
        reference
        if reference is not None
        else make_guest_data()
    )

    st.download_button(
        "⬇️ Download Sample CSV",
        sample_data.to_csv(index=False),
        "customer_churn_sample.csv",
        "text/csv",
    )

    st.stop()


# ============================================================
# REQUIRE DATA
# ============================================================

if st.session_state.dataset is None:

    st.warning(
        "No customer data is loaded yet."
    )

    st.info(
        "Go to **📂 Upload Data** "
        "or start the **Guest Demo** from Home."
    )

    st.stop()


df = st.session_state.dataset


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare(data):

    x = data[FEATURES].copy()

    for column in FEATURES:

        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )

    x = x.fillna(
        x.median(
            numeric_only=True
        )
    )

    x = x.fillna(0)

    return x


# ============================================================
# SHAP
# ============================================================

def explain_values(x_scaled):

    values = (
        shap.TreeExplainer(
            model
        ).shap_values(x_scaled)
    )

    if isinstance(values, list):

        values = values[-1]

    values = np.asarray(values)

    if values.ndim == 3:

        values = values[:, :, -1]

    return values


# ============================================================
# PREDICTION
# ============================================================

def predict(data):

    x = prepare(data)

    x_scaled = scaler.transform(x)

    probability = (
        model.predict_proba(
            x_scaled
        )[:, 1]
        * 100
    )

    shap_values = explain_values(
        x_scaled
    )

    drivers = [
        FEATURES[index]
        for index in np.argmax(
            shap_values,
            axis=1,
        )
    ]

    output = data.copy()

    output["Churn_Prob_%"] = (
        probability.round(2)
    )

    output["Risk_Level"] = [
        risk_band(value)
        for value in probability
    ]

    output["Top_Reason"] = drivers

    output["Recommended_Action"] = [
        retention_action(
            driver,
            risk,
        )
        for driver, risk in zip(
            drivers,
            output["Risk_Level"],
        )
    ]

    return output, shap_values


# ============================================================
# RUN PREDICTION
# ============================================================

if st.session_state.results is None:

    st.session_state.results = predict(
        df
    )


results, shap_values = (
    st.session_state.results
)


high_risk_count = int(
    (
        results["Risk_Level"]
        == "High"
    ).sum()
)

average_risk = float(
    results["Churn_Prob_%"].mean()
)

revenue_risk = (
    float(
        results["Revenue"].sum()
        * average_risk
        / 100
    )
    if "Revenue" in results
    else 0
)


# ============================================================
# EXPLORE CUSTOMERS
# ============================================================

if page == "👀 Explore Customers":

    st.header(
        "👀 Explore Customers"
    )

    st.write(
        "Browse and search your customer records."
    )

    search = st.text_input(
        "🔍 Search customer",
        placeholder=(
            "Customer ID, Order ID, "
            "or any value"
        ),
    )

    view = results.copy()

    if search:

        view = view[
            view.astype(str)
            .apply(
                lambda row:
                row.str.contains(
                    search,
                    case=False,
                    na=False,
                ).any(),
                axis=1,
            )
        ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Customers",
        f"{len(results):,}",
    )

    c2.metric(
        "High Risk",
        f"{high_risk_count:,}",
    )

    c3.metric(
        "Average Risk",
        f"{average_risk:.1f}%",
    )

    st.dataframe(
        view,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download Customer Data",
        view.to_csv(index=False),
        "customers.csv",
        "text/csv",
    )


# ============================================================
# CHURN INSIGHTS
# ============================================================

elif page == "📊 Churn Insights":

    st.header(
        "📊 Churn Insights"
    )

    st.write(
        "See which customers are most likely to leave."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers",
        f"{len(results):,}",
    )

    c2.metric(
        "High Risk",
        f"{high_risk_count:,}",
    )

    c3.metric(
        "Average Risk",
        f"{average_risk:.1f}%",
    )

    c4.metric(
        "Revenue at Risk",
        (
            f"${revenue_risk:,.0f}"
            if revenue_risk
            else "N/A"
        ),
    )

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Risk Levels"
        )

        risk_counts = (
            results["Risk_Level"]
            .value_counts()
            .reindex(
                [
                    "High",
                    "Medium",
                    "Low",
                ]
            )
            .fillna(0)
        )

        st.bar_chart(
            risk_counts
        )

    with right:

        st.subheader(
            "Churn Probability"
        )

        st.bar_chart(
            results[
                "Churn_Prob_%"
            ]
            .round()
            .value_counts()
            .sort_index()
        )

    st.subheader(
        "Customers Most at Risk"
    )

    st.dataframe(
        results.sort_values(
            "Churn_Prob_%",
            ascending=False,
        ).head(50),
        use_container_width=True,
    )


# ============================================================
# WHY THEY MAY LEAVE
# ============================================================

elif page == "💡 Why They May Leave":

    st.header(
        "💡 Why Customers May Leave"
    )

    st.write(
        "See the factors influencing the churn prediction."
    )

    importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": np.abs(
                shap_values
            ).mean(axis=0),
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    st.subheader(
        "What matters most?"
    )

    st.bar_chart(
        importance.set_index(
            "Feature"
        )
    )

    customer_index = st.selectbox(
        "Choose a customer",
        results.index,
        format_func=lambda index:
        str(
            results.loc[index].get(
                "Customer_ID",
                index,
            )
        ),
    )

    local = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Impact": shap_values[
                customer_index
            ],
            "Value": prepare(
                results
            ).loc[
                customer_index
            ].values,
        }
    )

    local = local.sort_values(
        "Impact",
        key=abs,
        ascending=False,
    )

    st.subheader(
        "Customer-specific reasons"
    )

    st.dataframe(
        local,
        use_container_width=True,
    )

    st.caption(
        "Positive impact increases predicted churn. "
        "Negative impact reduces predicted churn."
    )


# ============================================================
# RECOMMENDED ACTIONS
# ============================================================

elif page == "🎯 Recommended Actions":

    st.header(
        "🎯 Recommended Actions"
    )

    st.write(
        "See what action is recommended for each customer."
    )

    columns = [
        column
        for column in [
            "Customer_ID",
            "Order_ID",
            "Churn_Prob_%",
            "Risk_Level",
            "Top_Reason",
            "Recommended_Action",
        ]
        if column in results
    ]

    actions = (
        results[columns]
        .sort_values(
            "Churn_Prob_%",
            ascending=False,
        )
    )

    st.dataframe(
        actions,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download Action List",
        actions.to_csv(index=False),
        "retention_actions.csv",
        "text/csv",
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

elif page == "💰 Business Impact":

    st.header(
        "💰 Business Impact"
    )

    st.write(
        "Estimate the potential value of retention actions."
    )

    customers_targeted = st.number_input(
        "Customers targeted",
        min_value=0,
        max_value=max(
            len(results),
            1,
        ),
        value=max(
            high_risk_count,
            1,
        ),
        step=1,
    )

    revenue_at_risk_input = st.number_input(
        "Revenue at risk ($)",
        min_value=0.0,
        value=(
            revenue_risk
            if revenue_risk
            else 10000.0
        ),
        step=100.0,
    )

    intervention_cost = st.number_input(
        "Cost per customer ($)",
        min_value=0.0,
        value=25.0,
        step=5.0,
    )

    save_rate = st.slider(
        "Expected save rate (%)",
        min_value=0,
        max_value=100,
        value=30,
    )

    roi = roi_estimate(
        customers_targeted,
        revenue_at_risk_input,
        intervention_cost,
        save_rate,
    )

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Expected Value Saved",
        f"${roi['expected_saved']:,.0f}",
    )

    r2.metric(
        "Intervention Cost",
        f"${roi['intervention_cost']:,.0f}",
    )

    r3.metric(
        "Net Value",
        f"${roi['net_value']:,.0f}",
    )

    r4.metric(
        "ROI",
        f"{roi['roi_percent']:.1f}%",
    )

    st.subheader(
        "System Status"
    )

    st.success(
        "✓ Customer data loaded\n\n"
        "✓ Model ready\n\n"
        "✓ Churn predictions available"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction & Action System "
    "• Streamlit • XGBoost • SHAP • Guest Demo "
    "• CSV • Excel • PDF"
)
