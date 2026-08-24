from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retention import retention_action, risk_band, roi_estimate

st.set_page_config(
    page_title="Customer Churn Action System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📉 Customer Churn Prediction & Action System")
st.caption("Predict risk → explain why → recommend action → estimate business impact")

MODEL_PATH = ROOT / "models/best_churn_model.pkl"
SCALER_PATH = ROOT / "models/scaler.pkl"
FEATURE_PATH = ROOT / "models/feature_names.pkl"
REFERENCE_PATH = ROOT / "data/Business_Analytics_Dataset_10000_Rows.csv"


@st.cache_resource
def load_artifacts():
    if not all(p.exists() for p in [MODEL_PATH, SCALER_PATH, FEATURE_PATH]):
        return None, None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(FEATURE_PATH)


@st.cache_data
 def load_reference():
    if not REFERENCE_PATH.exists():
        return None
    return pd.read_csv(REFERENCE_PATH)


model, scaler, expected_features = load_artifacts()
reference_df = load_reference()

if model is None:
    st.error("Model artifacts are missing. Run `python src/train_dashboard.py` first.")
    st.stop()

st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload customer CSV", type="csv")
use_demo = st.sidebar.checkbox("Use repository demo data", value=uploaded is None)

if uploaded is not None:
    source_df = pd.read_csv(uploaded)
elif use_demo and reference_df is not None:
    source_df = reference_df.copy()
else:
    st.info("Upload a CSV or enable demo data from the sidebar.")
    st.stop()


# The dashboard model is trained on these six numerical business features.
NUMERIC_FEATURES = list(expected_features)
missing = [c for c in NUMERIC_FEATURES if c not in source_df.columns]
if missing:
    st.error(f"Missing model features: {', '.join(missing)}")
    st.stop()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[NUMERIC_FEATURES].copy()
    for col in NUMERIC_FEATURES:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x.fillna(x.median(numeric_only=True)).fillna(0)


def predict_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x = prepare_features(df)
    x_scaled = scaler.transform(x)
    probability = model.predict_proba(x_scaled)[:, 1] * 100
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(x_scaled)
    if isinstance(raw_shap, list):
        raw_shap = raw_shap[-1]
    shap_values = np.asarray(raw_shap)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    positive_driver_idx = np.argmax(shap_values, axis=1)
    top_drivers = [NUMERIC_FEATURES[i] for i in positive_driver_idx]

    result = df.copy()
    result["Churn_Prob_%"] = np.round(probability, 2)
    result["Risk_Level"] = [risk_band(p) for p in probability]
    result["Top_Reason"] = top_drivers
    result["Recommended_Action"] = [
        retention_action(d, r) for d, r in zip(top_drivers, result["Risk_Level"])
    ]
    return result, shap_values


@st.cache_data(show_spinner=False)
def reference_model_comparison():
    if reference_df is None:
        return pd.DataFrame()
    df = reference_df.copy()
    np.random.seed(42)
    y = np.where(
        (df["Profit"] < 50)
        | ((df["Discount_Rate"] > 0.2) & (df["Quantity"] <= 2)),
        1,
        0,
    )
    noise = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
    y = np.where(noise == 1, 1 - y, y)
    x = prepare_features(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    local_scaler = StandardScaler()
    x_train = local_scaler.fit_transform(x_train)
    x_test = local_scaler.transform(x_test)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }
    rows = []
    for name, candidate in candidates.items():
        candidate.fit(x_train, y_train)
        pred = candidate.predict(x_test)
        prob = candidate.predict_proba(x_test)[:, 1]
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, prob),
            }
        )
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)


# Calculate predictions once and keep them in session state for all tabs.
if "results_df" not in st.session_state or st.session_state.get("source_signature") != source_df.shape:
    results, shap_values = predict_frame(source_df)
    st.session_state.results_df = results
    st.session_state.shap_values = shap_values
    st.session_state.source_signature = source_df.shape
else:
    results = st.session_state.results_df
    shap_values = st.session_state.shap_values

high = int((results["Risk_Level"] == "High").sum())
medium = int((results["Risk_Level"] == "Medium").sum())
low = int((results["Risk_Level"] == "Low").sum())
avg_risk = float(results["Churn_Prob_%"].mean())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(results):,}")
m2.metric("High Risk", f"{high:,}")
m3.metric("Revenue at Risk", f"${results['Revenue'].sum() * avg_risk / 100:,.0f}" if "Revenue" in results else "N/A")
m4.metric("Average Churn Risk", f"{avg_risk:.1f}%")

st.divider()

tabs = st.tabs(
    [
        "🏠 Customer 360",
        "📊 Analytics",
        "🧠 Explainability",
        "🎯 Threshold",
        "📦 Batch Scoring",
        "🔬 What-if Simulator",
        "💰 ROI Simulator",
        "🛡️ Model Monitoring",
    ]
)

with tabs[0]:
    st.subheader("Customer 360")
    st.caption("One place to understand risk, drivers and the next best retention action.")
    id_candidates = [c for c in ["customerID", "Customer_ID", "CustomerID", "Order_ID"] if c in results]
    if id_candidates:
        id_col = id_candidates[0]
        selected = st.selectbox("Select customer", results.index, format_func=lambda i: str(results.loc[i, id_col]))
    else:
        selected = st.selectbox("Select customer row", results.index)
    row = results.loc[selected]
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{row['Churn_Prob_%']:.1f}%")
    c2.metric("Risk tier", str(row["Risk_Level"]))
    c3.metric("Top driver", str(row["Top_Reason"]))
    st.markdown(f"### Recommended action\n**{row['Recommended_Action']}**")
    profile_cols = [c for c in results.columns if c not in ["Churn_Prob_%", "Risk_Level", "Top_Reason", "Recommended_Action"]]
    st.dataframe(row[profile_cols].to_frame("Value"), use_container_width=True)

with tabs[1]:
    st.subheader("Business Analytics")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Risk distribution")
        st.bar_chart(results["Risk_Level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0))
    with right:
        st.markdown("#### Churn probability distribution")
        st.bar_chart(results["Churn_Prob_%"].round().value_counts().sort_index())

    st.markdown("#### Model comparison")
    comparison = reference_model_comparison()
    if not comparison.empty:
        st.dataframe(comparison.style.format({c: "{:.3f}" for c in comparison.columns[1:]}), use_container_width=True)
        st.bar_chart(comparison.set_index("Model")["ROC-AUC"])
    else:
        st.info("Reference dataset is unavailable for model comparison.")

    st.markdown("#### Highest-risk customers")
    show_cols = [c for c in ["customerID", "Customer_ID"] if c in results] + ["Churn_Prob_%", "Risk_Level", "Top_Reason", "Recommended_Action"]
    st.dataframe(results.sort_values("Churn_Prob_%", ascending=False)[show_cols].head(20), use_container_width=True)

with tabs[2]:
    st.subheader("SHAP Explainability Center")
    x = prepare_features(results)
    global_importance = pd.DataFrame({"Feature": NUMERIC_FEATURES, "Mean |SHAP|": np.abs(shap_values).mean(axis=0)}).sort_values("Mean |SHAP|", ascending=False)
    st.bar_chart(global_importance.set_index("Feature"))

    selected = st.selectbox("Explain a customer", results.index, key="shap_customer")
    local = pd.DataFrame({"Feature": NUMERIC_FEATURES, "SHAP impact": shap_values[selected], "Value": x.loc[selected].values})
    local = local.reindex(local["SHAP impact"].abs().sort_values(ascending=False).index)
    st.dataframe(local, use_container_width=True)
    st.caption("Positive SHAP values push the prediction toward churn; negative values push it away from churn.")

with tabs[3]:
    st.subheader("Business Threshold Optimization")
    st.caption("The default 70% high-risk boundary can be changed according to the cost of missed churn versus intervention.")
    threshold = st.slider("High-risk threshold (%)", 10, 95, 70)
    high_at_threshold = results["Churn_Prob_%"] >= threshold
    st.metric("Customers targeted", int(high_at_threshold.sum()))
    st.metric("Target share", f"{high_at_threshold.mean() * 100:.1f}%")
    if "Churn" in source_df.columns:
        actual = pd.to_numeric(source_df["Churn"], errors="coerce").fillna(0).astype(int)
        pred = high_at_threshold.astype(int)
        st.write({"Precision": round(precision_score(actual, pred, zero_division=0), 3), "Recall": round(recall_score(actual, pred, zero_division=0), 3), "F1": round(f1_score(actual, pred, zero_division=0), 3)})
    else:
        st.info("Upload a labeled dataset containing `Churn` to calculate threshold performance against actual outcomes.")

with tabs[4]:
    st.subheader("Batch Prediction & Export")
    st.write(f"Scored **{len(results):,}** customers from the current dataset.")
    risk_choice = st.multiselect("Export risk tiers", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    export_df = results[results["Risk_Level"].isin(risk_choice)]
    st.dataframe(export_df.head(100), use_container_width=True)
    st.download_button("⬇️ Download scored customers", export_df.to_csv(index=False), "customer_churn_predictions.csv", "text/csv")

with tabs[5]:
    st.subheader("What-if Churn Simulator")
    st.caption("Change customer attributes and immediately see how the predicted risk changes.")
    defaults = prepare_features(source_df).median()
    cols = st.columns(3)
    values = {}
    for i, feature in enumerate(NUMERIC_FEATURES):
        with cols[i % 3]:
            values[feature] = st.number_input(feature, value=float(defaults[feature]), step=1.0, key=f"whatif_{feature}")
    whatif = pd.DataFrame([values])
    wx = scaler.transform(whatif[NUMERIC_FEATURES])
    wp = float(model.predict_proba(wx)[0, 1] * 100)
    wrisk = risk_band(wp)
    wc1, wc2, wc3 = st.columns(3)
    wc1.metric("Simulated churn risk", f"{wp:.1f}%")
    wc2.metric("Risk tier", wrisk)
    wc3.metric("Action", retention_action(" / ".join(NUMERIC_FEATURES), wrisk))

with tabs[6]:
    st.subheader("Retention ROI Simulator")
    st.caption("Estimate whether targeting high-risk customers is economically worthwhile.")
    default_target = max(high, 1)
    rc1, rc2 = st.columns(2)
    with rc1:
        target_count = st.number_input("Customers targeted", min_value=0, value=default_target, step=1)
        revenue_risk = st.number_input("Revenue at risk ($)", min_value=0.0, value=float(results["Revenue"].sum() * avg_risk / 100) if "Revenue" in results else 10000.0, step=100.0)
    with rc2:
        intervention = st.number_input("Intervention cost / customer ($)", min_value=0.0, value=25.0, step=5.0)
        save_rate = st.slider("Expected save rate (%)", 0, 100, 30)
    roi = roi_estimate(target_count, revenue_risk, intervention, save_rate)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Expected value saved", f"${roi['expected_saved']:,.0f}")
    r2.metric("Intervention cost", f"${roi['intervention_cost']:,.0f}")
    r3.metric("Net value", f"${roi['net_value']:,.0f}")
    r4.metric("Estimated ROI", f"{roi['roi_percent']:.1f}%")

with tabs[7]:
    st.subheader("Model Monitoring")
    if reference_df is None:
        st.warning("Reference dataset unavailable; drift monitoring cannot be calculated.")
    else:
        ref = prepare_features(reference_df)
        cur = prepare_features(source_df)
        rows = []
        for feature in NUMERIC_FEATURES:
            ref_mean, cur_mean = float(ref[feature].mean()), float(cur[feature].mean())
            ref_std = float(ref[feature].std()) or 1.0
            standardized_shift = abs(cur_mean - ref_mean) / ref_std
            rows.append({"Feature": feature, "Reference mean": ref_mean, "Current mean": cur_mean, "Standardized shift": standardized_shift, "Status": "⚠️ Review" if standardized_shift > 1 else "✅ Stable"})
        drift_df = pd.DataFrame(rows).sort_values("Standardized shift", ascending=False)
        st.dataframe(drift_df, use_container_width=True)
        st.bar_chart(drift_df.set_index("Feature")["Standardized shift"])
        st.caption("This is a lightweight monitoring signal based on standardized mean shift. Production monitoring should also track PSI/KS, missingness, label delay and model performance over time.")

st.divider()
st.caption("Built for explainable, action-oriented customer retention decisions. No FastAPI layer is used in this version.")
