from __future__ import annotations

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retention import retention_action, risk_band, roi_estimate

st.set_page_config(page_title="Customer Churn Action System", page_icon="📉", layout="wide")
MODEL, SCALER, FEATURE_FILE = [ROOT / p for p in ("models/best_churn_model.pkl", "models/scaler.pkl", "models/feature_names.pkl")]
REFERENCE = ROOT / "data/Business_Analytics_Dataset_10000_Rows.csv"

@st.cache_resource
def load_artifacts():
    if not all(p.exists() for p in (MODEL, SCALER, FEATURE_FILE)):
        return None, None, None
    return joblib.load(MODEL), joblib.load(SCALER), joblib.load(FEATURE_FILE)

@st.cache_data
def load_reference():
    return pd.read_csv(REFERENCE) if REFERENCE.exists() else None

model, scaler, expected_features = load_artifacts()
reference = load_reference()
if model is None:
    st.error("Model artifacts not found. Run `python src/train_dashboard.py` first.")
    st.stop()

st.title("📉 Customer Churn Prediction & Action System")
st.caption("Predict risk → explain why → recommend action → measure business impact")

uploaded = st.sidebar.file_uploader("Upload customer CSV", type="csv")
df = pd.read_csv(uploaded) if uploaded is not None else reference
if df is None:
    st.warning("Upload a CSV containing the model features.")
    st.stop()

FEATURES = list(expected_features)
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Missing model features: {', '.join(missing)}")
    st.stop()

def prepare(data):
    x = data[FEATURES].copy()
    for c in FEATURES:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.fillna(x.median(numeric_only=True)).fillna(0)

def predict(data):
    x = prepare(data)
    xs = scaler.transform(x)
    probability = model.predict_proba(xs)[:, 1] * 100
    values = shap.TreeExplainer(model).shap_values(xs)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    drivers = [FEATURES[i] for i in np.argmax(values, axis=1)]
    out = data.copy()
    out["Churn_Prob_%"] = probability.round(2)
    out["Risk_Level"] = [risk_band(v) for v in probability]
    out["Top_Reason"] = drivers
    out["Recommended_Action"] = [retention_action(d, r) for d, r in zip(drivers, out["Risk_Level"])]
    return out, values

results, shap_values = predict(df)
high = int((results["Risk_Level"] == "High").sum())
avg_risk = float(results["Churn_Prob_%"].mean())
revenue_risk = float(results["Revenue"].sum() * avg_risk / 100) if "Revenue" in results else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(results):,}")
m2.metric("High Risk", f"{high:,}")
m3.metric("Average Churn Risk", f"{avg_risk:.1f}%")
m4.metric("Revenue at Risk", f"${revenue_risk:,.0f}" if revenue_risk else "N/A")

tabs = st.tabs(["🏠 Customer 360", "📊 Analytics", "🧠 Explainability", "🎯 Threshold", "📦 Batch", "🔬 What-if", "💰 ROI", "🛡️ Monitoring"])

with tabs[0]:
    st.subheader("Customer 360")
    ids = [c for c in ("customerID", "Customer_ID", "CustomerID", "Order_ID") if c in results]
    idx = st.selectbox("Customer", results.index, format_func=(lambda i: str(results.loc[i, ids[0]])) if ids else None)
    row = results.loc[idx]
    a, b, c = st.columns(3)
    a.metric("Churn probability", f"{row['Churn_Prob_%']:.1f}%")
    b.metric("Risk tier", row["Risk_Level"])
    c.metric("Top driver", row["Top_Reason"])
    st.success(f"Recommended action: {row['Recommended_Action']}")
    st.dataframe(row.to_frame("Value"), use_container_width=True)

with tabs[1]:
    st.subheader("Business Analytics & Model Comparison")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Risk distribution")
        st.bar_chart(results["Risk_Level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0))
    with right:
        st.markdown("#### Churn probability")
        st.bar_chart(results["Churn_Prob_%"].round().value_counts().sort_index())
    if reference is not None:
        y = np.where((reference.Profit < 50) | ((reference.Discount_Rate > .2) & (reference.Quantity <= 2)), 1, 0)
        rng = np.random.default_rng(42)
        y = np.where(rng.choice([0, 1], len(y), p=[.85, .15]) == 1, 1 - y, y)
        x = prepare(reference)
        xt, xv, yt, yv = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)
        s = StandardScaler(); xt = s.fit_transform(xt); xv = s.transform(xv)
        candidates = {"Logistic Regression": LogisticRegression(max_iter=1000), "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42), "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)}
        rows = []
        for name, candidate in candidates.items():
            candidate.fit(xt, yt); pred = candidate.predict(xv); prob = candidate.predict_proba(xv)[:, 1]
            rows.append({"Model": name, "Accuracy": accuracy_score(yv, pred), "Precision": precision_score(yv, pred, zero_division=0), "Recall": recall_score(yv, pred, zero_division=0), "F1": f1_score(yv, pred, zero_division=0), "ROC-AUC": roc_auc_score(yv, prob)})
        comparison = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
        st.dataframe(comparison.style.format({c: "{:.3f}" for c in comparison.columns[1:]}), use_container_width=True)
        st.bar_chart(comparison.set_index("Model")["ROC-AUC"])
    cols = [c for c in ("customerID", "Customer_ID") if c in results] + ["Churn_Prob_%", "Risk_Level", "Top_Reason", "Recommended_Action"]
    st.dataframe(results.sort_values("Churn_Prob_%", ascending=False)[cols].head(25), use_container_width=True)

with tabs[2]:
    st.subheader("SHAP Explainability Center")
    global_imp = pd.DataFrame({"Feature": FEATURES, "Mean |SHAP|": np.abs(shap_values).mean(axis=0)}).sort_values("Mean |SHAP|", ascending=False)
    st.bar_chart(global_imp.set_index("Feature"))
    idx = st.selectbox("Explain customer", results.index, key="explain")
    local = pd.DataFrame({"Feature": FEATURES, "SHAP impact": shap_values[idx], "Value": prepare(results).loc[idx].values})
    local = local.iloc[local["SHAP impact"].abs().argsort()[::-1]]
    st.dataframe(local, use_container_width=True)
    st.caption("Positive SHAP pushes toward churn; negative SHAP pushes away from churn.")

with tabs[3]:
    st.subheader("Business Threshold Optimization")
    threshold = st.slider("High-risk threshold (%)", 10, 95, 70)
    targeted = results["Churn_Prob_%"] >= threshold
    a, b = st.columns(2)
    a.metric("Customers targeted", int(targeted.sum()))
    b.metric("Target share", f"{targeted.mean()*100:.1f}%")
    if "Churn" in df.columns:
        actual = pd.to_numeric(df["Churn"], errors="coerce").fillna(0).astype(int)
        pred = targeted.astype(int)
        st.write({"Precision": round(precision_score(actual, pred, zero_division=0), 3), "Recall": round(recall_score(actual, pred, zero_division=0), 3), "F1": round(f1_score(actual, pred, zero_division=0), 3)})
    else:
        st.info("Upload labeled data with a `Churn` column to validate the threshold.")

with tabs[4]:
    st.subheader("Batch Prediction & Export")
    risk_filter = st.multiselect("Export risk tiers", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    export = results[results["Risk_Level"].isin(risk_filter)]
    st.dataframe(export.head(100), use_container_width=True)
    st.download_button("⬇️ Download scored CSV", export.to_csv(index=False), "customer_churn_predictions.csv", "text/csv")

with tabs[5]:
    st.subheader("What-if Churn Simulator")
    defaults = prepare(df).median()
    values = {}
    boxes = st.columns(3)
    for i, feature in enumerate(FEATURES):
        with boxes[i % 3]:
            values[feature] = st.number_input(feature, value=float(defaults[feature]), step=1.0, key=f"whatif_{feature}")
    one = pd.DataFrame([values])
    risk = float(model.predict_proba(scaler.transform(one[FEATURES]))[0, 1] * 100)
    band = risk_band(risk)
    a, b = st.columns(2)
    a.metric("Simulated churn risk", f"{risk:.1f}%")
    b.metric("Risk tier", band)
    st.success(retention_action(" / ".join(FEATURES), band))

with tabs[6]:
    st.subheader("Retention ROI Simulator")
    n = st.number_input("Customers targeted", min_value=0, value=max(high, 1), step=1)
    risk_value = st.number_input("Revenue at risk ($)", min_value=0.0, value=revenue_risk or 10000.0, step=100.0)
    cost = st.number_input("Intervention cost / customer ($)", min_value=0.0, value=25.0, step=5.0)
    save_rate = st.slider("Expected save rate (%)", 0, 100, 30)
    roi = roi_estimate(n, risk_value, cost, save_rate)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Expected value saved", f"${roi['expected_saved']:,.0f}")
    r2.metric("Intervention cost", f"${roi['intervention_cost']:,.0f}")
    r3.metric("Net value", f"${roi['net_value']:,.0f}")
    r4.metric("ROI", f"{roi['roi_percent']:.1f}%")

with tabs[7]:
    st.subheader("Model & Data Monitoring")
    if reference is None:
        st.warning("Reference data unavailable.")
    else:
        ref, cur = prepare(reference), prepare(df)
        rows = []
        for feature in FEATURES:
            mean_ref, mean_cur = ref[feature].mean(), cur[feature].mean()
            std_ref = ref[feature].std() or 1.0
            shift = abs(mean_cur - mean_ref) / std_ref
            rows.append({"Feature": feature, "Reference mean": mean_ref, "Current mean": mean_cur, "Standardized shift": shift, "Status": "⚠️ Review" if shift > 1 else "✅ Stable"})
        drift = pd.DataFrame(rows).sort_values("Standardized shift", ascending=False)
        st.dataframe(drift, use_container_width=True)
        st.bar_chart(drift.set_index("Feature")["Standardized shift"])
        st.caption("Lightweight drift signal. Production monitoring should additionally track PSI/KS, missingness, label delay and live model performance.")

st.divider()
st.caption("Streamlit • XGBoost • SHAP • Customer 360 • Batch scoring • What-if • ROI • Monitoring")
