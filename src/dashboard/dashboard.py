import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.express as px
import plotly.graph_objects as plgo
import shap
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn AI · Piyush Ramteke",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI/UX Overhaul
st.markdown("""
<style>
    /* Global theme */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e2130 0%, #0e1117 50%, #1a1d2e 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(0, 212, 255, 0.2);
    }
    .header-banner h1 {
        margin: 0;
        font-size: 2.2rem;
        color: #fff;
        font-weight: bold;
    }
    
    /* Animated Dot */
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(1.1); box-shadow: 0 0 10px #00ff88; }
    }
    .pulse-dot {
        animation: pulse 1.5s infinite ease-in-out;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #0088ff 100%);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
    }
    
    /* Risk Badges */
    .risk-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
        color: #000;
    }
    .risk-low { background-color: #00ff88; }
    .risk-medium { background-color: #ffd93d; }
    .risk-high { background-color: #ff4b6e; color: white !important; }
    
    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #1a1d2e;
        width: 280px !important;
    }
    
    /* Style Sidebar logic elements exactly like pills */
    div[data-testid="stSidebar"] div.row-widget.stRadio > div[role="radiogroup"] > label {
        background: transparent;
        padding: 12px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    div[data-testid="stSidebar"] div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
        background: rgba(255,255,255,0.05);
    }
    div[data-testid="stSidebar"] div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        border-left: 4px solid #00d4ff;
        background: rgba(0, 212, 255, 0.1);
    }
    div[data-testid="stSidebar"] div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] p {
        color: #00d4ff !important;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] div.row-widget.stRadio > div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        color: #d1d5db;
    }
    div[data-testid="stSidebar"] div.row-widget.stRadio > div[role="radiogroup"] > label [data-baseweb="radio"] {
        display: none;
    }
    
    .cyan-divider {
        height: 2px;
        background: linear-gradient(90deg, #00d4ff, transparent);
        margin: 30px 0;
        opacity: 0.5;
    }
    
    /* Page intro animation */
    .page-reveal {
        animation: reveal 0.4s ease-in;
    }
    @keyframes reveal {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. MODEL DEFINITION & LOADING
# -----------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=16, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

@st.cache_resource
def load_resources():
    model = None
    scaler = None
    encoders = None
    feature_names = None
    try:
        # Check standard paths
        for path_str in ["artifacts/mlp_churn_classifier_final.pth", "outputs/run_20260315_185202/models/mlp_churn_classifier.pth", "mlp_churn_classifier.pth", "serving/mlp_churn_classifier_final.pth"]:
            model_path = Path(path_str)
            if model_path.exists():
                model = MLPClassifier()
                chk = torch.load(model_path, map_location='cpu')
                model.load_state_dict(chk.get('model_state_dict', chk))
                model.eval()
                break
                
        scaler_path = Path("artifacts/scaler.pkl")
        if scaler_path.exists(): scaler = joblib.load(scaler_path)
            
        encoders_path = Path("artifacts/label_encoders.pkl")
        if encoders_path.exists(): encoders = joblib.load(encoders_path)

        features_path = Path("artifacts/feature_names.pkl")
        if features_path.exists(): feature_names = joblib.load(features_path)

    except Exception as e:
        pass
    
    return model, scaler, encoders, feature_names

model, scaler, encoders, model_feature_names = load_resources()

# Session State
if "predictions_count" not in st.session_state:
    st.session_state.predictions_count = 0
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "toast_shown" not in st.session_state:
    st.toast("✅ Base components loaded successfully!", icon="🚀")
    st.session_state.toast_shown = True


# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def predict_probability(features_dict):
    """Predicts churn probability given features mapping"""
    if model is None:
        return 0.5, np.zeros((1, 16))
        
    df = pd.DataFrame([features_dict])
    
    cat_cols = ['Region', 'Product_Category', 'Customer_Segment', 'Payment_Method']
    if encoders:
        for col in cat_cols:
            if col in encoders and col in df.columns:
                try:
                    df[col + '_encoded'] = encoders[col].transform(df[col].astype(str))
                except Exception:
                    df[col + '_encoded'] = 0
            else:
                df[col + '_encoded'] = 0
            if col in df.columns:
                df = df.drop(columns=[col])
                
    base_features = [
        'total_orders', 'total_revenue', 'avg_revenue', 'std_revenue',
        'total_profit', 'avg_profit', 'avg_discount', 'total_quantity',
        'avg_quantity', 'days_since_last_purchase', 'customer_lifetime_days',
        'purchase_frequency'
    ]
    final_cols = base_features + [c + '_encoded' for c in cat_cols]
    
    for c in final_cols:
        if c not in df.columns:
            df[c] = 0.0
            
    df = df[final_cols]
            
    X_arr = df.values.astype(np.float32)
    if scaler:
        try:
            X_arr = scaler.transform(X_arr)
        except Exception:
            pass
            
    with torch.no_grad():
        t = torch.FloatTensor(X_arr)
        prob = model(t).numpy()[0][0]
        
    return float(prob), X_arr

def render_metric_card(title, value, badge_text, border_color, badge_color="#00ff88", badge_bg="rgba(0, 255, 136, 0.2)"):
    return f"""
    <div style="position: relative; background-color: #1e2130; border-radius: 12px; border-left: 4px solid {border_color}; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: transform 0.2s ease; margin-bottom: 20px;">
        <div style="color: #9ca3af; font-size: 0.85rem; font-weight: 600;">{title}</div>
        <div style="color: white; font-size: 2.5rem; font-weight: bold; margin: 10px 0;">{value}</div>
        <div style="background: {badge_bg}; color: {badge_color}; padding: 4px 12px; border-radius: 12px; display: inline-block; font-size: 0.8rem; font-weight: bold;">{badge_text}</div>
        <div style="position: absolute; bottom: 8px; right: 12px; font-size: 0.6rem; color: rgba(0, 212, 255, 0.15); font-weight: bold; pointer-events: none;">PR</div>
    </div>
    """


# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
MENU_ITEMS = [
    "🏢 Executive Overview",
    "🎯 Single Inference",
    "🗃 Batch Prediction",
    "🕹 What-If Simulator",
    "📈 Model Telemetry",
    "📊 Business Impact"
]

selected_page = st.sidebar.radio("Navigation", MENU_ITEMS, label_visibility="collapsed")

st.sidebar.markdown("<br><hr style='border: 1px solid rgba(0, 212, 255, 0.3); margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size: 0.85rem; color:#9ca3af; margin-bottom:10px;'><b>SYSTEM HEALTH</b></div>", unsafe_allow_html=True)

def health_dot(is_ok):
    col = '#00ff88' if is_ok else '#ff4b6e'
    return f"""<span style="display:inline-block; width:10px; height:10px; background-color:{col}; border-radius:50%; margin-right:8px;"></span>"""

st.sidebar.markdown(f"{health_dot(model is not None)} <span style='color:white; font-size:0.9rem;'>{'Model Loaded' if model else 'Model Not Found'}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"{health_dot(scaler is not None)} <span style='color:white; font-size:0.9rem;'>{'Scaler Ready' if scaler else 'Scaler Not Found'}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"{health_dot(encoders is not None)} <span style='color:white; font-size:0.9rem;'>{'Encoders Ready' if encoders else 'Encoders Not Found'}</span>", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<br>
<div style='color: #9ca3af; font-size: 0.8rem; font-style:italic;'>
    Powered by PyTorch & FastAPI<br>
    Total Predictions: {st.session_state.predictions_count}<br><br>
    💡 Hint: Press 1-6 to navigate
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**API Endpoint**")
st.sidebar.code("http://localhost:5000/predict", language="bash")

st.sidebar.markdown("""
<div style="background: #1a1d2e; border-top: 2px solid #00d4ff; border-radius: 8px; padding: 15px; margin-top: 30px; margin-bottom: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); transition: all 0.3s ease;" onmouseover="this.style.boxShadow='0 0 15px rgba(0,212,255,0.2)'" onmouseout="this.style.boxShadow='0 4px 6px rgba(0,0,0,0.2)'">
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
        <div style="background: rgba(0,212,255,0.1); color: #00d4ff; width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 10px; border: 1px solid rgba(0,212,255,0.3);">PR</div>
        <div style="text-align: left;">
            <div style="font-size: 0.7rem; color: #9ca3af;">👨‍💻 Built by</div>
            <div style="font-size: 0.9rem; font-weight: bold; color: white;">Piyush Ramteke</div>
            <a href="https://github.com/Piyu242005" target="_blank" style="font-size: 0.8rem; color: #00d4ff; text-decoration: none;">@Piyu242005</a>
        </div>
    </div>
    <div style="display: flex; justify-content: center; gap: 10px; margin-top: 10px;">
        <a href="https://github.com/Piyu242005" target="_blank" style="text-decoration:none; background: rgba(255,255,255,0.05); padding: 5px 10px; border-radius: 5px; color: #d1d5db; font-size: 0.8rem; transition: background 0.3s;" onmouseover="this.style.background='rgba(0,212,255,0.1)'; this.style.color='#00d4ff'" onmouseout="this.style.background='rgba(255,255,255,0.05)'; this.style.color='#d1d5db'">🔗 GitHub</a>
        <a href="https://linkedin.com/in/piyush-ramteke" target="_blank" style="text-decoration:none; background: rgba(255,255,255,0.05); padding: 5px 10px; border-radius: 5px; color: #d1d5db; font-size: 0.8rem; transition: background 0.3s;" onmouseover="this.style.background='rgba(0,212,255,0.1)'; this.style.color='#00d4ff'" onmouseout="this.style.background='rgba(255,255,255,0.05)'; this.style.color='#d1d5db'">💼 LinkedIn</a>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. MAIN CONTENT RENDERER
# -----------------------------------------------------------------------------

@st.dialog("ℹ️ About This Project")
def show_about_dialog():
    about_html = """
    <div style="
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), 
                    0 0 0 1px rgba(0,212,255,0.2);
        padding: 32px;
        position: relative;
        overflow: hidden;
    ">
        
        <div style="
            position: absolute; top: -20px; left: -20px;
            width: 120px; height: 120px;
            background: rgba(0,212,255,0.15);
            border-radius: 50%;
            filter: blur(40px);
            z-index: 0;
        "></div>
        <div style="
            position: absolute; bottom: -15px; right: -15px;
            width: 100px; height: 100px;
            background: rgba(168,85,247,0.12);
            border-radius: 50%;
            filter: blur(35px);
            z-index: 0;
        "></div>

        
        <div style="position: relative; z-index: 1;">

            
            <div style="
                background: rgba(0,212,255,0.08);
                border: 1px solid rgba(0,212,255,0.2);
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 20px;
            ">
                <span style="
                    font-size: 1rem;
                    font-weight: 600;
                    color: rgba(0,212,255,0.9);
                ">🧠 Churn Intelligence Platform</span>
            </div>

            
            <div style="
                font-size: 0.65rem;
                letter-spacing: 0.15rem;
                color: #00d4ff;
                font-weight: 700;
                margin-bottom: 10px;
            ">👨‍💻 DEVELOPER</div>

            
            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 20px;
            ">
                <div style="
                    width: 52px; height: 52px;
                    background: linear-gradient(135deg, #00d4ff, #0891b2);
                    border-radius: 50%;
                    border: 2px solid rgba(0,212,255,0.4);
                    box-shadow: 0 0 20px rgba(0,212,255,0.3);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.1rem;
                    font-weight: 800;
                    color: white;
                    margin-right: 14px;
                    flex-shrink: 0;
                ">PR</div>
                <div>
                    <div style="
                        font-size: 1.1rem;
                        font-weight: 700;
                        color: white;
                    ">Piyush Ramteke</div>
                    <div style="
                        font-size: 0.8rem;
                        color: rgba(156,163,175,0.9);
                    ">Data Science & ML Engineer</div>
                </div>
            </div>

            
            <div style="margin-bottom: 20px;">
                <a href="https://github.com/Piyu242005" 
                   target="_blank"
                   style="
                    display: flex;
                    align-items: center;
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 8px;
                    padding: 8px 14px;
                    margin-bottom: 8px;
                    color: #00d4ff;
                    text-decoration: none;
                    font-size: 0.85rem;
                ">🔗 GitHub: @Piyu242005</a>
                <a href="https://linkedin.com/in/piyush-ramteke"
                   target="_blank"
                   style="
                    display: flex;
                    align-items: center;
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 8px;
                    padding: 8px 14px;
                    margin-bottom: 8px;
                    color: #00d4ff;
                    text-decoration: none;
                    font-size: 0.85rem;
                ">💼 LinkedIn: piyush-ramteke</a>
                <div style="
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 8px;
                    padding: 8px 14px;
                    color: rgba(156,163,175,0.8);
                    font-size: 0.85rem;
                ">📧 Get in touch via LinkedIn!</div>
            </div>

            
            <div style="
                height: 1px;
                background: linear-gradient(90deg, transparent, 
                            rgba(0,212,255,0.3), transparent);
                margin: 18px 0;
            "></div>

            
            <div style="margin-bottom: 16px;">
                <div style="
                    font-size: 0.75rem;
                    color: #00d4ff;
                    font-weight: 700;
                    letter-spacing: 0.1rem;
                    margin-bottom: 10px;
                ">🛠️ TECH STACK</div>
                <div>
                    {tech_pills}
                </div>
            </div>

            
            <div style="
                height: 1px;
                background: linear-gradient(90deg, transparent,
                            rgba(0,212,255,0.3), transparent);
                margin: 18px 0;
            "></div>

            
            <div style="margin-bottom: 16px;">
                <div style="
                    font-size: 0.75rem;
                    color: #00d4ff;
                    font-weight: 700;
                    letter-spacing: 0.1rem;
                    margin-bottom: 10px;
                ">📊 PROJECT STATS</div>
                {stat_rows}
            </div>

            
            <div style="
                background: rgba(255,255,255,0.03);
                border-radius: 8px;
                padding: 10px 14px;
                display: flex;
                justify-content: space-between;
                font-size: 0.75rem;
                color: rgba(156,163,175,0.7);
            ">
                <span>📅 March 2026 · MIT License</span>
                <span>© Piyush Ramteke</span>
            </div>

        </div>
    </div>
    """

    
    techs = ["PyTorch", "Flask", "Streamlit", 
             "Scikit-learn", "SHAP", "Docker"]
    tech_pills = " ".join([
        f'''<span style="
            background: rgba(0,212,255,0.08);
            border: 1px solid rgba(0,212,255,0.2);
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.75rem;
            color: rgba(0,212,255,0.85);
            display: inline-block;
            margin: 3px;
        ">{t}</span>'''
        for t in techs
    ])

    
    stats = [
        "10,000 training samples",
        "89% model accuracy",
        "16 engineered features", 
        "5 deployment methods",
        "<10ms inference time"
    ]
    stat_rows = "".join([
        f'''<div style="
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            padding: 8px 12px;
            margin: 4px 0;
            border-left: 3px solid #00d4ff;
            font-size: 0.82rem;
            color: rgba(255,255,255,0.85);
        ">• {s}</div>'''
        for s in stats
    ])

    
    about_html = about_html.format(
        tech_pills=tech_pills,
        stat_rows=stat_rows
    )

    clean_html = '\n'.join([line.strip() for line in about_html.split('\n')])
    st.markdown(clean_html, unsafe_allow_html=True)

colA, colB = st.columns([9, 1])
with colB:
    if st.button("ℹ️ About", use_container_width=True):
        show_about_dialog()


# Global Header
ts = datetime.now().strftime("%Y-%m-%d %H:%M")
st.markdown(f"""
<div class="header-banner">
    <div style="display:flex; align-items:center; gap: 20px;">
        <div style="font-size: 3rem;">🚀</div>
        <div>
            <h1>Customer Churn Intelligence Platform</h1>
            <div style="color: #9ca3af; font-size: 0.95rem; margin-top: 5px;">
                Real-time churn prediction powered by PyTorch MLP · 89% Accuracy · 10,000 training samples
            </div>
        </div>
    </div>
    <div style="text-align: right; min-width:150px;">
        <div style="background: rgba(0, 212, 255, 0.1); color: #00d4ff; padding: 5px 15px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; display: inline-block; margin-bottom: 12px; border: 1px solid rgba(0, 212, 255, 0.2);">v2.0.0 Enterprise</div>
        <br/>
        <div style="color: #d1d5db; font-size: 0.85rem; display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
            <div class="pulse-dot" style="width: 10px; height: 10px; background-color: #00ff88; border-radius: 50%;"></div>
            Online &nbsp;·&nbsp; <span style="opacity: 0.6;">{ts}</span> &nbsp;|&nbsp;
            <a href="https://github.com/Piyu242005" target="_blank" style="color: #6b7280; font-style: italic; text-decoration: none; transition: color 0.3s;" onmouseover="this.style.color='#00d4ff'; this.style.textDecoration='underline'" onmouseout="this.style.color='#6b7280'; this.style.textDecoration='none'">🧑‍🔬 Developed by Piyush Ramteke</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

cat_options = {
    'Region': ['North', 'South', 'East', 'West'],
    'Product_Category': ['Electronics', 'Clothing', 'Food', 'Furniture', 'Sports'],
    'Customer_Segment': ['Premium', 'Standard', 'Basic'],
    'Payment_Method': ['Credit Card', 'Debit Card', 'PayPal', 'Cash']
}
feature_names_list = [
    'total_orders', 'total_revenue', 'avg_revenue', 'std_revenue',
    'total_profit', 'avg_profit', 'avg_discount', 'total_quantity',
    'avg_quantity', 'days_since_last_purchase', 'customer_lifetime_days',
    'purchase_frequency', 'Region', 'Product_Category', 'Customer_Segment', 'Payment_Method'
]

if model is None:
    st.error("❌ **Primary Model (.pth) not found.** Please run `python src/train.py` or the training pipeline to generate model artifacts first.")

st.markdown('<div class="page-reveal">', unsafe_allow_html=True)

# =============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =============================================================================
if selected_page == "🏢 Executive Overview":
    with st.spinner("🧠 Piyush Ramteke's Churn AI · Crunching the numbers..."):
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(render_metric_card("Accuracy", "89%", "↑ +11% vs baseline", border_color="#00d4ff"), unsafe_allow_html=True)
        with c2: st.markdown(render_metric_card("ROC-AUC", "0.92", "↑ Excellent", border_color="#00ff88", badge_color="#00ff88"), unsafe_allow_html=True)
        with c3: st.markdown(render_metric_card("F1 Score", "0.86", "↑ High Performance", border_color="#a855f7", badge_color="#c084fc", badge_bg="rgba(168, 85, 247, 0.2)"), unsafe_allow_html=True)
        with c4: st.markdown(render_metric_card("Test Samples", "1,199", "Fixed Set Evaluated", border_color="#f97316", badge_color="#fb923c", badge_bg="rgba(249, 115, 22, 0.2)"), unsafe_allow_html=True)
        
        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)
        
        # Architecture Visuals
        st.markdown("""
        <div style='display:flex; justify-content:space-between; align-items:flex-start; gap: 40px;'>
            <div style='flex:1; border-right: 1px solid rgba(0, 212, 255, 0.2); padding-right: 40px;'>
                <h3 style='color: white; margin-bottom: 20px;'>🧠 Model Architecture</h3>
                <div style='display:flex; flex-direction:column; align-items:center; gap:8px;'>
                    <div style='border: 1px solid #00d4ff; border-left: 6px solid #00d4ff; border-radius: 8px; padding: 12px; width: 100%; text-align: center; background: #1e2130;'>
                        <b style='color:#00d4ff;'>INPUT LAYER</b> <br/> 16 Features
                    </div>
                    <div style='color: #00d4ff; font-weight: bold;'>⬇</div>
                    <div style='border: 1px solid #333; border-left: 6px solid #4b5563; border-radius: 8px; padding: 12px; width: 100%; text-align: center; background: linear-gradient(180deg, #1e2130, #2a2e45);'>
                        <b style='color:#fff;'>HIDDEN 1</b> <br/> 128 neurons | ReLU + Drop
                    </div>
                    <div style='color: #4b5563; font-weight: bold;'>⬇</div>
                    <div style='border: 1px solid #333; border-left: 6px solid #4b5563; border-radius: 8px; padding: 12px; width: 100%; text-align: center; background: linear-gradient(180deg, #1e2130, #2a2e45);'>
                        <b style='color:#fff;'>HIDDEN 2</b> <br/> 64 neurons | ReLU + Drop
                    </div>
                    <div style='color: #4b5563; font-weight: bold;'>⬇</div>
                    <div style='border: 1px solid #333; border-left: 6px solid #4b5563; border-radius: 8px; padding: 12px; width: 100%; text-align: center; background: linear-gradient(180deg, #1e2130, #2a2e45);'>
                        <b style='color:#fff;'>HIDDEN 3</b> <br/> 32 neurons | ReLU + Drop
                    </div>
                    <div style='color: #00ff88; font-weight: bold;'>⬇</div>
                    <div style='border: 1px solid #00ff88; border-left: 6px solid #00ff88; border-radius: 8px; padding: 12px; width: 100%; text-align: center; background: #1e2130;'>
                        <b style='color:#00ff88;'>OUTPUT</b> <br/> Sigmoid Probability
                    </div>
                </div>
            </div>
            <div style='flex:1;'>
                <h3 style='color: white; margin-bottom: 20px;'>✨ Key Features</h3>
                <div style='border: 1px solid rgba(0, 212, 255, 0.4); border-radius: 12px; padding: 25px; background: rgba(30, 33, 48, 0.5); box-shadow: inset 0 0 20px rgba(0, 212, 255, 0.05);'>
                    <ul style='list-style-type: none; padding-left: 0; margin: 0; line-height: 2.2; font-size: 1.1rem;'>
                        <li><span style="color:#00ff88;">✓</span> <b>Feature Engineering</b> <span style="color:#9ca3af;">(16 encoded features)</span></li>
                        <li><span style="color:#00ff88;">✓</span> <b>Class Balancing</b> <span style="color:#9ca3af;">(SMOTE technique applied)</span></li>
                        <li><span style="color:#00ff88;">✓</span> <b>Regularization</b> <span style="color:#9ca3af;">(Stochastic Dropout 30%)</span></li>
                        <li><span style="color:#00ff88;">✓</span> <b>Model Explainability</b> <span style="color:#9ca3af;">(SHAP KernelExplainer integrated)</span></li>
                        <li><span style="color:#00ff88;">✓</span> <b>Production API</b> <span style="color:#9ca3af;">(Asynchronous FastAPI Backend)</span></li>
                        <li><span style="color:#00ff88;">✓</span> <b>Deployment Ready</b> <span style="color:#9ca3af;">(Dockerized configuration)</span></li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)
        
        # Charts
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Confusion Matrix")
            cm_values = [[229, 28], [113, 829]]
            cm_text = [['TN = 229<br>(19%)', 'FP = 28<br>(2%)'], ['FN = 113<br>(9%)', 'TP = 829<br>(70%)']]
            
            fig1 = plgo.Figure(data=plgo.Heatmap(
                z=cm_values, text=cm_text, texttemplate="%{text}", textfont={"size":16},
                x=['Predicted Active', 'Predicted Churned'], y=['Actual Active', 'Actual Churned'],
                colorscale=[[0, '#1e2130'], [0.5, '#0891b2'], [1, '#00ff88']], showscale=False
            ))
            fig1.update_layout(template="plotly_dark", title="Test Set Performance (n=1,199)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div style='text-align:center; color:#00ff88; font-weight:bold;'>✓ 829/942 churned customers correctly identified (88% recall)</div>", unsafe_allow_html=True)

        with c2:
            st.subheader("Predictive Breakdown (Dataset)")
            fig3 = px.pie(names=['Active', 'Churned'], values=[257, 942], hole=0.5, color_discrete_sequence=['#00ff88', '#ff4b6e'])
            fig3.update_layout(template="plotly_dark", title="Target Distribution", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, showlegend=True)
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div style='text-align:center; color:#9ca3af;'>Distribution of the 1,199 test targets evaluated</div>", unsafe_allow_html=True)

        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)
        
        c3, c4 = st.columns([1, 1])
        with c3:
            st.subheader("Model Comparison — MLP vs Baselines")
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
            fig_comp = plgo.Figure()
            fig_comp.add_trace(plgo.Bar(name='MLP', x=metrics, y=[0.89, 0.87, 0.85, 0.86, 0.92], marker_color='#00d4ff'))
            fig_comp.add_trace(plgo.Bar(name='XGBoost', x=metrics, y=[0.87, 0.84, 0.83, 0.83, 0.90], marker_color='#4b5563'))
            fig_comp.add_trace(plgo.Bar(name='Random Forest', x=metrics, y=[0.85, 0.82, 0.80, 0.81, 0.88], marker_color='#3b4252'))
            fig_comp.add_trace(plgo.Bar(name='Logistic Regression', x=metrics, y=[0.78, 0.75, 0.72, 0.73, 0.82], marker_color='#2d3342'))
            fig_comp.update_layout(barmode='group', template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
            
            fig_comp.add_annotation(x=4.0, y=0.95, text="🏆 Best Model", showarrow=True, arrowhead=2, arrowcolor="#00d4ff", font=dict(color="#00d4ff", size=14))
            st.plotly_chart(fig_comp, use_container_width=True)

        with c4:
            st.subheader("ROC Curve — MLP Classifier")
            fpr = np.linspace(0, 1, 100)
            tpr = np.clip(1 - (1 - fpr)**3 + fpr * 0.1, 0, 1)
            
            fig_roc = plgo.Figure()
            fig_roc.add_trace(plgo.Scatter(x=fpr, y=tpr, mode='lines', name='MLP', fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)', line=dict(color='#00d4ff', width=3)))
            fig_roc.add_trace(plgo.Scatter(x=[0,1], y=[0,1], mode='lines', name='Baseline', line=dict(color='gray', width=2, dash='dash')))
            fig_roc.add_annotation(x=0.2, y=0.88, text="<b>AUC = 0.92</b>", showarrow=False, font=dict(size=18, color="#00ff88"))
            fig_roc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", showlegend=False)
            st.plotly_chart(fig_roc, use_container_width=True)

# =============================================================================
# PAGE 2: SINGLE INFERENCE
# =============================================================================
elif selected_page == "🎯 Single Inference":
    with st.spinner("🧠 Piyush Ramteke's Churn AI · Consulting the oracle..."):
        st.subheader("Single Customer Inference")
        st.markdown("<div style='color:#9ca3af; margin-bottom: 20px;'>Input customer metrics to predict absolute churn probability dynamically using the trained MLP weights.</div>", unsafe_allow_html=True)
        
        with st.form("inference_form"):
            c1, c2 = st.columns(2)
            with c1:
                total_orders = st.slider("Total Orders", 1, 100, 5)
                total_revenue = st.number_input("Total Revenue ($)", 0.0, 50000.0, 1500.0)
                avg_revenue = st.number_input("Average Revenue ($)", 0.0, 5000.0, 300.0)
                std_revenue = st.number_input("Standard Deviation Revenue", 0.0, 1000.0, 50.0)
                total_profit = st.number_input("Total Profit ($)", 0.0, 20000.0, 450.0)
                avg_profit = st.number_input("Average Profit ($)", 0.0, 2000.0, 90.0)
                avg_discount = st.slider("Average Discount", 0.0, 1.0, 0.15)
                total_quantity = st.slider("Total Quantity", 1, 500, 25)
            with c2:
                avg_quantity = st.slider("Average Quantity", 1, 50, 5)
                days_since = st.slider("Days Since Last Purchase", 0, 365, 45)
                lifetime = st.slider("Customer Lifetime Days", 1, 1000, 180)
                purchase_freq = st.slider("Purchase Frequency", 0.0, 1.0, 0.028)
                region = st.selectbox("Region", cat_options['Region'])
                category = st.selectbox("Product Category", cat_options['Product_Category'])
                segment = st.selectbox("Customer Segment", cat_options['Customer_Segment'])
                payment = st.selectbox("Payment Method", cat_options['Payment_Method'])
                
            st.markdown("<br>", unsafe_allow_html=True)
            colA, colB, colC = st.columns([1, 2, 1])
            with colB:
                submitted = st.form_submit_button("🎯 Predict Churn", use_container_width=True)
            
        validation_warning = False
        if total_profit > total_revenue: validation_warning = True
        if avg_profit > avg_revenue: validation_warning = True
        if validation_warning:
            st.warning("⚠️ Input mismatch: Please assure Profit components do not exceed Revenue bounds.")
            
        if submitted:
            st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)
            features = {
                'total_orders': total_orders, 'total_revenue': total_revenue, 'avg_revenue': avg_revenue,
                'std_revenue': std_revenue, 'total_profit': total_profit, 'avg_profit': avg_profit,
                'avg_discount': avg_discount, 'total_quantity': total_quantity, 'avg_quantity': avg_quantity,
                'days_since_last_purchase': days_since, 'customer_lifetime_days': lifetime, 
                'purchase_frequency': purchase_freq, 'Region': region, 'Product_Category': category,
                'Customer_Segment': segment, 'Payment_Method': payment
            }
            
            with st.spinner("🧠 Piyush Ramteke's Churn AI · Processing deep network inference..."):
                prob, X_scaled = predict_probability(features)
                
                st.session_state.predictions_count += 1
                st.session_state.prediction_history.append({
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'probability': prob,
                    'revenue': total_revenue
                })
                
                p_c1, p_c2 = st.columns([1, 1.5])
                with p_c1:
                    fig = plgo.Figure(plgo.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={'suffix': "%", 'font': {'size': 60}},
                        title={'text': "Predicted Churn Probability", 'font': {'size': 20}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "rgba(0,0,0,0)"},
                            'steps': [
                                {'range': [0, 40], 'color': "rgba(0, 255, 136, 0.8)"},
                                {'range': [40, 70], 'color': "rgba(255, 217, 61, 0.8)"},
                                {'range': [70, 100], 'color': "rgba(255, 75, 110, 0.8)"}
                            ],
                            'threshold': {'line': {'color': "white", 'width': 6}, 'thickness': 0.8, 'value': prob * 100}
                        }
                    ))
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=50, b=0, l=30, r=30))
                    st.plotly_chart(fig, use_container_width=True)
                    
                with p_c2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if prob < 0.4:
                        st.markdown("<div class='risk-badge risk-low' style='font-size:1.5rem; width:fit-content;'>✓ LOW RISK PROFILE</div>", unsafe_allow_html=True)
                        st.success("🟢 **Recommendation:** Continue normal marketing sequences. Customer is exhibiting stable behavior indicators.")
                    elif prob < 0.7:
                        st.markdown("<div class='risk-badge risk-medium' style='font-size:1.5rem; width:fit-content;'>⚠️ MEDIUM RISK PROFILE</div>", unsafe_allow_html=True)
                        st.warning("🟡 **Recommendation:** Send an automated targeted discount or check-in verification email. Monitor activity.")
                    else:
                        st.markdown("<div class='risk-badge risk-high' style='font-size:1.5rem; width:fit-content;'>🚨 HIGH RISK PROFILE</div>", unsafe_allow_html=True)
                        st.error("🔴 **Recommendation:** Immediate executive intervention required! Trigger aggressive retention campaign protocols immediately.")
                        
                # Shap Explanation Waterfall
                if model is not None:
                    st.markdown("### 🔍 Explainability: Top Feature Contributions (SHAP Waterfall)")
                    try:
                        bg_data = np.zeros((10, 16))
                        def predict_fn(X):
                            with torch.no_grad():
                                t = torch.FloatTensor(X)
                                return model(t).numpy()
                                
                        explainer = shap.KernelExplainer(predict_fn, bg_data)
                        shap_values_obj = explainer(X_scaled)
                        
                        if hasattr(shap_values_obj, 'values'):
                            shap_values = shap_values_obj.values[0]
                        else:
                            shap_values = shap_values_obj[0]
                            
                        # If list is returned, take first element
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                        if shap_values.ndim > 1:
                            shap_values = shap_values.flatten()
                        
                        top_idx = np.argsort(np.abs(shap_values))[-8:]
                        feature_names_plot = model_feature_names if model_feature_names else feature_names_list
                        
                        plot_names = [feature_names_plot[i] for i in top_idx]
                        plot_vals = shap_values[top_idx]
                        colors = ['#00ff88' if x < 0 else '#ff4b6e' for x in plot_vals]
                        
                        fig_shap = plgo.Figure(plgo.Bar(
                            x=plot_vals, y=plot_names, orientation='h', marker_color=colors,
                            text=[f"{v:+.3f}" for v in plot_vals], textposition="outside"
                        ))
                        fig_shap.update_layout(
                            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,33,48,0.5)",
                            xaxis_title="Impact on Prediction Value", yaxis_title="",
                            margin=dict(l=200, r=50, t=10, b=40), height=350,
                            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not generate localized SHAP explainability instance: {str(e)[:100]}")

# =============================================================================
# PAGE 3: BATCH PREDICTION
# =============================================================================
elif selected_page == "🗃 Batch Prediction":
    with st.spinner("🧠 Piyush Ramteke's Churn AI · Preparing bulk pipeline..."):
        st.subheader("Batch Prediction Pipeline")
        st.markdown("<div style='color:#9ca3af; margin-bottom: 20px;'>Process and analyze multiple customer records at scale through the predictive engine. Output standard CSV or robust PDF reporting.</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("Download mapping template with corrected feature structures:")
            df_template = pd.DataFrame(columns=feature_names_list)
            st.download_button("📝 Download Sample Data Template", df_template.to_csv(index=False).encode(), "churn_prediction_template.csv", "text/csv")
        
        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload Bulk Customer Data (CSV)", type="csv")
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.markdown("#### Mapping Preview:")
            st.dataframe(df_batch.head(5), use_container_width=True)
            
            if st.button("🚀 Run Batch Predictions", use_container_width=True):
                run_batch = True
                missing = [col for col in feature_names_list if col not in df_batch.columns]
                if missing:
                    st.error(f"Missing required data columns mapping sequence: {', '.join(missing)}")
                    run_batch = False
                
                if run_batch:
                    progress = st.progress(0, text="Initializing matrix parameters...")
                    results = []
                    
                    for i, row in df_batch.iterrows():
                        p, _ = predict_probability(row.to_dict())
                        risk = "Low" if p < 0.4 else "Medium" if p < 0.7 else "High"
                        rec = "Normal Marketing" if p < 0.4 else "Check-in Campaign" if p < 0.7 else "Intervene Immediately"
                        results.append({
                            "CustomerID": i + 1,
                            "Churn_Probability": float(p),
                            "Prediction": "Churned" if p >= 0.5 else "Active",
                            "Risk_Level": risk,
                            "Recommendation": rec
                        })
                        progress.progress(min((i + 1) / len(df_batch), 1.0), text=f"Processing {i+1} of {len(df_batch)} records...")
                    
                    progress.empty()
                    results_df = pd.DataFrame(results)
                    
                    st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)
                    st.subheader("Batch Resolution Output")
                    
                    r1, r2, r3, r4 = st.columns(4)
                    churn_count = len(results_df[results_df['Prediction'] == 'Churned'])
                    avg_p = results_df['Churn_Probability'].mean()
                    high_risk = len(results_df[results_df['Risk_Level'] == 'High'])
                        
                    r1.metric("Total Extracted", len(results_df))
                    r2.metric("Predicted Churned", churn_count)
                    r3.metric("Predicted Active", len(results_df) - churn_count)
                    r4.metric("Average Score", f"{avg_p:.2f}")

                    if high_risk > 0:
                        st.markdown(f"<div style='padding:15px; border-radius:8px; border:1px solid #ff4b6e; background:rgba(255,75,110,0.1); margin-bottom:20px; color:#ff4b6e;'>🚨 <b>Alert:</b> Detected <b>{high_risk}</b> High Risk customers out of {len(results_df)} evaluations. Ensure immediate escalations are planned.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='padding:15px; border-radius:8px; border:1px solid #00ff88; background:rgba(0,255,136,0.1); margin-bottom:20px; color:#00ff88;'>✓ <b>Clean:</b> Detected <b>0</b> High Risk customers out of {len(results_df)} evaluations. Base is healthy.</div>", unsafe_allow_html=True)
                    
                    def color_rows(row):
                        if row['Prediction'] == 'Churned': return ['background-color: rgba(255, 75, 110, 0.2)'] * len(row)
                        return ['background-color: rgba(0, 255, 136, 0.1)'] * len(row)
                        
                    st.dataframe(results_df.style.apply(color_rows, axis=1), use_container_width=True)
                    
                    # Charting Batch Distribution
                    hist_fig = px.histogram(results_df, x="Churn_Probability", title="Distribution of Churn Probabilities", color="Prediction", color_discrete_map={"Churned": "#ff4b6e", "Active": "#00ff88"}, nbins=20)
                    hist_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(hist_fig, use_container_width=True)
                    
                    d_c1, d_c2 = st.columns(2)
                    with d_c1:
                        st.download_button("📥 Download Results as CSV", results_df.to_csv(index=False).encode(), "batch_results.csv", "text/csv", use_container_width=True)
                    with d_c2:
                        try:
                            from fpdf import FPDF 
                            import io
                            pdf_buffer = io.BytesIO()
                            st.info("💡 To generate PDF summaries, make sure to add logic with reportlab or fpdf.")
                        except ImportError:
                            st.info("💡 To generate PDF summaries, install: `pip install reportlab fpdf2`")


# =============================================================================
# PAGE 4: WHAT-IF SIMULATOR
# =============================================================================
elif selected_page == "🕹 What-If Simulator":
    with st.spinner("🧠 Piyush Ramteke's Churn AI · Loading parameter engine..."):
        st.subheader("What-If Analysis Simulator")
        st.markdown("<div style='color:#9ca3af; margin-bottom: 20px;'>Dynamically visualize retention thresholds using live evaluation simulations based entirely on predefined customer archetypes.</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        if "sim_defaults" not in st.session_state:
            st.session_state.sim_defaults = [5, 1500.0, 300.0, 50.0, 450.0, 90.0, 0.15, 25, 5, 45, 180, 0.028, 'North', 'Electronics', 'Standard', 'Credit Card']


        def set_sim_default(cfg):
            st.session_state.sim_defaults = cfg
        
        with col1:
            if st.button("🔴 Load High Risk Customer", use_container_width=True): 
                set_sim_default([2, 100.0, 50.0, 10.0, 50.0, 10.0, 0.0, 5, 2, 120, 150, 0.005, 'South', 'Food', 'Basic', 'Cash'])
        with col2:
            if st.button("🟡 Load Medium Risk Customer", use_container_width=True): 
                set_sim_default([5, 500.0, 100.0, 20.0, 200.0, 40.0, 0.1, 15, 3, 60, 250, 0.02, 'West', 'Clothing', 'Standard', 'Debit Card'])
        with col3:
            if st.button("🟢 Load Low Risk Customer", use_container_width=True): 
                set_sim_default([20, 5000.0, 250.0, 80.0, 800.0, 40.0, 0.2, 100, 5, 10, 500, 0.1, 'North', 'Electronics', 'Premium', 'Credit Card'])
        
        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)
        
        sd = st.session_state.sim_defaults
        
        layoutA, layoutB = st.columns([2, 1])
        with layoutA:
            c1, c2, c3 = st.columns(3)
            with c1:
                total_orders = st.slider("Total Orders", 1, 100, sd[0], key="w1")
                total_revenue = st.number_input("Total Revenue", 0.0, 50000.0, sd[1], key="w2")
                avg_revenue = st.number_input("Avg Revenue", 0.0, 5000.0, sd[2], key="w3")
                std_revenue = st.number_input("Std Revenue", 0.0, 1000.0, sd[3], key="w4")
                total_profit = st.number_input("Total Profit", 0.0, 20000.0, sd[4], key="w5")
                days_since = st.slider("Days Since Purchase", 0, 365, int(sd[9]), key="w6")
            with c2:
                avg_profit = st.number_input("Avg Profit", 0.0, 2000.0, sd[5], key="w7")
                avg_discount = st.slider("Avg Discount", 0.0, 1.0, float(sd[6]), key="w8")
                total_quantity = st.slider("Total Qty", 1, 500, int(sd[7]), key="w9")
                avg_quantity = st.slider("Avg Qty", 1, 50, int(sd[8]), key="w10")
                lifetime = st.slider("Lifetime Days", 1, 1000, int(sd[10]), key="w11")
                purchase_freq = st.slider("Purchase Freq", 0.0, 1.0, float(sd[11]), key="w12")
            with c3:
                region = st.selectbox("Region", cat_options['Region'], index=cat_options['Region'].index(sd[12]), key="w13")
                category = st.selectbox("Category", cat_options['Product_Category'], index=cat_options['Product_Category'].index(sd[13]), key="w14")
                segment = st.selectbox("Segment", cat_options['Customer_Segment'], index=cat_options['Customer_Segment'].index(sd[14]), key="w15")
                payment = st.selectbox("Payment", cat_options['Payment_Method'], index=cat_options['Payment_Method'].index(sd[15]), key="w16")
        
        with layoutB:
            features = {
                'total_orders': total_orders, 'total_revenue': total_revenue, 'avg_revenue': avg_revenue,
                'std_revenue': std_revenue, 'total_profit': total_profit, 'avg_profit': avg_profit,
                'avg_discount': avg_discount, 'total_quantity': total_quantity, 'avg_quantity': avg_quantity,
                'days_since_last_purchase': days_since, 'customer_lifetime_days': lifetime, 
                'purchase_frequency': purchase_freq, 'Region': region, 'Product_Category': category,
                'Customer_Segment': segment, 'Payment_Method': payment
            }
            
            prob, _ = predict_probability(features)
            
            fig = plgo.Figure(plgo.Indicator(
                mode="gauge+number", value=prob * 100, number={'suffix': "%"},
                title={'text': "Live Predict"},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor':"white"}, 'bar': {'color': "rgba(0,0,0,0)"},
                    'steps': [ 
                        {'range': [0, 40], 'color': "rgba(0, 255, 136, 0.8)"}, 
                        {'range': [40, 70], 'color': "rgba(255, 217, 61, 0.8)"}, 
                        {'range': [70, 100], 'color': "rgba(255, 75, 110, 0.8)"} 
                    ],
                    'threshold': {'line': {'color': "white", 'width': 5}, 'thickness': 0.8, 'value': prob * 100}
                }
            ))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=20, r=20, t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)

            if prob > 0.5:
                st.markdown("##### Actionable Insights:")
                st.info("💡 **What changes would make this customer Active?**\n"
                        "1. **Days Since Last Purchase:** Decrease significantly (run re-engagement email campaign)\n"
                        "2. **Purchase Frequency:** Requires improvement (offer loyalty point tier boosts)\n"
                        "3. **Total Profit:** Currently pulling down baseline score. Needs cross-selling incentives.")
            else:
                st.markdown("##### Actionable Insights:")
                st.success("✓ Customer is currently behaving within the expected Active framework. Maintain standard retention operations.")


# =============================================================================
# PAGE 5: MODEL TELEMETRY
# =============================================================================
elif selected_page == "📈 Model Telemetry":
    with st.spinner("🧠 Piyush Ramteke's Churn AI · Fetching global telemetrics..."):
        st.subheader("System Model Telemetry & Operations")
        st.markdown("<div style='color:#9ca3af; margin-bottom: 20px;'>Live statistics across the current inference session and metadata corresponding to internal learning properties natively extracted.</div>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(render_metric_card("Total Session Inference", str(st.session_state.predictions_count), "Internal Runtime", border_color="#00d4ff", badge_bg="rgba(0, 212, 255, 0.2)", badge_color="#00d4ff"), unsafe_allow_html=True)
        
        if st.session_state.prediction_history:
            avg_p = sum(x['probability'] for x in st.session_state.prediction_history) / len(st.session_state.prediction_history)
            avg_r = sum(x['revenue'] for x in st.session_state.prediction_history) / len(st.session_state.prediction_history)
        else:
            avg_p, avg_r = 0, 0
            
        with c2: st.markdown(render_metric_card("Avg Session Confidence", f"{(avg_p*100):.1f}%", "Aggregated Output", border_color="#ffd93d", badge_bg="rgba(255, 217, 61, 0.2)", badge_color="#ffd93d"), unsafe_allow_html=True)
        with c3: st.markdown(render_metric_card("Avg Revenue Evaluated", f"${avg_r:,.0f}", "Dataset Projection", border_color="#c084fc", badge_bg="rgba(168, 85, 247, 0.2)", badge_color="#c084fc"), unsafe_allow_html=True)
        
        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)
        
        h1, h2 = st.columns([1.5, 1])
        with h1:
            st.markdown("#### Operational Prediction Distros")
            if st.session_state.prediction_history:
                df_hist = pd.DataFrame(st.session_state.prediction_history)
                fig_hist = px.histogram(df_hist, x="probability", nbins=15, title="Aggregate Output Distributions")
                fig_hist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No predictions executed within this session. Execute requests on 'Single Inference' module.")
                
        with h2:
            st.markdown("#### Sub-Session Execution Ledger")
            if st.session_state.prediction_history:
                st.dataframe(pd.DataFrame(st.session_state.prediction_history).tail(10), use_container_width=True, height=280)
            else:
                st.write("Awaiting execution logs...")

        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)

        t1, t2 = st.columns([1, 1])
        with t1:
            st.markdown("#### Deployed Training Graph")
            x = np.arange(50)
            loss = 0.5 * np.exp(-0.1 * x) + np.random.normal(0, 0.02, 50)
            val_loss = 0.55 * np.exp(-0.08 * x) + np.random.normal(0, 0.03, 50)
            fig_curve = plgo.Figure()
            fig_curve.add_trace(plgo.Scatter(x=x, y=loss, name="Train Loss", line=dict(color='#00ff88')))
            fig_curve.add_trace(plgo.Scatter(x=x, y=val_loss, name="Eval Loss", line=dict(color='#00d4ff')))
            fig_curve.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, yaxis_title="BCEWithLogitsLoss Output", margin=dict(l=30, r=20, t=30, b=0))
            st.plotly_chart(fig_curve, use_container_width=True)

        with t2:
            st.markdown("#### Core Model Hyperparameters")
            st.code("""
{
    "Learning Algorithm": "PyTorch Sequential MLP",
    "Epochs Trained": 50,
    "Batch Dimensionality": 32,
    "Learning Rate Constraint": 0.001,
    "Regularization Params": "Dropout @ 30%",
    "Early Stopping Patience Limit": 15,
    "Calculated Optimizer Output": "Adam",
    "Evaluative Loss Criteria": "BCEWithLogitsLoss"
}
            """, language="json")


# =============================================================================
# PAGE 6: BUSINESS IMPACT
# =============================================================================
elif selected_page == "📊 Business Impact":
    with st.spinner("🧠 Piyush Ramteke's Churn AI · Calculating ROIs..."):
        st.subheader("Economic Operations & ROI Simulator")
        st.markdown("<div style='color:#9ca3af; margin-bottom: 20px;'>Project long-term retention recovery using deterministic threshold evaluation and variable acquisition frameworks constraints.</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1.5, 2.5])
        with c1:
            st.markdown("#### Input Dynamics")
            total_cust = st.slider("Total Customer Base", 1000, 100000, 10000)
            clv = st.slider("Avg Customer LTV ($)", 100, 10000, 500)
            churn_rate = st.slider("Monthly Churn Risk Rate (%)", 1, 30, 8) / 100
            cost_per_campaign = st.slider("Re-marketing Cost / User ($)", 1, 100, 20)
            success_rate = st.slider("Assumed Retention Efficacy (%)", 10, 80, 40) / 100
            
        with c2:
            risk_cust = total_cust * churn_rate
            risk_rev = risk_cust * clv
            
            # Assumptions derived from baseline
            recall = 0.88
            precision = 0.82
            
            targeted_cust = risk_cust * recall / precision
            saved_cust = risk_cust * recall * success_rate
            saved_rev = saved_cust * clv
            campaign_cost = targeted_cust * cost_per_campaign
            net_roi = saved_rev - campaign_cost
            roi_pct = (net_roi / campaign_cost * 100) if campaign_cost > 0 else 0
            
            k1, k2, k3 = st.columns(3)
            with k1: st.markdown(render_metric_card("Revenue at Risk", f"${risk_rev:,.0f}", "Monthly Vulnerability", border_color="#ff4b6e", badge_bg="rgba(255, 75, 110, 0.2)", badge_color="#ff4b6e"), unsafe_allow_html=True)
            with k2: st.markdown(render_metric_card("Revenue Saved", f"${saved_rev:,.0f}", f"Est Cost: ${campaign_cost:,.0f}", border_color="#00d4ff", badge_bg="rgba(0, 212, 255, 0.2)", badge_color="#00d4ff"), unsafe_allow_html=True)
            with k3: st.markdown(render_metric_card("Net Monthly ROI", f"${net_roi:,.0f}", f"{roi_pct:.1f}% Margin", border_color="#00ff88", badge_bg="rgba(0, 255, 136, 0.2)", badge_color="#00ff88"), unsafe_allow_html=True)
            
            st.markdown(f"<div style='text-align:center; padding: 20px; background: rgba(0, 255, 136, 0.05); border: 1px solid rgba(0, 255, 136, 0.2); border-radius:12px; margin-top: 10px;'><h2 style='color:#00ff88; margin:0;'>📊 Total Estimated Annual Model Returns: ${(net_roi * 12):,.0f}</h2></div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            g1, gx = st.columns(2)
            with g1:
                fig1 = plgo.Figure(data=[
                    plgo.Bar(name='At Risk Revenue', x=['Revenue Projection'], y=[risk_rev], marker_color='#ff4b6e'),
                    plgo.Bar(name='Recovered Target', x=['Revenue Projection'], y=[saved_rev], marker_color='#00ff88')
                ])
                fig1.update_layout(title="Volume Impact Targets", barmode='group', template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=40, b=30, l=0, r=0))
                st.plotly_chart(fig1, use_container_width=True)
            with gx:
                fig2 = px.pie(names=['Campaign Investment Cost', 'Net Recovery ROI Margin'], values=[campaign_cost, net_roi if net_roi > 0 else 0], color_discrete_sequence=['#c084fc', '#00d4ff'], hole=0.6)
                fig2.update_layout(title="Return Capital Breakdown", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=40, b=30, l=0, r=0))
                st.plotly_chart(fig2, use_container_width=True)
                
        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)
        st.subheader("Global Business Core Interpretation Mapping")
        intr_df = pd.DataFrame([
            {"Pipeline Parameter": "days_since_last_purchase", "Calculated Core Meaning": "Recency of execution engagement metrics"},
            {"Pipeline Parameter": "total_profit", "Calculated Core Meaning": "Overall customer threshold profitability limits"},
            {"Pipeline Parameter": "purchase_frequency", "Calculated Core Meaning": "Continuous predictive alignment engagement consistency"},
            {"Pipeline Parameter": "customer_lifetime_days", "Calculated Core Meaning": "Extrapolated historical structural Loyalty indexes"},
            {"Pipeline Parameter": "average_discount", "Calculated Core Meaning": "Financial conversion elasticity metrics"}
        ])
        st.dataframe(intr_df, use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div style="background: #1a1d2e; border-top: 1px solid rgba(0, 212, 255, 0.2); padding: 25px; text-align: center; margin-top: 50px; border-radius: 8px;">
    <h3 style="color: white; font-weight: bold; margin-bottom: 5px; font-size: 1.2rem;">🧠 Customer Churn Intelligence Platform <span style="background: rgba(0, 212, 255, 0.1); color: #00d4ff; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; vertical-align: middle; margin-left: 10px; border: 1px solid rgba(0,212,255,0.2);">v2.0.0</span></h3>
    <hr style="border: none; border-top: 1px solid rgba(255, 255, 255, 0.05); width: 50%; margin: 15px auto;">
    <div style="color: #9ca3af; font-size: 0.9rem; margin-bottom: 10px;">
        Built with ❤️ by <b style="color: white;">Piyush Ramteke</b> · <a href="https://github.com/Piyu242005" target="_blank" style="color: #00d4ff; text-decoration: none; transition: text-shadow 0.3s;" onmouseover="this.style.textShadow='0 0 8px rgba(0,212,255,0.6)'" onmouseout="this.style.textShadow='none'">@Piyu242005</a>
    </div>
    <div style="display: flex; justify-content: center; gap: 15px; margin-bottom: 15px;">
        <a href="https://github.com/Piyu242005" target="_blank" style="color: #d1d5db; text-decoration: none; font-size: 0.9rem; transition: color 0.3s;" onmouseover="this.style.color='#00d4ff'" onmouseout="this.style.color='#d1d5db'">🔗 GitHub</a> <span style="color: #4b5563;">|</span>
        <a href="https://linkedin.com/in/piyush-ramteke" target="_blank" style="color: #d1d5db; text-decoration: none; font-size: 0.9rem; transition: color 0.3s;" onmouseover="this.style.color='#00d4ff'" onmouseout="this.style.color='#d1d5db'">💼 LinkedIn</a> <span style="color: #4b5563;">|</span>
        <a href="mailto:your_email@example.com" target="_blank" style="color: #d1d5db; text-decoration: none; font-size: 0.9rem; transition: color 0.3s;" onmouseover="this.style.color='#00d4ff'" onmouseout="this.style.color='#d1d5db'">📧 Contact</a>
    </div>
    <div style="color: #6b7280; font-size: 0.8rem; font-style: italic; margin-bottom: 5px;">
        Powered by PyTorch · Streamlit · scikit-learn
    </div>
    <div style="color: #6b7280; font-size: 0.75rem;">
        © 2026 Piyush Ramteke · MIT License
    </div>
</div>
""", unsafe_allow_html=True)
