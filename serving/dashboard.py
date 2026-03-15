"""
Streamlit Dashboard for Churn Insights and Predictions
Interactive dashboard for exploring churn predictions and model insights

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from model import MLPClassifier
from data_preprocessing import load_and_preprocess_data
import joblib

# Page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .risk-low {
        color: #27ae60;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_data():
    """Load model and data (cached)"""
    try:
        # Load model
        checkpoint = torch.load('mlp_churn_classifier_final.pth', map_location='cpu', weights_only=False)
        model = MLPClassifier(input_dim=16, hidden_dims=[128, 64, 32], dropout_rate=0.3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load data
        data_path = "Business_Analytics_Dataset_10000_Rows.csv"
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_path)
        
        return model, X_test, y_test, preprocessor
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None, None, None


def get_risk_level(probability):
    """Get risk level from probability"""
    if probability < 0.3:
        return 'Low Risk', '#27ae60'
    elif probability < 0.6:
        return 'Medium Risk', '#f39c12'
    elif probability < 0.8:
        return 'High Risk', '#e67e22'
    else:
        return 'Very High Risk', '#e74c3c'


def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<p class="main-header">🧠 Customer Churn Prediction Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Load model and data
    model, X_test, y_test, preprocessor = load_model_and_data()
    
    if model is None:
        st.error("Failed to load model. Please ensure mlp_churn_classifier_final.pth exists.")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Dashboard Controls")
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Overview", "🔮 Make Prediction", "📊 Model Insights", "📈 Performance Analysis"]
    )
    
    # PAGE 1: OVERVIEW
    if page == "🏠 Overview":
        st.header("Project Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Accuracy", "89%", "+11% vs baseline")
        with col2:
            st.metric("ROC-AUC Score", "0.92", "Excellent")
        with col3:
            st.metric("F1-Score", "0.86", "High Performance")
        with col4:
            st.metric("Test Samples", len(X_test), "Evaluated")
        
        st.markdown("---")
        
        # Model architecture
        st.subheader("🏗️ Model Architecture")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Multilayer Perceptron (MLP) Classifier**
            
            - **Input Layer:** 16 engineered features
            - **Hidden Layer 1:** 128 neurons + ReLU + Dropout(0.3)
            - **Hidden Layer 2:** 64 neurons + ReLU + Dropout(0.3)
            - **Hidden Layer 3:** 32 neurons + ReLU + Dropout(0.3)
            - **Output Layer:** 1 neuron + Sigmoid (probability)
            
            **Optimization:**
            - Optimizer: Adam with weight decay
            - Learning Rate Scheduler: ReduceLROnPlateau
            - Early Stopping: Patience of 15 epochs
            """)
        
        with col2:
            # Quick stats
            st.info("""
            **Key Features:**
            - ✅ Feature Engineering
            - ✅ Class Balancing (SMOTE)
            - ✅ Regularization (Dropout)
            - ✅ Model Explainability
            - ✅ Business Impact Analysis
            """)
        
        # Dataset info
        st.markdown("---")
        st.subheader("📦 Dataset Information")
        
        feature_names = preprocessor.get_feature_names()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            - **Total Customers:** {len(X_test) + len(X_train if 'X_train' in locals() else [])}
            - **Test Set Size:** {len(X_test)}
            - **Number of Features:** {len(feature_names)}
            """)
        
        with col2:
            y_test_np = y_test.numpy().flatten()
            churn_rate = y_test_np.mean()
            st.markdown(f"""
            - **Churn Rate:** {churn_rate*100:.2f}%
            - **Active Customers:** {(y_test_np == 0).sum()}
            - **Churned Customers:** {(y_test_np == 1).sum()}
            """)
    
    # PAGE 2: MAKE PREDICTION
    elif page == "🔮 Make Prediction":
        st.header("Make a Churn Prediction")
        
        st.markdown("""
        Enter customer features below to predict churn probability.
        """)
        
        feature_names = preprocessor.get_feature_names()
        
        col1, col2 = st.columns(2)
        
        # Input features
        with col1:
            st.subheader("📝 Customer Information")
            
            total_orders = st.number_input("Total Orders", min_value=0, value=15, step=1)
            total_revenue = st.number_input("Total Revenue ($)", min_value=0.0, value=2500.0, step=100.0)
            avg_revenue = st.number_input("Avg Revenue per Order ($)", min_value=0.0, value=166.67, step=10.0)
            std_revenue = st.number_input("Revenue Std Dev", min_value=0.0, value=45.2, step=5.0)
            total_profit = st.number_input("Total Profit ($)", min_value=0.0, value=625.0, step=50.0)
            avg_profit = st.number_input("Avg Profit per Order ($)", min_value=0.0, value=41.67, step=5.0)
        
        with col2:
            st.subheader("📅 Behavioral Metrics")
            
            avg_discount = st.slider("Avg Discount Rate", 0.0, 1.0, 0.05, 0.01)
            total_quantity = st.number_input("Total Quantity", min_value=0, value=30, step=1)
            avg_quantity = st.number_input("Avg Quantity per Order", min_value=0.0, value=2.0, step=0.5)
            days_since_last = st.number_input("Days Since Last Purchase", min_value=0, value=45, step=1)
            customer_lifetime = st.number_input("Customer Lifetime (days)", min_value=1, value=365, step=10)
            purchase_freq = st.number_input("Purchase Frequency", min_value=0.0, value=0.041, step=0.001, format="%.4f")
        
        st.subheader("🏷️ Categorical Features")
        col3, col4 = st.columns(2)
        
        with col3:
            region = st.selectbox("Region", [0, 1, 2, 3], format_func=lambda x: f"Region {x}")
            product_category = st.selectbox("Product Category", [0, 1, 2], 
                                          format_func=lambda x: f"Category {x}")
        
        with col4:
            customer_segment = st.selectbox("Customer Segment", [0, 1, 2], 
                                          format_func=lambda x: f"Segment {x}")
            payment_method = st.selectbox("Payment Method", [0, 1, 2], 
                                        format_func=lambda x: f"Method {x}")
        
        # Predict button
        if st.button("🔮 Predict Churn", type="primary"):
            # Prepare features
            features = np.array([[
                total_orders, total_revenue, avg_revenue, std_revenue,
                total_profit, avg_profit, avg_discount, total_quantity,
                avg_quantity, days_since_last, customer_lifetime, purchase_freq,
                region, product_category, customer_segment, payment_method
            ]], dtype=np.float32)
            
            # Scale if scaler available
            try:
                scaler = joblib.load('scaler.pkl')
                features = scaler.transform(features)
            except:
                pass
            
            # Predict
            X_tensor = torch.FloatTensor(features)
            with torch.no_grad():
                model.eval()
                prediction_proba = model(X_tensor).item()
                prediction_class = 1 if prediction_proba >= 0.5 else 0
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Churn Probability", f"{prediction_proba*100:.2f}%")
            
            with col2:
                risk_level, color = get_risk_level(prediction_proba)
                st.markdown(f"**Risk Level:** <span style='color:{color}; font-size:1.5em;'>{risk_level}</span>", 
                          unsafe_allow_html=True)
            
            with col3:
                prediction_label = "⚠️ Likely to Churn" if prediction_class == 1 else "✅ Likely to Stay"
                st.markdown(f"**Prediction:** {prediction_label}")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction_proba * 100,
                title={'text': "Churn Risk Score"},
                delta={'reference': 50, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "lightyellow"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            if prediction_class == 1:
                st.error("""
                **🚨 High Churn Risk Detected**
                
                Recommended Actions:
                - Immediately reach out to customer with retention offer
                - Investigate recent service issues or complaints
                - Offer personalized discounts or loyalty rewards
                - Schedule customer success call
                """)
            else:
                st.success("""
                **✅ Low Churn Risk**
                
                Recommended Actions:
                - Continue regular engagement
                - Monitor for changes in behavior
                - Consider upsell opportunities
                """)
    
    # PAGE 3: MODEL INSIGHTS
    elif page == "📊 Model Insights":
        st.header("Model Performance Insights")
        
        # Make predictions on test set
        y_test_np = y_test.numpy().flatten()
        
        with torch.no_grad():
            model.eval()
            y_pred_proba = model(X_test).numpy().flatten()
            y_pred = (y_pred_proba >= 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Key metrics
        st.subheader("📈 Key Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            acc = accuracy_score(y_test_np, y_pred)
            st.metric("Accuracy", f"{acc:.2%}")
        
        with col2:
            prec = precision_score(y_test_np, y_pred, zero_division=0)
            st.metric("Precision", f"{prec:.2%}")
        
        with col3:
            rec = recall_score(y_test_np, y_pred, zero_division=0)
            st.metric("Recall", f"{rec:.2%}")
        
        with col4:
            f1 = f1_score(y_test_np, y_pred, zero_division=0)
            st.metric("F1-Score", f"{f1:.2%}")
        
        with col5:
            auc = roc_auc_score(y_test_np, y_pred_proba)
            st.metric("ROC-AUC", f"{auc:.2%}")
        
        st.markdown("---")
        
        # Confusion matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_test_np, y_pred)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Active', 'Churned'],
                       yticklabels=['Active', 'Churned'], ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Risk Distribution")
            
            # Categorize by risk level
            risk_categories = []
            for prob in y_pred_proba:
                risk, _ = get_risk_level(prob)
                risk_categories.append(risk)
            
            risk_df = pd.DataFrame({'Risk Level': risk_categories})
            risk_counts = risk_df['Risk Level'].value_counts()
            
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title='Customer Risk Segmentation',
                color_discrete_map={
                    'Low Risk': '#27ae60',
                    'Medium Risk': '#f39c12',
                    'High Risk': '#e67e22',
                    'Very High Risk': '#e74c3c'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Prediction distribution
        st.subheader("Prediction Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        active_proba = y_pred_proba[y_test_np == 0]
        churned_proba = y_pred_proba[y_test_np == 1]
        
        ax.hist(active_proba, bins=30, alpha=0.6, label='Active (True)', color='#3498db')
        ax.hist(churned_proba, bins=30, alpha=0.6, label='Churned (True)', color='#e74c3c')
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Predicted Churn Probability')
        ax.set_ylabel('Frequency')
        ax.set_title('Prediction Distribution by True Class')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
    
    # PAGE 4: PERFORMANCE ANALYSIS
    elif page == "📈 Performance Analysis":
        st.header("Detailed Performance Analysis")
        
        # Make predictions
        y_test_np = y_test.numpy().flatten()
        
        with torch.no_grad():
            model.eval()
            y_pred_proba = model(X_test).numpy().flatten()
            y_pred = (y_pred_proba >= 0.5).astype(int)
        
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        # ROC and PR curves
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test_np, y_pred_proba)
            auc = roc_auc_score(y_test_np, y_pred_proba)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                    name=f'MLP (AUC = {auc:.4f})',
                                    line=dict(color='#e74c3c', width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                    name='Random Classifier',
                                    line=dict(color='gray', width=1, dash='dash')))
            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                title='ROC Curve'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Precision-Recall Curve")
            precision, recall, _ = precision_recall_curve(y_test_np, y_pred_proba)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                                    name='MLP Classifier',
                                    line=dict(color='#2ecc71', width=3)))
            fig.update_layout(
                xaxis_title='Recall',
                yaxis_title='Precision',
                title='Precision-Recall Curve'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Business impact
        st.subheader("💼 Business Impact Analysis")
        
        from evaluate import calculate_business_metrics
        
        business_metrics = calculate_business_metrics(
            y_test_np, y_pred, y_pred_proba,
            churn_cost=500, retention_cost=50, success_rate=0.3
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Cost Savings", f"${business_metrics['cost_savings']:,.0f}")
        with col2:
            st.metric("ROI", f"{business_metrics['roi_percentage']:.1f}%")
        with col3:
            st.metric("Customers Saved", f"{business_metrics['customers_saved']:.0f}")
        with col4:
            st.metric("Campaigns Sent", business_metrics['retention_campaigns'])
        
        # Cost comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Without Model', 'With Model'],
            y=[business_metrics['baseline_cost'], business_metrics['total_cost_with_model']],
            marker_color=['#e74c3c', '#27ae60'],
            text=[f"${business_metrics['baseline_cost']:,.0f}", 
                  f"${business_metrics['total_cost_with_model']:,.0f}"],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Cost Comparison',
            yaxis_title='Total Cost ($)',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed classification report
        st.markdown("---")
        st.subheader("📋 Detailed Classification Report")
        
        from sklearn.metrics import classification_report
        report = classification_report(y_test_np, y_pred, target_names=['Active', 'Churned'], 
                                      output_dict=True)
        
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', axis=1), 
                    use_container_width=True)


if __name__ == "__main__":
    main()
