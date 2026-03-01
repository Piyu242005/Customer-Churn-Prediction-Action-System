"""
Flask API for MLP Churn Classifier
REST API for serving churn predictions

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
import logging

from model import MLPClassifier

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and preprocessor
model = None
scaler = None
feature_names = None
label_encoders = None


def load_model_and_preprocessor():
    """Load trained model and preprocessing artifacts"""
    global model, scaler, feature_names, label_encoders
    
    try:
        # Load model
        logger.info("Loading model...")
        checkpoint = torch.load('mlp_churn_classifier_final.pth', map_location='cpu', weights_only=False)
        model = MLPClassifier(input_dim=16, hidden_dims=[128, 64, 32], dropout_rate=0.3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("✓ Model loaded successfully")
        
        # Load preprocessor artifacts
        logger.info("Loading preprocessing artifacts...")
        try:
            scaler = joblib.load('scaler.pkl')
            feature_names = joblib.load('feature_names.pkl')
            label_encoders = joblib.load('label_encoders.pkl')
            logger.info("✓ Preprocessing artifacts loaded")
        except FileNotFoundError:
            logger.warning("⚠ Preprocessing artifacts not found. Will use default feature names.")
            feature_names = [
                'total_orders', 'total_revenue', 'avg_revenue', 'std_revenue',
                'total_profit', 'avg_profit', 'avg_discount', 'total_quantity',
                'avg_quantity', 'days_since_last_purchase', 'customer_lifetime_days',
                'purchase_frequency', 'Region_encoded', 'Product_Category_encoded',
                'Customer_Segment_encoded', 'Payment_Method_encoded'
            ]
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MLP Churn Classifier API</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 40px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                padding: 30px;
                border-radius: 10px;
                color: #333;
                box-shadow: 0 8px 20px rgba(0,0,0,0.2);
            }
            h1 { color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; }
            h2 { color: #764ba2; margin-top: 30px; }
            .endpoint { 
                background: #f8f9fa; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px;
                border-left: 4px solid #667eea;
            }
            .method { 
                display: inline-block;
                background: #667eea; 
                color: white; 
                padding: 5px 10px; 
                border-radius: 3px;
                font-weight: bold;
                margin-right: 10px;
            }
            code { 
                background: #e9ecef; 
                padding: 2px 6px; 
                border-radius: 3px;
                font-family: monospace;
            }
            pre {
                background: #2d2d2d;
                color: #f8f8f2;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }
            .badge {
                display: inline-block;
                background: #27ae60;
                color: white;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 MLP Churn Classifier API</h1>
            <p><span class="badge">ONLINE</span> Neural Network-based Customer Churn Prediction Service</p>
            
            <h2>📡 API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/health</code>
                <p>Check API health status</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/predict</code>
                <p>Predict churn probability for a single customer</p>
                <p><strong>Request Body:</strong></p>
                <pre>{
  "features": {
    "total_orders": 15,
    "total_revenue": 2500.0,
    "avg_revenue": 166.67,
    "days_since_last_purchase": 45,
    ...
  }
}</pre>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/predict_batch</code>
                <p>Predict churn for multiple customers</p>
                <p><strong>Request Body:</strong></p>
                <pre>{
  "customers": [
    {"features": {...}},
    {"features": {...}}
  ]
}</pre>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/model/info</code>
                <p>Get model architecture and metadata</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/features</code>
                <p>Get list of required features</p>
            </div>
            
            <h2>📚 Usage Example</h2>
            <pre>
import requests

url = "http://localhost:5000/predict"
data = {
    "features": {
        "total_orders": 15,
        "total_revenue": 2500.0,
        "avg_revenue": 166.67,
        "std_revenue": 45.2,
        "total_profit": 625.0,
        "avg_profit": 41.67,
        "avg_discount": 0.05,
        "total_quantity": 30,
        "avg_quantity": 2.0,
        "days_since_last_purchase": 45,
        "customer_lifetime_days": 365,
        "purchase_frequency": 0.041,
        "Region_encoded": 2,
        "Product_Category_encoded": 1,
        "Customer_Segment_encoded": 0,
        "Payment_Method_encoded": 1
    }
}

response = requests.post(url, json=data)
print(response.json())
            </pre>
            
            <p style="margin-top: 30px; text-align: center; color: #999;">
                Powered by PyTorch & Flask | © 2026 Piyush Ramteke
            </p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'service': 'MLP Churn Classifier API',
        'version': '1.0.0'
    }
    return jsonify(status), 200


@app.route('/features', methods=['GET'])
def get_features():
    """Get list of required features"""
    if feature_names is None:
        return jsonify({'error': 'Feature names not loaded'}), 500
    
    return jsonify({
        'features': feature_names,
        'count': len(feature_names),
        'description': 'List of required input features for prediction'
    }), 200


@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = model.get_model_info()
    
    return jsonify({
        'architecture': 'Multilayer Perceptron (MLP)',
        'framework': 'PyTorch',
        'model_details': info,
        'input_features': len(feature_names) if feature_names else 16,
        'output': 'Binary classification (churn probability)'
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict churn probability for a single customer
    
    Request JSON format:
    {
        "features": {
            "total_orders": 15,
            "total_revenue": 2500.0,
            ...
        }
    }
    """
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request'}), 400
        
        features_dict = data['features']
        
        # Validate features
        if feature_names:
            missing_features = [f for f in feature_names if f not in features_dict]
            if missing_features:
                return jsonify({
                    'error': 'Missing features',
                    'missing': missing_features
                }), 400
        
        # Prepare feature vector
        feature_values = [features_dict[f] for f in feature_names]
        X = np.array([feature_values], dtype=np.float32)
        
        # Scale features if scaler is available
        if scaler:
            X = scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Predict
        with torch.no_grad():
            model.eval()
            prediction_proba = model(X_tensor).item()
            prediction_class = 1 if prediction_proba >= 0.5 else 0
        
        # Prepare response
        response = {
            'churn_probability': round(prediction_proba, 4),
            'churn_prediction': bool(prediction_class),
            'prediction_label': 'Churned' if prediction_class == 1 else 'Active',
            'confidence': round(max(prediction_proba, 1 - prediction_proba), 4),
            'risk_level': get_risk_level(prediction_proba),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: {response['prediction_label']} ({response['churn_probability']:.4f})")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict churn for multiple customers
    
    Request JSON format:
    {
        "customers": [
            {"customer_id": "C001", "features": {...}},
            {"customer_id": "C002", "features": {...}}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if 'customers' not in data:
            return jsonify({'error': 'Missing "customers" in request'}), 400
        
        customers = data['customers']
        predictions = []
        
        for customer in customers:
            customer_id = customer.get('customer_id', 'Unknown')
            features_dict = customer.get('features', {})
            
            # Prepare feature vector
            feature_values = [features_dict.get(f, 0) for f in feature_names]
            X = np.array([feature_values], dtype=np.float32)
            
            # Scale features
            if scaler:
                X = scaler.transform(X)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X)
            
            # Predict
            with torch.no_grad():
                model.eval()
                prediction_proba = model(X_tensor).item()
                prediction_class = 1 if prediction_proba >= 0.5 else 0
            
            predictions.append({
                'customer_id': customer_id,
                'churn_probability': round(prediction_proba, 4),
                'churn_prediction': bool(prediction_class),
                'prediction_label': 'Churned' if prediction_class == 1 else 'Active',
                'risk_level': get_risk_level(prediction_proba)
            })
        
        response = {
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch prediction made for {len(predictions)} customers")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/explain', methods=['POST'])
def explain_prediction():
    """
    Explain a prediction (simplified version)
    
    Request JSON format:
    {
        "features": {...}
    }
    """
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request'}), 400
        
        features_dict = data['features']
        
        # Make prediction first
        feature_values = [features_dict[f] for f in feature_names]
        X = np.array([feature_values], dtype=np.float32)
        
        if scaler:
            X = scaler.transform(X)
        
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            model.eval()
            prediction_proba = model(X_tensor).item()
            prediction_class = 1 if prediction_proba >= 0.5 else 0
        
        # Simple feature importance (gradient-based)
        X_tensor.requires_grad = True
        output = model(X_tensor)
        output.backward()
        
        gradients = X_tensor.grad.detach().numpy().flatten()
        feature_importance = np.abs(gradients)
        
        # Get top features
        top_indices = np.argsort(feature_importance)[-5:][::-1]
        top_features = [
            {
                'feature': feature_names[i],
                'value': float(feature_values[i]),
                'importance': float(feature_importance[i]),
                'impact': 'Increases churn risk' if gradients[i] > 0 else 'Decreases churn risk'
            }
            for i in top_indices
        ]
        
        response = {
            'prediction': {
                'churn_probability': round(prediction_proba, 4),
                'prediction_label': 'Churned' if prediction_class == 1 else 'Active',
                'risk_level': get_risk_level(prediction_proba)
            },
            'explanation': {
                'top_contributing_features': top_features,
                'method': 'Gradient-based feature importance'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_statistics():
    """Get API usage statistics (placeholder)"""
    stats = {
        'api_version': '1.0.0',
        'model_type': 'MLP Neural Network',
        'framework': 'PyTorch',
        'status': 'operational',
        'uptime': 'N/A',
        'total_predictions': 'N/A'
    }
    return jsonify(stats), 200


def get_risk_level(probability):
    """
    Categorize churn risk level
    
    Args:
        probability: Churn probability
        
    Returns:
        str: Risk level
    """
    if probability < 0.3:
        return 'Low Risk'
    elif probability < 0.6:
        return 'Medium Risk'
    elif probability < 0.8:
        return 'High Risk'
    else:
        return 'Very High Risk'


def preprocess_and_save_artifacts():
    """
    Preprocess data and save preprocessing artifacts for API use
    """
    from data_preprocessing import load_and_preprocess_data
    import joblib
    
    print("Preprocessing data and saving artifacts...")
    
    data_path = "Business_Analytics_Dataset_10000_Rows.csv"
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_path)
    
    # Save preprocessing artifacts
    joblib.dump(preprocessor.scaler, 'scaler.pkl')
    joblib.dump(preprocessor.get_feature_names(), 'feature_names.pkl')
    joblib.dump(preprocessor.label_encoders, 'label_encoders.pkl')
    
    print("✓ Preprocessing artifacts saved:")
    print("  - scaler.pkl")
    print("  - feature_names.pkl")
    print("  - label_encoders.pkl")


if __name__ == '__main__':
    import sys
    
    # Check if artifacts exist, if not create them
    try:
        joblib.load('scaler.pkl')
    except:
        print("Preprocessing artifacts not found. Creating them...")
        preprocess_and_save_artifacts()
    
    # Load model and preprocessor
    if load_model_and_preprocessor():
        logger.info("="*60)
        logger.info("🚀 MLP CHURN CLASSIFIER API")
        logger.info("="*60)
        logger.info("Starting Flask server...")
        logger.info("API will be available at: http://localhost:5000")
        logger.info("="*60)
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to load model. Exiting.")
        sys.exit(1)
