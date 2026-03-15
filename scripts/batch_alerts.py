import os
import sys
import argparse
import logging
import json
import sqlite3
import pandas as pd
import numpy as np
import torch
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ChurnAlerts')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

try:
    from model import MLPClassifier
except ImportError:
    logger.error("Could not import model. Make sure you are running from the project root.")
    sys.exit(1)

def load_inference_dependencies():
    """Load the trained model and scaler"""
    try:
        # Load Model
        logger.info("Loading PyTorch model from artifacts...")
        checkpoint = torch.load('artifacts/mlp_churn_classifier_final.pth', map_location='cpu', weights_only=False)
        model = MLPClassifier(input_dim=16, hidden_dims=[128, 64, 32], dropout_rate=0.3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load Scaler
        logger.info("Loading Scaler from artifacts...")
        scaler = joblib.load('artifacts/scaler.pkl')

        return model, scaler
    except Exception as e:
        logger.error(f"Failed to load dependencies: {e}")
        return None, None

def send_alert(customer_id, risk_score, channels):
    """Simulate sending alerts via webhook/email"""
    message = f"🚨 HIGH CHURN RISK ALERT 🚨\nCustomer {customer_id} has a {risk_score*100:.1f}% probability of churning."
    
    if 'slack' in channels:
        logger.info(f"[SLACK WEBHOOK TRIGGERED]: {message}")
        # In production: requests.post(SLACK_WEBHOOK_URL, json={"text": message})
        
    if 'email' in channels:
        logger.info(f"[EMAIL TRIGGERED]: Sent retention workflow email to account manager for {customer_id}")
        # In production: email_client.send_email(to="acct_manager@co.com", subject="Risk Alert", body=message)

def run_batch_inference_and_alert(threshold=0.75, limit=10, channels=['slack']):
    """
    Simulate reading new daily active users from the database,
    scoring them, and sending alerts for high-risk users.
    """
    logger.info("Starting Daily Batch Inference Job...")
    
    # 1. Load ML dependencies
    model, scaler = load_inference_dependencies()
    if not model or not scaler:
        return
        
    # 2. Extract "New" Data (Mocking by grabbing random users from DB)
    db_path = 'data/churn_data.db'
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}")
        return
        
    try:
        logger.info(f"Connecting to database to extract up to {limit} unscored users...")
        conn = sqlite3.connect(db_path)
        # We'll just grab some random users for the simulation
        query = f"SELECT * FROM transactions ORDER BY RANDOM() LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        logger.error(f"Database error: {e}")
        return

    if df.empty:
        logger.info("No new users to score.")
        return

    # 3. Quick internal feature engineering (Matching basic structure)
    # Note: In production, we would use the exact same feature engineering pipeline module
    # For this script we will use dummy numeric values that match the 16 dims just for the alert demo
    logger.info(f"Extracted {len(df)} users for scoring.")
    
    alerts_sent = 0
    
    # 4. Predict & Alert loop
    for _, row in df.iterrows():
        customer_id = row['Customer_ID']
        
        # Create a mock feature array of 16 dimensions
        # This simulates the processed output of our standard pipeline
        mock_features = np.random.rand(1, 16).astype(np.float32)
        
        # Scale
        scaled_features = scaler.transform(mock_features)
        
        # Predict
        tensor_features = torch.FloatTensor(scaled_features)
        with torch.no_grad():
            risk_score = model(tensor_features).item()
            
        # 5. Threshold logic
        if risk_score >= threshold:
            send_alert(customer_id, risk_score, channels)
            alerts_sent += 1
            
    logger.info(f"Batch job complete. Evaluated {len(df)} users. Sent {alerts_sent} alerts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Inference & Alerting Script")
    parser.add_argument('--threshold', type=float, default=0.70, help='Churn probability threshold to trigger alert')
    parser.add_argument('--limit', type=int, default=50, help='Number of customers to score in this batch')
    parser.add_argument('--channels', nargs='+', default=['slack', 'email'], help='Alert channels (slack, email)')
    
    args = parser.parse_args()
    
    run_batch_inference_and_alert(
        threshold=args.threshold,
        limit=args.limit,
        channels=args.channels
    )
