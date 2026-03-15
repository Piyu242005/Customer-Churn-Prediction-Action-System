"""
Verification script for Churn Ensemble model.
Loads trained models and evaluates ensemble performance.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ensemble import create_default_ensemble
from data_preprocessing import load_and_preprocess_data

def verify_ensemble():
    print("="*60)
    print("VERIFYING CHURN ENSEMBLE PERFORMANCE")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load data
    print("\n1. Loading assessment data...")
    # Use existing data for testing
    data_path = os.path.join("data", "Business_Analytics_Dataset_10000_Rows.csv")
    if not os.path.exists(data_path):
        data_path = "Business_Analytics_Dataset_10000_Rows.csv" # Fallback
        
    if not os.path.exists(data_path):
        print(f"✗ Error: Data file {data_path} not found. Please ensure the dataset is in the data/ directory.")
        return
        
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_path)
    print(f"✓ Data loaded. Test samples: {len(X_test)}")
    
    # 2. Create Ensemble
    print("\n2. Creating ensemble from artifacts...")
    try:
        ensemble = create_default_ensemble(artifacts_dir='artifacts', device=device)
        print(f"✓ Ensemble created with {len(ensemble.models)} models")
    except Exception as e:
        print(f"✗ Failed to create ensemble: {e}")
        return

    # 3. Evaluate Ensemble
    print("\n3. Evaluating Ensemble on test set...")
    
    # Convert test data to tensor
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_np = y_test.numpy().flatten()
    
    # For sequence models in ensemble, we need sequence data.
    # For this basic verification, we'll simulate sequence data from static features
    # (since our default ensemble expects 10 timesteps)
    X_test_seq = X_test_tensor.unsqueeze(1).repeat(1, 10, 1)
    
    with torch.no_grad():
        y_probs = ensemble.predict_proba(X_test_tensor, X_test_seq).cpu().numpy().flatten()
        y_preds = (y_probs >= 0.5).astype(int)
        
    acc = accuracy_score(y_test_np, y_preds)
    f1 = f1_score(y_test_np, y_preds)
    auc = roc_auc_score(y_test_np, y_probs)
    
    print("-" * 30)
    print(f"Ensemble Accuracy: {acc:.4f}")
    print(f"Ensemble F1-Score: {f1:.4f}")
    print(f"Ensemble ROC-AUC:  {auc:.4f}")
    print("-" * 30)
    
    # 4. Compare with individual models
    print("\n4. Individual Model Performance:")
    for i, model in enumerate(ensemble.models):
        m_type = ensemble.model_metadata[i]['type']
        m_path = ensemble.model_metadata[i]['path']
        with torch.no_grad():
            if m_type == 'sequence':
                probs, _ = model(X_test_seq)
            elif m_type == 'mtl':
                probs, _ = model(X_test_tensor)
            else:
                probs = model(X_test_tensor)
            
            p = probs.cpu().numpy().flatten()
            pred = (p >= 0.5).astype(int)
            m_acc = accuracy_score(y_test_np, pred)
            print(f"- {m_type.upper():<10} Accuracy: {m_acc:.4f} | Source: {m_path}")

    # 5. Check missing models
    loaded_types = [m['type'] for m in ensemble.model_metadata]
    all_types = ['mlp', 'mtl', 'sequence']
    missing = [t for t in all_types if t not in loaded_types]
    if missing:
        print(f"\n⚠️ Missing model types: {', '.join(missing)}")
        print("Recommendation: Run corresponding training scripts to generate these components.")

    print("\n" + "="*60)
    print("✓ VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    verify_ensemble()
