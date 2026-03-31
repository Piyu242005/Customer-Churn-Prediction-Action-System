import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

os.makedirs('models', exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv('data/Business_Analytics_Dataset_10000_Rows.csv')

# Create a target variable 'Churn' (Mocked based on business assumptions for the dataset)
np.random.seed(42)
# Churn is more likely if Profit is low, Discount_Rate is high, and Quantity is low
df['Churn'] = np.where((df['Profit'] < 50) | ((df['Discount_Rate'] > 0.2) & (df['Quantity'] <= 2)), 1, 0)
# Add some randomness to make it realistic
noise = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
df['Churn'] = np.where(noise == 1, 1 - df['Churn'], df['Churn'])

# Feature Engineering
features = ['Quantity', 'Unit_Price', 'Discount_Rate', 'Revenue', 'Cost', 'Profit']
X = df[features]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

best_model = None
best_roc = 0
best_name = ""

print("\n--- Model Comparison Results ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    print(f"{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc:.4f}\n")
    
    # We select the best model based on ROC-AUC for this iteration
    if roc > best_roc:
        best_roc = roc
        best_model = model
        best_name = name

# Save the best model and scaler
joblib.dump(best_model, 'models/best_churn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(features, 'models/feature_names.pkl')
print(f"====== Selected Best Model: {best_name} ======")
print("Saved to models/best_churn_model.pkl")