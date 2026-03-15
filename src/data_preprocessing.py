"""
Data Preprocessing Module for Customer Churn Prediction
Handles data loading, feature engineering, and preprocessing

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from datetime import datetime, timedelta


class ChurnDataPreprocessor:
    """
    Preprocessor for customer churn prediction
    Handles feature engineering and data transformation
    """
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor
        
        Args:
            data_path (str): Path to the CSV data file
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self):
        """Load the dataset from CSV"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df
    
    def engineer_churn_label(self):
        """
        Engineer churn label from transactional data
        
        Churn is defined as:
        - Customer hasn't made a purchase in the last 90 days (from latest date)
        - OR customer has declining purchase frequency
        - OR customer's total profit contribution is in bottom 25%
        """
        print("Engineering churn label...")
        
        # Convert Order_Date to datetime
        self.df['Order_Date'] = pd.to_datetime(self.df['Order_Date'])
        
        # Find the latest date in dataset
        latest_date = self.df['Order_Date'].max()
        
        # Aggregate customer-level features
        customer_features = self.df.groupby('Customer_ID').agg({
            'Order_ID': 'count',  # Total orders
            'Order_Date': ['min', 'max'],  # First and last purchase
            'Revenue': ['sum', 'mean', 'std'],  # Revenue metrics
            'Profit': ['sum', 'mean'],  # Profit metrics
            'Discount_Rate': 'mean',  # Average discount
            'Quantity': ['sum', 'mean']  # Quantity metrics
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = ['Customer_ID', 'total_orders', 'first_purchase', 
                                     'last_purchase', 'total_revenue', 'avg_revenue', 
                                     'std_revenue', 'total_profit', 'avg_profit', 
                                     'avg_discount', 'total_quantity', 'avg_quantity']
        
        # Calculate days since last purchase
        customer_features['days_since_last_purchase'] = (
            latest_date - customer_features['last_purchase']
        ).dt.days
        
        # Calculate customer lifetime (days)
        customer_features['customer_lifetime_days'] = (
            customer_features['last_purchase'] - customer_features['first_purchase']
        ).dt.days + 1
        
        # Calculate purchase frequency (orders per day)
        customer_features['purchase_frequency'] = (
            customer_features['total_orders'] / customer_features['customer_lifetime_days']
        ).replace([np.inf, -np.inf], 0)
        
        # Define churn based on multiple criteria
        churn_threshold_days = 90
        profit_bottom_quartile = customer_features['total_profit'].quantile(0.30)
        
        customer_features['Churn'] = (
            (customer_features['days_since_last_purchase'] > churn_threshold_days) |
            (customer_features['total_profit'] < profit_bottom_quartile) |
            ((customer_features['total_orders'] < 3) & 
             (customer_features['days_since_last_purchase'] > 60))
        ).astype(int)
        
        # Add categorical features per customer
        categorical_features = self.df.groupby('Customer_ID').agg({
            'Region': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'Product_Category': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'Customer_Segment': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'Payment_Method': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        }).reset_index()
        
        # Merge categorical features
        customer_features = customer_features.merge(categorical_features, on='Customer_ID')
        
        # Fill missing values
        customer_features['std_revenue'].fillna(0, inplace=True)
        
        print(f"Churn rate: {customer_features['Churn'].mean():.2%}")
        print(f"Total customers: {len(customer_features)}")
        print(f"Churned customers: {customer_features['Churn'].sum()}")
        
        self.customer_df = customer_features
        return customer_features
    
    def prepare_features(self):
        """
        Prepare features for model training
        """
        print("Preparing features...")
        
        # Select numerical features
        numerical_features = [
            'total_orders', 'total_revenue', 'avg_revenue', 'std_revenue',
            'total_profit', 'avg_profit', 'avg_discount', 'total_quantity',
            'avg_quantity', 'days_since_last_purchase', 'customer_lifetime_days',
            'purchase_frequency'
        ]
        
        # Select categorical features
        categorical_features = [
            'Region', 'Product_Category', 'Customer_Segment', 'Payment_Method'
        ]
        
        # Encode categorical features
        df_encoded = self.customer_df.copy()
        
        for col in categorical_features:
            le = LabelEncoder()
            df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
            self.label_encoders[col] = le
        
        # Prepare final feature set
        encoded_categorical = [col + '_encoded' for col in categorical_features]
        self.feature_names = numerical_features + encoded_categorical
        
        X = df_encoded[self.feature_names].values
        y = df_encoded['Churn'].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        return X, y
    
    def split_and_scale(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train/test sets and scale features
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
            test_size (float): Proportion for test set
            random_state (int): Random seed
            
        Returns:
            tuple: Scaled train and test sets
        """
        print("Splitting and scaling data...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def to_torch_tensors(self, X_train, X_test, y_train, y_test):
        """
        Convert numpy arrays to PyTorch tensors
        
        Args:
            X_train, X_test, y_train, y_test: Numpy arrays
            
        Returns:
            tuple: PyTorch tensors
        """
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
    
    def get_feature_names(self):
        """Return list of feature names"""
        return self.feature_names
    
    def process_pipeline(self, test_size=0.2, random_state=42):
        """
        Run complete preprocessing pipeline
        
        Args:
            test_size (float): Proportion for test set
            random_state (int): Random seed
            
        Returns:
            tuple: Processed PyTorch tensors ready for training
        """
        # Load data
        self.load_data()
        
        # Engineer churn label
        self.engineer_churn_label()
        
        # Prepare features
        X, y = self.prepare_features()
        
        # Split and scale
        X_train, X_test, y_train, y_test = self.split_and_scale(
            X, y, test_size, random_state
        )
        
        # Convert to PyTorch tensors
        tensors = self.to_torch_tensors(X_train, X_test, y_train, y_test)
        
        print("Preprocessing complete!")
        return tensors


def load_and_preprocess_data(data_path, test_size=0.2, random_state=42):
    """
    Convenience function to load and preprocess data
    
    Args:
        data_path (str): Path to data file
        test_size (float): Test set proportion
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    preprocessor = ChurnDataPreprocessor(data_path)
    tensors = preprocessor.process_pipeline(test_size, random_state)
    
    return (*tensors, preprocessor)
