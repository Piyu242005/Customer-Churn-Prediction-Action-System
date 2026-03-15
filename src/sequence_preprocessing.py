"""
Temporal Sequence Data Preprocessing for LSTM/GRU Models
Processes transactional data into sequences for time-series modeling

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class SequenceDataPreprocessor:
    """
    Preprocessor for creating temporal sequences from transactional data
    """
    
    def __init__(self, data_path, seq_length=10, min_transactions=5):
        """
        Initialize the sequence preprocessor
        
        Args:
            data_path (str): Path to the CSV data file or database
            seq_length (int): Length of sequences to create
            min_transactions (int): Minimum transactions required per customer
        """
        self.data_path = data_path
        self.seq_length = seq_length
        self.min_transactions = min_transactions
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self):
        """Load the dataset from CSV or SQLite Database"""
        if self.data_path.endswith('.db') or self.data_path.startswith('sqlite:///'):
            import sqlalchemy
            
            db_url = self.data_path if self.data_path.startswith('sqlite:///') else f'sqlite:///{self.data_path}'
            
            print(f"Loading data from database {db_url}...")
            engine = sqlalchemy.create_engine(db_url)
            query = "SELECT * FROM transactions"
            self.df = pd.read_sql(query, engine)
        else:
            print(f"Loading data from {self.data_path}...")
            self.df = pd.read_csv(self.data_path)
            
        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df
    
    def engineer_features(self):
        """
        Engineer temporal features for sequence modeling
        """
        print("Engineering temporal features...")
        
        # Convert Order_Date to datetime
        self.df['Order_Date'] = pd.to_datetime(self.df['Order_Date'])
        
        # Sort by customer and date
        self.df = self.df.sort_values(['Customer_ID', 'Order_Date'])
        
        # Create temporal features
        self.df['days_since_prev_order'] = self.df.groupby('Customer_ID')['Order_Date'].diff().dt.days
        self.df['days_since_prev_order'] = self.df['days_since_prev_order'].fillna(0)
        
        # Create cumulative features
        self.df['cumulative_orders'] = self.df.groupby('Customer_ID').cumcount() + 1
        self.df['cumulative_revenue'] = self.df.groupby('Customer_ID')['Revenue'].cumsum()
        self.df['cumulative_profit'] = self.df.groupby('Customer_ID')['Profit'].cumsum()
        
        # Create rolling averages
        self.df['rolling_avg_revenue_3'] = self.df.groupby('Customer_ID')['Revenue'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        self.df['rolling_avg_profit_3'] = self.df.groupby('Customer_ID')['Profit'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        
        # Create revenue trend (change from previous order)
        self.df['revenue_change'] = self.df.groupby('Customer_ID')['Revenue'].diff().fillna(0)
        
        # Encode categorical features
        categorical_cols = ['Region', 'Product_Category', 'Customer_Segment', 'Payment_Method']
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
        
        print("Feature engineering complete")
        return self.df
    
    def create_sequences(self):
        """
        Create sequences for each customer
        
        Returns:
            tuple: (sequences, targets, customer_ids)
        """
        print(f"Creating sequences of length {self.seq_length}...")
        
        # Define sequence features
        sequence_features = [
            'Quantity', 'Unit_Price', 'Discount_Rate', 'Revenue', 'Cost', 'Profit',
            'days_since_prev_order', 'cumulative_orders', 'cumulative_revenue', 
            'cumulative_profit', 'rolling_avg_revenue_3', 'rolling_avg_profit_3',
            'revenue_change', 'Region_encoded', 'Product_Category_encoded', 
            'Customer_Segment_encoded', 'Payment_Method_encoded'
        ]
        
        # Filter to available features
        available_features = [f for f in sequence_features if f in self.df.columns]
        self.feature_names = available_features
        print(f"Using {len(available_features)} features: {available_features}")
        
        sequences = []
        targets = []
        customer_ids = []
        sequence_dates = []
        
        # Group by customer
        for customer_id, customer_data in self.df.groupby('Customer_ID'):
            if len(customer_data) < self.min_transactions:
                continue
                
            # Sort by date
            customer_data = customer_data.sort_values('Order_Date')
            
            # Get feature matrix
            feature_matrix = customer_data[available_features].values
            
            # Create sequences
            for i in range(len(feature_matrix) - self.seq_length):
                # Sequence: features from i to i+seq_length
                seq = feature_matrix[i:i+self.seq_length]
                
                # Target: next transaction's features (we'll predict churn probability and CLV)
                target_features = feature_matrix[i+self.seq_length]
                
                # Create churn target: 1 if this is the last transaction or customer churns after this
                is_last_transaction = (i + self.seq_length == len(feature_matrix) - 1)
                
                # Additional churn logic: if no more transactions within 90 days
                if not is_last_transaction:
                    next_date = customer_data.iloc[i+self.seq_length]['Order_Date']
                    current_date = customer_data.iloc[i+self.seq_length-1]['Order_Date']
                    days_gap = (next_date - current_date).days
                    churn_target = 1 if days_gap > 90 else 0
                else:
                    # For last transaction, check if it's been >90 days from dataset end
                    dataset_end = self.df['Order_Date'].max()
                    last_date = customer_data.iloc[-1]['Order_Date']
                    days_since_end = (dataset_end - last_date).days
                    churn_target = 1 if days_since_end > 90 else 0
                
                # CLV target: use cumulative profit as proxy
                clv_target = customer_data.iloc[i+self.seq_length]['cumulative_profit']
                
                # Store sequence and targets
                sequences.append(seq)
                targets.append({
                    'churn': churn_target,
                    'clv': clv_target,
                    'next_revenue': target_features[3] if len(target_features) > 3 else 0,
                    'next_profit': target_features[4] if len(target_features) > 4 else 0
                })
                customer_ids.append(customer_id)
                sequence_dates.append(customer_data.iloc[i+self.seq_length-1]['Order_Date'])
        
        print(f"Created {len(sequences)} sequences from {len(set(customer_ids))} customers")
        
        return np.array(sequences), targets, customer_ids, sequence_dates
    
    def engineer_churn_label_from_sequences(self, targets):
        """
        Create churn labels from sequence targets
        
        Args:
            targets: List of target dictionaries
            
        Returns:
            tuple: (churn_labels, clv_values)
        """
        churn_labels = np.array([t['churn'] for t in targets])
        clv_values = np.array([t['clv'] for t in targets])
        
        # Normalize CLV values
        clv_values = np.log1p(np.abs(clv_values))  # Log transform
        clv_values = (clv_values - clv_values.min()) / (clv_values.max() - clv_values.min())  # Normalize
        
        print(f"Churn rate: {churn_labels.mean():.2%}")
        print(f"CLV range: [{clv_values.min():.3f}, {clv_values.max():.3f}]")
        
        return churn_labels, clv_values
    
    def scale_sequences(self, sequences):
        """
        Scale sequence features
        
        Args:
            sequences: numpy array of shape (n_sequences, seq_length, n_features)
            
        Returns:
            scaled sequences
        """
        print("Scaling sequence features...")
        
        # Reshape for scaling: (n_sequences * seq_length, n_features)
        original_shape = sequences.shape
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        
        # Fit scaler on training data and transform
        sequences_scaled = self.scaler.fit_transform(sequences_reshaped)
        
        # Reshape back: (n_sequences, seq_length, n_features)
        sequences_scaled = sequences_scaled.reshape(original_shape)
        
        print("Sequence scaling complete")
        return sequences_scaled
    
    def split_and_scale(self, sequences, churn_labels, clv_values, test_size=0.2, random_state=42):
        """
        Split data into train/test sets and scale features
        
        Args:
            sequences: Sequence data
            churn_labels: Churn targets
            clv_values: CLV targets
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            tuple: Scaled and split data
        """
        print("Splitting and scaling data...")
        
        # Split data
        indices = np.arange(len(sequences))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=churn_labels
        )
        
        # Split sequences
        train_sequences = sequences[train_idx]
        test_sequences = sequences[test_idx]
        
        # Scale sequences (fit on training only)
        train_sequences_scaled = self.scale_sequences(train_sequences)
        
        # Transform test sequences using fitted scaler
        test_reshaped = test_sequences.reshape(-1, test_sequences.shape[-1])
        test_sequences_scaled = self.scaler.transform(test_reshaped)
        test_sequences_scaled = test_sequences_scaled.reshape(test_sequences.shape)
        
        # Split targets
        y_churn_train = churn_labels[train_idx]
        y_churn_test = churn_labels[test_idx]
        y_clv_train = clv_values[train_idx]
        y_clv_test = clv_values[test_idx]
        
        print(f"Training set: {train_sequences_scaled.shape}")
        print(f"Test set: {test_sequences_scaled.shape}")
        
        return (train_sequences_scaled, test_sequences_scaled, 
                y_churn_train, y_churn_test, y_clv_train, y_clv_test)
    
    def to_torch_tensors(self, train_sequences, test_sequences, 
                        y_churn_train, y_churn_test, y_clv_train, y_clv_test):
        """
        Convert numpy arrays to PyTorch tensors
        
        Args:
            Various numpy arrays
            
        Returns:
            tuple: PyTorch tensors
        """
        train_sequences_tensor = torch.FloatTensor(train_sequences)
        test_sequences_tensor = torch.FloatTensor(test_sequences)
        y_churn_train_tensor = torch.FloatTensor(y_churn_train).unsqueeze(1)
        y_churn_test_tensor = torch.FloatTensor(y_churn_test).unsqueeze(1)
        y_clv_train_tensor = torch.FloatTensor(y_clv_train).unsqueeze(1)
        y_clv_test_tensor = torch.FloatTensor(y_clv_test).unsqueeze(1)
        
        return (train_sequences_tensor, test_sequences_tensor,
                y_churn_train_tensor, y_churn_test_tensor,
                y_clv_train_tensor, y_clv_test_tensor)
    
    def get_feature_names(self):
        """Return list of feature names"""
        return self.feature_names
    
    def process_pipeline(self, test_size=0.2, random_state=42):
        """
        Run complete sequence preprocessing pipeline
        
        Args:
            test_size (float): Proportion for test set
            random_state (int): Random seed
            
        Returns:
            tuple: Processed PyTorch tensors ready for training
        """
        # Load data
        self.load_data()
        
        # Engineer features
        self.engineer_features()
        
        # Create sequences
        sequences, targets, customer_ids, sequence_dates = self.create_sequences()
        
        # Create labels
        churn_labels, clv_values = self.engineer_churn_label_from_sequences(targets)
        
        # Split and scale
        (train_sequences, test_sequences, 
         y_churn_train, y_churn_test, y_clv_train, y_clv_test) = self.split_and_scale(
            sequences, churn_labels, clv_values, test_size, random_state
        )
        
        # Convert to PyTorch tensors
        tensors = self.to_torch_tensors(
            train_sequences, test_sequences,
            y_churn_train, y_churn_test, y_clv_train, y_clv_test
        )
        
        print("Sequence preprocessing complete!")
        return (*tensors, self, customer_ids, sequence_dates, targets)


def load_and_preprocess_sequence_data(data_path, seq_length=10, min_transactions=5, 
                                    test_size=0.2, random_state=42):
    """
    Convenience function to load and preprocess sequence data
    
    Args:
        data_path (str): Path to data file
        seq_length (int): Length of sequences
        min_transactions (int): Minimum transactions per customer
        test_size (float): Test set proportion
        random_state (int): Random seed
        
    Returns:
        tuple: (train_sequences, test_sequences, y_churn_train, y_churn_test, 
                y_clv_train, y_clv_test, preprocessor, customer_ids, sequence_dates, targets)
    """
    preprocessor = SequenceDataPreprocessor(data_path, seq_length, min_transactions)
    tensors = preprocessor.process_pipeline(test_size, random_state)
    
    return tensors


def analyze_sequence_data(sequences, targets, customer_ids):
    """
    Analyze sequence data characteristics
    
    Args:
        sequences: Sequence data
        targets: Target data
        customer_ids: Customer IDs
    """
    print("\n" + "="*60)
    print("Sequence Data Analysis")
    print("="*60)
    
    print(f"Total sequences: {len(sequences)}")
    print(f"Unique customers: {len(set(customer_ids))}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Average sequences per customer: {len(sequences) / len(set(customer_ids)):.2f}")
    
    # Churn analysis
    churn_labels = [t['churn'] for t in targets]
    churn_rate = np.mean(churn_labels)
    print(f"Overall churn rate: {churn_rate:.2%}")
    
    # CLV analysis
    clv_values = [t['clv'] for t in targets]
    print(f"CLV stats - Mean: {np.mean(clv_values):.2f}, Std: {np.std(clv_values):.2f}")
    print(f"CLV range: [{np.min(clv_values):.2f}, {np.max(clv_values):.2f}]")
    
    # Revenue analysis
    next_revenue = [t['next_revenue'] for t in targets]
    print(f"Next transaction revenue - Mean: {np.mean(next_revenue):.2f}")
    
    print("="*60)


if __name__ == "__main__":
    # Test the sequence preprocessing
    data_path = "Business_Analytics_Dataset_10000_Rows.csv"
    
    print("Testing Sequence Data Preprocessing")
    print("="*60)
    
    try:
        (train_sequences, test_sequences, y_churn_train, y_churn_test,
         y_clv_train, y_clv_test, preprocessor, customer_ids, sequence_dates, targets) = load_and_preprocess_sequence_data(
            data_path, seq_length=8, min_transactions=5
        )
        
        analyze_sequence_data(train_sequences, targets, customer_ids)
        
        print(f"\nPreprocessing successful!")
        print(f"Training sequences shape: {train_sequences.shape}")
        print(f"Test sequences shape: {test_sequences.shape}")
        print(f"Feature names: {preprocessor.get_feature_names()}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
