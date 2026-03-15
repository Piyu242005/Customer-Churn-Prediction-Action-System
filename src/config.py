"""
Configuration file for MLP Churn Classifier
Centralized hyperparameters and settings

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

# Data Configuration
DATA_CONFIG = {
    'data_path': 'data/Business_Analytics_Dataset_10000_Rows.csv',
    'test_size': 0.2,
    'random_state': 42,
    'churn_threshold_days': 90,
    'profit_bottom_quartile': 0.30
}

# Model Architecture
MODEL_CONFIG = {
    'input_dim': 16,  # Will be determined from data
    'hidden_dims': [128, 64, 32],
    'dropout_rate': 0.3,
    'activation': 'relu'
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Learning Rate Scheduler Configuration
SCHEDULER_CONFIG = {
    'mode': 'min',
    'factor': 0.5,
    'patience': 5
}

# Cross-Validation Configuration
CV_CONFIG = {
    'n_folds': 5,
    'shuffle': True,
    'random_state': 42
}

# Evaluation Configuration
EVAL_CONFIG = {
    'threshold': 0.5,
    'save_plots': True,
    'plot_dpi': 300
}

# File Paths
PATHS = {
    'model_checkpoint': 'artifacts/mlp_churn_classifier.pth',
    'model_final': 'artifacts/mlp_churn_classifier_final.pth',
    'training_history': 'outputs/training_history.json',
    'plots_dir': 'Assets/'
}

# Reproducibility
SEED = 42

import torch
import numpy as np
import random

def set_seed(seed=SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
