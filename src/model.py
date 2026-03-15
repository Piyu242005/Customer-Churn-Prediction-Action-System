"""
MLP Classifier Model for Customer Churn Prediction
Implements a Multilayer Perceptron with ReLU activation and dropout regularization

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Multilayer Perceptron Classifier for Binary Classification
    
    Architecture:
    - Input Layer
    - Hidden Layer 1 (128 neurons) with ReLU and Dropout
    - Hidden Layer 2 (64 neurons) with ReLU and Dropout
    - Hidden Layer 3 (32 neurons) with ReLU and Dropout
    - Output Layer (1 neuron) with Sigmoid
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        """
        Initialize the MLP Classifier
        
        Args:
            input_dim (int): Number of input features
            hidden_dims (list): List of hidden layer dimensions
            dropout_rate (float): Dropout probability for regularization
        """
        super(MLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build the network layers
        layers = []
        
        # Input to first hidden layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Predicted probabilities
        """
        return self.network(x)
    
    def predict(self, x, threshold=0.5):
        """
        Make binary predictions
        
        Args:
            x (torch.Tensor): Input features
            threshold (float): Classification threshold
            
        Returns:
            torch.Tensor: Binary predictions (0 or 1)
        """
        with torch.no_grad():
            probabilities = self.forward(x)
            predictions = (probabilities >= threshold).float()
        return predictions
    
    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


def create_model(input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
    """
    Factory function to create an MLP Classifier
    
    Args:
        input_dim (int): Number of input features
        hidden_dims (list): List of hidden layer dimensions
        dropout_rate (float): Dropout probability
        
    Returns:
        MLPClassifier: Initialized model
    """
    model = MLPClassifier(input_dim, hidden_dims, dropout_rate)
    return model
