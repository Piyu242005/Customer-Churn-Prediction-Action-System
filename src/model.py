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


class MLPMultiTaskHead(nn.Module):
    """
    Multi-Task MLP with shared trunk and two heads:
    - Binary churn probability (sigmoid output)
    - Continuous CLV / customer spend regression (linear output)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims=[128, 64, 32],
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Shared trunk (same as MLPClassifier, without final sigmoid)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        self.shared = nn.Sequential(*layers)

        # Task-specific heads
        self.churn_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid(),
        )
        self.clv_head = nn.Linear(hidden_dims[-1], 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            churn_prob: (batch_size, 1)
            clv:        (batch_size, 1)
        """
        h = self.shared(x)
        churn_prob = self.churn_head(h)
        clv = self.clv_head(h)
        return churn_prob, clv

    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'MTL_MLP',
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class SequenceMTLModel(nn.Module):
    """
    Temporal sequence model with GRU/LSTM backbone and two heads:
    - Churn probability
    - CLV regression

    Expected input shape: (batch_size, seq_len, feature_dim)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        use_lstm: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        rnn_cls = nn.LSTM if use_lstm else nn.GRU
        self.backbone = rnn_cls(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.churn_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        self.clv_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, feature_dim)
        Returns:
            churn_prob: (batch_size, 1)
            clv:        (batch_size, 1)
        """
        outputs, _ = self.backbone(x)
        # Use final timestep representation
        h_last = outputs[:, -1, :]
        churn_prob = self.churn_head(h_last)
        clv = self.clv_head(h_last)
        return churn_prob, clv

    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'Sequence_MTL',
            'feature_dim': self.backbone.input_size,
            'hidden_size': self.backbone.hidden_size,
            'num_layers': self.backbone.num_layers,
            'dropout': self.backbone.dropout,
            'rnn_type': type(self.backbone).__name__,
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


def create_mtl_model(input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
    """
    Factory for the multi-task MLP model with churn + CLV heads.
    """
    return MLPMultiTaskHead(input_dim, hidden_dims, dropout_rate)


def create_sequence_mtl_model(
    feature_dim: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    use_lstm: bool = False,
    dropout: float = 0.3,
):
    """
    Factory for sequence-based MTL model (GRU/LSTM backbone).
    """
    return SequenceMTLModel(
        feature_dim=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        use_lstm=use_lstm,
        dropout=dropout,
    )

