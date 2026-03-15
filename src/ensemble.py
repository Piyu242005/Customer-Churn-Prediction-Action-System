"""
Model Ensemble Module for Churn Prediction
Aggregates predictions from multiple base models (MLP, MTL, Sequence)
using weighted voting or soft voting strategies.

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Union, Optional
import os

from src.model.model import MLPClassifier, MLPMultiTaskHead, SequenceMTLModel

class ChurnEnsemble:
    """
    Ensemble model that combines multiple churn classifiers.
    Supports Weighted Soft Voting.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.models = []
        self.weights = []
        self.model_metadata = []
        self.device = device
        
    def add_model(self, model: nn.Module, weight: float = 1.0, metadata: Optional[Dict] = None):
        """Add a model to the ensemble"""
        model.to(self.device)
        model.eval()
        self.models.append(model)
        self.weights.append(weight)
        self.model_metadata.append(metadata or {})
        
    def load_from_checkpoints(self, checkpoint_configs: List[Dict[str, Any]]):
        """
        Load multiple models from checkpoints.
        
        Args:
            checkpoint_configs: List of dicts with:
                - 'path': path to .pth file
                - 'type': 'mlp', 'mtl', or 'sequence'
                - 'weight': voting weight (default 1.0)
                - 'params': dict of model init params
        """
        for config in checkpoint_configs:
            path = config['path']
            m_type = config['type']
            weight = config.get('weight', 1.0)
            params = config.get('params', {})
            
            if not os.path.exists(path):
                print(f"Warning: Checkpoint not found at {path}")
                continue
                
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            if m_type == 'mlp':
                model = MLPClassifier(**params)
            elif m_type == 'mtl':
                model = MLPMultiTaskHead(**params)
            elif m_type == 'sequence':
                model = SequenceMTLModel(**params)
            else:
                raise ValueError(f"Unknown model type: {m_type}")
                
            model.load_state_dict(checkpoint['model_state_dict'])
            self.add_model(model, weight, metadata={'type': m_type, 'path': path})
            print(f"✓ Loaded {m_type.upper()} model from {path} with weight {weight}")

    def predict_proba(self, x: torch.Tensor, x_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get ensemble churn probabilities
        
        Args:
            x: 2D feature tensor for MLP/MTL models (batch_size, input_dim)
            x_seq: 3D feature tensor for Sequence models (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Weighted average churn probabilities
        """
        if not self.models:
            raise ValueError("No models added to ensemble")
            
        all_probs = []
        normalized_weights = np.array(self.weights) / sum(self.weights)
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                m_type = self.model_metadata[i].get('type')
                
                if m_type == 'sequence':
                    if x_seq is None:
                        # Fallback: if x_seq not provided but model is sequence, 
                        # try to unsqueeze x if it fits or skip
                        raise ValueError("Sequence model requires x_seq input")
                    probs, _ = model(x_seq.to(self.device))
                elif m_type == 'mtl':
                    probs, _ = model(x.to(self.device))
                else: # mlp
                    probs = model(x.to(self.device))
                    
                all_probs.append(probs * normalized_weights[i])
                
        ensemble_prob = torch.stack(all_probs).sum(dim=0)
        return ensemble_prob

    def predict(self, x: torch.Tensor, x_seq: Optional[torch.Tensor] = None, threshold: float = 0.5) -> torch.Tensor:
        """Predict binary churn class"""
        probs = self.predict_proba(x, x_seq)
        return (probs >= threshold).float()

def create_default_ensemble(artifacts_dir: str = 'artifacts', device: str = 'cpu') -> ChurnEnsemble:
    """Create a default ensemble using best available models in project directories"""
    ensemble = ChurnEnsemble(device=device)
    
    # Define possible search paths for each model type
    model_search_configs = [
        {
            'type': 'mlp',
            'paths': [
                os.path.join(artifacts_dir, 'mlp_churn_classifier_final.pth'),
                'mlp_churn_classifier_final.pth',
                'serving/mlp_churn_classifier_final.pth',
                'mlp_classifier.pth'
            ],
            'weight': 1.0,
            'params': {'input_dim': 16, 'hidden_dims': [128, 64, 32]}
        },
        {
            'type': 'mtl',
            'paths': [
                os.path.join(artifacts_dir, 'mtl_churn_clv_classifier.pth'),
                'mtl_churn_clv_classifier.pth',
                'serving/mtl_churn_clv_classifier.pth',
                'mtl_classifier.pth'
            ],
            'weight': 1.2,
            'params': {'input_dim': 16, 'hidden_dims': [128, 64, 32]}
        },
        {
            'type': 'sequence',
            'paths': [
                os.path.join(artifacts_dir, 'sequence_mtl_lstm_classifier.pth'),
                'sequence_mtl_lstm_classifier.pth',
                'serving/sequence_mtl_lstm_classifier.pth',
                os.path.join(artifacts_dir, 'sequence_mtl_gru_classifier.pth'),
                'sequence_mtl_gru_classifier.pth'
            ],
            'weight': 1.5,
            'params': {'feature_dim': 16, 'hidden_size': 64, 'num_layers': 2, 'use_lstm': True}
        }
    ]
    
    loaded_any = False
    for config in model_search_configs:
        found_path = None
        for path in config['paths']:
            if os.path.exists(path):
                found_path = path
                break
        
        if found_path:
            # Update sequence params if GRU was found instead of LSTM
            params = config['params'].copy()
            if config['type'] == 'sequence' and 'gru' in found_path.lower():
                params['use_lstm'] = False
                
            try:
                ensemble.load_from_checkpoints([{
                    'path': found_path,
                    'type': config['type'],
                    'weight': config['weight'],
                    'params': params
                }])
                loaded_any = True
            except Exception as e:
                print(f"Error loading {config['type']} from {found_path}: {e}")
                
    if not loaded_any:
        print("Warning: No models could be loaded into the ensemble.")
            
    return ensemble

if __name__ == "__main__":
    # Test loading
    try:
        ensemble = create_default_ensemble()
        print(f"Ensemble created with {len(ensemble.models)} models")
    except Exception as e:
        print(f"Error creating ensemble: {e}")
