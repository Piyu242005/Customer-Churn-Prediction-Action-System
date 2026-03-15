"""
Advanced Training Pipeline - Unified Interface for All Models
Supports: MLP, MTL, Sequence-MTL (LSTM/GRU) with automatic model selection

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
import time
import json
import argparse
import os
import sys

# Import all training modules
from model import create_model, create_mtl_model, create_sequence_mtl_model
from train import MLPTrainer
from train_mtl import MTLTrainer, prepare_mtl_data
from train_sequence import SequenceMTLTrainer, load_and_preprocess_sequence_data


class AdvancedTrainingPipeline:
    """
    Unified training pipeline supporting multiple model architectures
    """
    
    def __init__(self, model_type='mlp', config=None):
        """
        Initialize the advanced training pipeline
        
        Args:
            model_type (str): Type of model ('mlp', 'mtl', 'sequence_lstm', 'sequence_gru')
            config (dict): Configuration parameters
        """
        self.model_type = model_type.lower()
        self.config = config or self.get_default_config(model_type)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing {model_type.upper()} training pipeline...")
        print(f"Using device: {self.device}")
        
    def get_default_config(self, model_type):
        """Get default configuration for each model type"""
        base_config = {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'patience': 15,
            'weight_decay': 1e-5,
            'test_size': 0.2,
            'random_state': 42
        }
        
        if model_type == 'mlp':
            base_config.update({
                'hidden_dims': [128, 64, 32],
                'dropout_rate': 0.3
            })
        elif model_type == 'mtl':
            base_config.update({
                'hidden_dims': [128, 64, 32],
                'dropout_rate': 0.3,
                'churn_weight': 1.0,
                'clv_weight': 0.5
            })
        elif model_type in ['sequence_lstm', 'sequence_gru']:
            base_config.update({
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.3,
                'seq_length': 10,
                'min_transactions': 5,
                'churn_weight': 1.0,
                'clv_weight': 0.5
            })
        
        return base_config
    
    def load_data(self):
        """Load data based on model type"""
        data_path = self.config.get('data_path', 'Business_Analytics_Dataset_10000_Rows.csv')
        
        if self.model_type == 'mlp':
            from data_preprocessing import load_and_preprocess_data
            return load_and_preprocess_data(data_path, self.config.get('test_size', 0.2), self.config.get('random_state', 42))
        
        elif self.model_type == 'mtl':
            return prepare_mtl_data(data_path, self.config.get('test_size', 0.2), self.config.get('random_state', 42))
        
        elif self.model_type in ['sequence_lstm', 'sequence_gru']:
            try:
                return load_and_preprocess_sequence_data(
                    data_path, 
                    seq_length=self.config.get('seq_length', 10),
                    min_transactions=self.config.get('min_transactions', 5),
                    test_size=self.config.get('test_size', 0.2),
                    random_state=self.config.get('random_state', 42)
                )
            except Exception as e:
                print(f"Error loading sequence data: {e}")
                print("Falling back to MTL model...")
                self.model_type = 'mtl'
                self.config = self.get_default_config('mtl')
                return self.load_data()
    
    def create_model(self, input_dim):
        """Create model based on type"""
        if self.model_type == 'mlp':
            return create_model(input_dim, self.config.get('hidden_dims', [128, 64, 32]), self.config.get('dropout_rate', 0.3))
        
        elif self.model_type == 'mtl':
            return create_mtl_model(input_dim, self.config.get('hidden_dims', [128, 64, 32]), self.config.get('dropout_rate', 0.3))
        
        elif self.model_type in ['sequence_lstm', 'sequence_gru']:
            use_lstm = 'lstm' in self.model_type
            return create_sequence_mtl_model(
                feature_dim=input_dim,
                hidden_size=self.config.get('hidden_size', 64),
                num_layers=self.config.get('num_layers', 2),
                use_lstm=use_lstm,
                dropout=self.config.get('dropout', 0.3)
            )
    
    def create_trainer(self, model):
        """Create trainer based on model type"""
        if self.model_type == 'mlp':
            return MLPTrainer(model, self.device)
        
        elif self.model_type == 'mtl':
            return MTLTrainer(
                model, self.device, 
                self.config.get('churn_weight', 1.0), 
                self.config.get('clv_weight', 0.5)
            )
        
        elif self.model_type in ['sequence_lstm', 'sequence_gru']:
            return SequenceMTLTrainer(
                model, self.device,
                self.config.get('churn_weight', 1.0),
                self.config.get('clv_weight', 0.5)
            )
    
    def prepare_data_loaders(self, data):
        """Prepare data loaders based on model type"""
        batch_size = self.config.get('batch_size', 32)
        
        if self.model_type == 'mlp':
            X_train, X_test, y_train, y_test, preprocessor = data
            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            
        elif self.model_type == 'mtl':
            (X_train, X_test, y_churn_train, y_churn_test,
             y_clv_train, y_clv_test, preprocessor) = data
            train_dataset = TensorDataset(X_train, y_churn_train, y_clv_train)
            test_dataset = TensorDataset(X_test, y_churn_test, y_clv_test)
            
        elif self.model_type in ['sequence_lstm', 'sequence_gru']:
            (train_sequences, test_sequences, y_churn_train, y_churn_test,
             y_clv_train, y_clv_test, preprocessor, customer_ids, sequence_dates, targets) = data
            train_dataset = TensorDataset(train_sequences, y_churn_train, y_clv_train)
            test_dataset = TensorDataset(test_sequences, y_churn_test, y_clv_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, test_loader, preprocessor
    
    def get_input_dimension(self, data):
        """Get input dimension for model creation"""
        if self.model_type == 'mlp':
            X_train, _, _, _, _ = data
            return X_train.shape[1]
        
        elif self.model_type == 'mtl':
            X_train, _, _, _, _, _, _ = data
            return X_train.shape[1]
        
        elif self.model_type in ['sequence_lstm', 'sequence_gru']:
            train_sequences, _, _, _, _, _, _, _, _, _ = data
            return train_sequences.shape[2]  # Features per timestep
    
    def train_model(self):
        """Main training pipeline"""
        print("="*80)
        print(f"Advanced Training Pipeline - {self.model_type.upper()} Model")
        print("="*80)
        
        # Load data
        print("\nLoading data...")
        data = self.load_data()
        
        # Prepare data loaders
        print("Preparing data loaders...")
        train_loader, test_loader, preprocessor = self.prepare_data_loaders(data)
        
        # Create model
        input_dim = self.get_input_dimension(data)
        print(f"Creating {self.model_type.upper()} model with input dimension: {input_dim}")
        model = self.create_model(input_dim)
        
        print(f"Model architecture:")
        print(model)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer
        trainer = self.create_trainer(model)
        
        # Train model
        print(f"\nTraining {self.model_type.upper()} model...")
        history = trainer.train(
            train_loader, test_loader,
            epochs=self.config['epochs'],
            learning_rate=self.config['learning_rate'],
            patience=self.config['patience'],
            weight_decay=self.config['weight_decay']
        )
        
        # Save model
        model_name = f"{self.model_type}_classifier.pth"
        trainer.save_model(model_name)
        
        # Save training history
        history_file = f"{self.model_type}_training_history.json"
        with open(history_file, 'w') as f:
            # Convert numpy types to Python types
            history_serializable = {}
            for k, v in history.items():
                if isinstance(v, list):
                    history_serializable[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                else:
                    history_serializable[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
            json.dump(history_serializable, f, indent=4)
        
        print(f"\nTraining complete!")
        print(f"Model saved to: {model_name}")
        print(f"History saved to: {history_file}")
        
        return history, model, preprocessor
    
    def compare_models(self, model_types=['mlp', 'mtl', 'sequence_lstm']):
        """
        Compare multiple model types
        
        Args:
            model_types: List of model types to compare
        """
        print("="*80)
        print("Model Comparison Pipeline")
        print("="*80)
        
        results = {}
        
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()} Model")
            print(f"{'='*60}")
            
            try:
                # Reset model type and config
                self.model_type = model_type
                self.config = self.get_default_config(model_type)
                
                # Train model
                history, model, preprocessor = self.train_model()
                
                # Store results
                results[model_type] = {
                    'history': history,
                    'model': model,
                    'preprocessor': preprocessor,
                    'best_val_loss': min(history['val_losses']),
                    'final_val_accuracy': history['val_accuracies'][np.argmin(history['val_losses'])]
                }
                
                if 'clv_mse' in history:
                    results[model_type]['final_val_clv_mse'] = history['val_clv_mse'][np.argmin(history['val_losses'])]
                
            except Exception as e:
                print(f"Error training {model_type}: {e}")
                continue
        
        # Print comparison
        print(f"\n{'='*80}")
        print("Model Comparison Results")
        print(f"{'='*80}")
        
        for model_type, result in results.items():
            print(f"\n{model_type.upper()}:")
            print(f"  Best Validation Loss: {result['best_val_loss']:.4f}")
            print(f"  Final Validation Accuracy: {result['final_val_accuracy']:.4f}")
            if 'final_val_clv_mse' in result:
                print(f"  Final Validation CLV MSE: {result['final_val_clv_mse']:.4f}")
        
        # Save comparison results
        comparison_results = {}
        for model_type, result in results.items():
            comparison_results[model_type] = {
                'best_val_loss': result['best_val_loss'],
                'final_val_accuracy': result['final_val_accuracy']
            }
            if 'final_val_clv_mse' in result:
                comparison_results[model_type]['final_val_clv_mse'] = result['final_val_clv_mse']
        
        with open('model_comparison_results.json', 'w') as f:
            json.dump(comparison_results, f, indent=4)
        
        print(f"\nComparison results saved to: model_comparison_results.json")
        
        return results


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Churn Prediction Training Pipeline')
    
    parser.add_argument('--model-type', type=str, default='mlp',
                        choices=['mlp', 'mtl', 'sequence_lstm', 'sequence_gru', 'compare'],
                        help='Type of model to train')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    
    parser.add_argument('--data-path', type=str, 
                        default='Business_Analytics_Dataset_10000_Rows.csv',
                        help='Path to data file')
    
    parser.add_argument('--compare-models', action='store_true',
                        help='Compare multiple model types')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Configuration
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'weight_decay': args.weight_decay,
        'data_path': args.data_path
    }
    
    # Initialize pipeline
    if args.model_type == 'compare' or args.compare_models:
        # Compare all models
        pipeline = AdvancedTrainingPipeline('mlp', config)
        results = pipeline.compare_models(['mlp', 'mtl', 'sequence_lstm', 'sequence_gru'])
    else:
        # Train single model
        pipeline = AdvancedTrainingPipeline(args.model_type, config)
        history, model, preprocessor = pipeline.train_model()
        
        print(f"\n{'='*80}")
        print("Training Summary")
        print(f"{'='*80}")
        print(f"Model Type: {args.model_type.upper()}")
        print(f"Best Validation Loss: {min(history['val_losses']):.4f}")
        print(f"Final Validation Accuracy: {history['val_accuracies'][np.argmin(history['val_losses'])]:.4f}")
        if 'clv_mse' in history:
            print(f"Final Validation CLV MSE: {history['val_clv_mse'][np.argmin(history['val_losses'])]:.4f}")
        print(f"Training Time: {history['training_time']:.2f} seconds")
        print(f"Epochs Trained: {history['epochs_trained']}")


if __name__ == "__main__":
    main()
