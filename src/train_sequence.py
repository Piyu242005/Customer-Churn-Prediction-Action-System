"""
Sequence Model Training Script for LSTM/GRU with Multi-Task Learning
Implements temporal sequence modeling for churn + CLV prediction

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
import mlflow
import mlflow.pytorch
import os

from model import create_sequence_mtl_model
from sequence_preprocessing import load_and_preprocess_sequence_data


class SequenceMTLTrainer:
    """
    Trainer class for Sequence-based Multi-Task Learning (Churn + CLV)
    """
    
    def __init__(self, model, device='cpu', churn_weight=1.0, clv_weight=0.5, **kwargs):
        """
        Initialize Sequence MTL trainer
        
        Args:
            model: PyTorch sequence MTL model
            device: Device to train on ('cpu' or 'cuda')
            churn_weight: Weight for churn loss
            clv_weight: Weight for CLV loss
        """
        self.model = model.to(device)
        self.device = device
        self.churn_weight = churn_weight
        self.clv_weight = clv_weight
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_churn_losses = []
        self.train_clv_losses = []
        self.val_churn_losses = []
        self.val_clv_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_clv_mse = []
        self.val_clv_mse = []
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Set up MLflow tracking URI and experiment
        self.use_mlflow = kwargs.get('use_mlflow', True)
        if self.use_mlflow:
            try:
                mlflow.set_tracking_uri("sqlite:///mlflow.db")
                mlflow.set_experiment("Churn_Prediction_Sequence_MTL")
            except Exception as e:
                print(f"Warning: MLflow setup failed: {e}. Proceeding without MLflow.")
                self.use_mlflow = False
        
    def compute_losses(self, churn_pred, clv_pred, churn_true, clv_true):
        """
        Compute losses for both tasks
        
        Args:
            churn_pred: Predicted churn probabilities
            clv_pred: Predicted CLV values
            churn_true: True churn labels
            clv_true: True CLV values
            
        Returns:
            tuple: (total_loss, churn_loss, clv_loss)
        """
        # Binary cross entropy for churn
        churn_criterion = nn.BCELoss()
        churn_loss = churn_criterion(churn_pred, churn_true)
        
        # Mean squared error for CLV
        clv_criterion = nn.MSELoss()
        clv_loss = clv_criterion(clv_pred, clv_true)
        
        # Weighted combination
        total_loss = self.churn_weight * churn_loss + self.clv_weight * clv_loss
        
        return total_loss, churn_loss, clv_loss
    
    def train_epoch(self, train_loader, optimizer):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer
            
        Returns:
            tuple: (total_loss, churn_loss, clv_loss, accuracy, clv_mse)
        """
        self.model.train()
        running_total_loss = 0.0
        running_churn_loss = 0.0
        running_clv_loss = 0.0
        correct = 0
        total = 0
        clv_predictions = []
        clv_targets = []
        
        for batch_X, batch_churn, batch_clv in train_loader:
            batch_X = batch_X.to(self.device)
            batch_churn = batch_churn.to(self.device)
            batch_clv = batch_clv.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            churn_pred, clv_pred = self.model(batch_X)
            
            # Compute losses
            total_loss, churn_loss, clv_loss = self.compute_losses(
                churn_pred, clv_pred, batch_churn, batch_clv
            )
            
            # Backward pass and optimization
            total_loss.backward()
            
            # Gradient clipping for sequences
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            batch_size = batch_X.size(0)
            running_total_loss += total_loss.item() * batch_size
            running_churn_loss += churn_loss.item() * batch_size
            running_clv_loss += clv_loss.item() * batch_size
            
            # Churn accuracy
            predicted = (churn_pred >= 0.5).float()
            total += batch_size
            correct += (predicted == batch_churn).sum().item()
            
            # CLV predictions for MSE calculation
            clv_predictions.extend(clv_pred.detach().cpu().numpy())
            clv_targets.extend(batch_clv.cpu().numpy())
        
        # Calculate metrics
        epoch_total_loss = running_total_loss / total
        epoch_churn_loss = running_churn_loss / total
        epoch_clv_loss = running_clv_loss / total
        epoch_acc = correct / total
        epoch_clv_mse = mean_squared_error(clv_targets, clv_predictions)
        
        return epoch_total_loss, epoch_churn_loss, epoch_clv_loss, epoch_acc, epoch_clv_mse
    
    def validate(self, val_loader):
        """
        Validate model
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            tuple: (total_loss, churn_loss, clv_loss, accuracy, clv_mse)
        """
        self.model.eval()
        running_total_loss = 0.0
        running_churn_loss = 0.0
        running_clv_loss = 0.0
        correct = 0
        total = 0
        clv_predictions = []
        clv_targets = []
        
        with torch.no_grad():
            for batch_X, batch_churn, batch_clv in val_loader:
                batch_X = batch_X.to(self.device)
                batch_churn = batch_churn.to(self.device)
                batch_clv = batch_clv.to(self.device)
                
                # Forward pass
                churn_pred, clv_pred = self.model(batch_X)
                
                # Compute losses
                total_loss, churn_loss, clv_loss = self.compute_losses(
                    churn_pred, clv_pred, batch_churn, batch_clv
                )
                
                # Statistics
                batch_size = batch_X.size(0)
                running_total_loss += total_loss.item() * batch_size
                running_churn_loss += churn_loss.item() * batch_size
                running_clv_loss += clv_loss.item() * batch_size
                
                # Churn accuracy
                predicted = (churn_pred >= 0.5).float()
                total += batch_size
                correct += (predicted == batch_churn).sum().item()
                
                # CLV predictions for MSE calculation
                clv_predictions.extend(clv_pred.cpu().numpy())
                clv_targets.extend(batch_clv.cpu().numpy())
        
        # Calculate metrics
        epoch_total_loss = running_total_loss / total
        epoch_churn_loss = running_churn_loss / total
        epoch_clv_loss = running_clv_loss / total
        epoch_acc = correct / total
        epoch_clv_mse = mean_squared_error(clv_targets, clv_predictions)
        
        return epoch_total_loss, epoch_churn_loss, epoch_clv_loss, epoch_acc, epoch_clv_mse
    
    def train(self, train_loader, val_loader, epochs=100, learning_rate=0.001, 
              patience=15, weight_decay=1e-5):
        """
        Train sequence MTL model with early stopping
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Maximum number of epochs
            learning_rate: Initial learning rate
            patience: Early stopping patience
            weight_decay: L2 regularization parameter
            
        Returns:
            dict: Training history
        """
        print(f"\nTraining Sequence MTL model on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Task weights - Churn: {self.churn_weight}, CLV: {self.clv_weight}")
        
        # Adam optimizer with weight decay
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping
        epochs_no_improve = 0
        
        start_time = time.time()
        
        print(f"{'#'*80}")
        
        class DummyContextManager:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc_val, exc_tb): pass

        ctx = mlflow.start_run(run_name="Sequence_MTL_Churn_CLV_Training", nested=True) if self.use_mlflow else DummyContextManager()
        
        with ctx:
            if self.use_mlflow:
                # Log Hyperparameters
                mlflow.log_params({
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "patience": patience,
                    "weight_decay": weight_decay,
                    "churn_weight": self.churn_weight,
                    "clv_weight": self.clv_weight,
                    "model_type": "Sequence_MTL",
                    "hidden_size": self.model.backbone.hidden_size,
                    "num_layers": self.model.backbone.num_layers,
                    "use_lstm": hasattr(self.model.backbone, 'lstm') or 
                                isinstance(self.model.backbone, torch.nn.LSTM)
                })
            
            for epoch in range(epochs):
                # Train
                (train_total_loss, train_churn_loss, train_clv_loss, 
                 train_acc, train_clv_mse) = self.train_epoch(train_loader, optimizer)
                
                # Validate
                (val_total_loss, val_churn_loss, val_clv_loss, 
                 val_acc, val_clv_mse) = self.validate(val_loader)
                
                # Learning rate scheduling
                scheduler.step(val_total_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Save history
                self.train_losses.append(train_total_loss)
                self.val_losses.append(val_total_loss)
                self.train_churn_losses.append(train_churn_loss)
                self.val_churn_losses.append(val_churn_loss)
                self.train_clv_losses.append(train_clv_loss)
                self.val_clv_losses.append(val_clv_loss)
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
                self.train_clv_mse.append(train_clv_mse)
                self.val_clv_mse.append(val_clv_mse)
                
                # Print progress
                if (epoch + 1) % 10 == 0 or (epoch == 0):
                    print(f"Epoch [{epoch+1}/{epochs}] "
                          f"Total Loss: {train_total_loss:.4f}, Val: {val_total_loss:.4f} | "
                          f"Churn Acc: {train_acc:.4f}, Val: {val_acc:.4f} | "
                          f"CLV MSE: {train_clv_mse:.4f}, Val: {val_clv_mse:.4f} | "
                          f"LR: {current_lr:.6f}")
                    
                # Log metrics to MLflow
                if self.use_mlflow:
                    mlflow.log_metrics({
                        "train_total_loss": train_total_loss,
                        "val_total_loss": val_total_loss,
                        "train_churn_loss": train_churn_loss,
                        "val_churn_loss": val_churn_loss,
                        "train_clv_loss": train_clv_loss,
                        "val_clv_loss": val_clv_loss,
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                        "train_clv_mse": train_clv_mse,
                        "val_clv_mse": val_clv_mse,
                        "learning_rate": current_lr
                    }, step=epoch)
                
                # Early stopping and best model saving
                if val_total_loss < self.best_val_loss:
                    self.best_val_loss = val_total_loss
                    self.best_model_state = self.model.state_dict().copy()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            
            # Load best model
            self.model.load_state_dict(self.best_model_state)
            
            training_time = time.time() - start_time
            print(f"\nTraining completed in {training_time:.2f} seconds")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            
            # Log final metrics and model to MLflow
            if self.use_mlflow:
                mlflow.log_metrics({
                    "final_best_val_loss": self.best_val_loss,
                    "final_val_accuracy": self.val_accuracies[np.argmin(self.val_losses)],
                    "final_val_clv_mse": self.val_clv_mse[np.argmin(self.val_losses)],
                    "training_time_seconds": training_time,
                    "epochs_trained": epoch + 1
                })
                
                # Log PyTorch model artifact
                try:
                    mlflow.pytorch.log_model(self.model, "sequence_mtl_churn_clv_classifier")
                    print("Sequence MTL model logged successfully to MLflow Registry")
                except Exception as e:
                    print(f"Error logging model to MLflow: {e}")
        
        print(f"{'#'*80}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_churn_losses': self.train_churn_losses,
            'val_churn_losses': self.val_churn_losses,
            'train_clv_losses': self.train_clv_losses,
            'val_clv_losses': self.val_clv_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'train_clv_mse': self.train_clv_mse,
            'val_clv_mse': self.val_clv_mse,
            'training_time': training_time,
            'epochs_trained': epoch + 1
        }
    
    def save_model(self, path='sequence_mtl_model.pth'):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_churn_losses': self.train_churn_losses,
            'val_churn_losses': self.val_churn_losses,
            'train_clv_losses': self.train_clv_losses,
            'val_clv_losses': self.val_clv_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'train_clv_mse': self.train_clv_mse,
            'val_clv_mse': self.val_clv_mse,
            'churn_weight': self.churn_weight,
            'clv_weight': self.clv_weight
        }, path)
        print(f"Sequence MTL model saved to {path}")


def main():
    """
    Main Sequence MTL training function
    """
    import argparse
    parser = argparse.ArgumentParser(description='Train Sequence MTL Model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model-type', type=str, default='lstm', choices=['lstm', 'gru'], help='Model type')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow tracking')
    args = parser.parse_args()

    print("="*80)
    print("Sequence Multi-Task Learning (Churn + CLV) Training")
    print("="*80)
    
    # Hyperparameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.3
    SEQ_LENGTH = 4
    MIN_TRANSACTIONS = 5
    USE_LSTM = args.model_type == 'lstm'
    PATIENCE = 15
    WEIGHT_DECAY = 1e-5
    CHURN_WEIGHT = 1.0
    CLV_WEIGHT = 0.5
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using {'LSTM' if USE_LSTM else 'GRU'} backbone")
    
    # Load and preprocess sequence data
    print("\n" + "="*80)
    print("Sequence Data Preprocessing")
    print("="*80)
    data_path = os.path.join("data", "Business_Analytics_Dataset_10000_Rows.csv")
    if not os.path.exists(data_path):
        data_path = "Business_Analytics_Dataset_10000_Rows.csv" # Fallback
    
    try:
        (train_sequences, test_sequences, y_churn_train, y_churn_test,
         y_clv_train, y_clv_test, preprocessor, customer_ids, sequence_dates, targets) = load_and_preprocess_sequence_data(
            data_path, seq_length=SEQ_LENGTH, min_transactions=MIN_TRANSACTIONS
        )
        
        print(f"Training sequences shape: {train_sequences.shape}")
        print(f"Test sequences shape: {test_sequences.shape}")
        print(f"Sequence length: {SEQ_LENGTH}")
        print(f"Features per timestep: {train_sequences.shape[2]}")
        print(f"Feature names: {preprocessor.get_feature_names()}")
        
    except Exception as e:
        print(f"Error loading sequence data: {e}")
        print("Falling back to regular MTL training...")
        from train_mtl import main as mtl_main
        return mtl_main()
    
    # Create data loaders
    train_dataset = TensorDataset(train_sequences, y_churn_train, y_clv_train)
    test_dataset = TensorDataset(test_sequences, y_churn_test, y_clv_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create Sequence MTL model
    feature_dim = train_sequences.shape[2]
    model = create_sequence_mtl_model(
        feature_dim=feature_dim,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        use_lstm=USE_LSTM,
        dropout=DROPOUT
    )
    
    print("\n" + "="*80)
    print("Sequence MTL Model Architecture")
    print("="*80)
    print(model)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n" + "="*80)
    print("Sequence MTL Training")
    print("="*80)
    trainer = SequenceMTLTrainer(model, device, CHURN_WEIGHT, CLV_WEIGHT, use_mlflow=not args.no_mlflow)
    history = trainer.train(
        train_loader, test_loader, 
        epochs=EPOCHS, 
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Save model
    model_type = "lstm" if USE_LSTM else "gru"
    trainer.save_model(f'sequence_mtl_{model_type}_classifier.pth')
    
    # Save training history
    with open(f'sequence_mtl_{model_type}_training_history.json', 'w') as f:
        # Convert numpy types to Python types
        history_serializable = {}
        for k, v in history.items():
            if isinstance(v, list):
                history_serializable[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
            else:
                history_serializable[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        json.dump(history_serializable, f, indent=4)
    
    print("\nSequence MTL training complete! Model and history saved.")
    
    # Plot training curves
    plot_sequence_training_curves(history, model_type)


def plot_sequence_training_curves(history, model_type="lstm"):
    """
    Plot Sequence MTL training curves
    
    Args:
        history: Training history dictionary
        model_type: Type of model (lstm/gru)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_losses'], label='Train Total Loss', linewidth=2)
    axes[0, 0].plot(history['val_losses'], label='Val Total Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title(f'Total Loss ({model_type.upper()})', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Churn loss
    axes[0, 1].plot(history['train_churn_losses'], label='Train Churn Loss', linewidth=2)
    axes[0, 1].plot(history['val_churn_losses'], label='Val Churn Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Churn Loss')
    axes[0, 1].set_title('Churn Binary Cross-Entropy Loss', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # CLV loss
    axes[0, 2].plot(history['train_clv_losses'], label='Train CLV Loss', linewidth=2)
    axes[0, 2].plot(history['val_clv_losses'], label='Val CLV Loss', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('CLV Loss')
    axes[0, 2].set_title('CLV MSE Loss', fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Churn accuracy
    axes[1, 0].plot(history['train_accuracies'], label='Train Churn Acc', linewidth=2)
    axes[1, 0].plot(history['val_accuracies'], label='Val Churn Acc', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Churn Classification Accuracy', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # CLV MSE
    axes[1, 1].plot(history['train_clv_mse'], label='Train CLV MSE', linewidth=2)
    axes[1, 1].plot(history['val_clv_mse'], label='Val CLV MSE', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('CLV Regression MSE', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Remove empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(f'sequence_mtl_{model_type}_training_curves.png', dpi=300, bbox_inches='tight')
    print(f"Sequence MTL training curves saved to sequence_mtl_{model_type}_training_curves.png")
    plt.close()


if __name__ == "__main__":
    main()
