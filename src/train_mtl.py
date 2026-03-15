"""
Multi-Task Learning Training Script for Churn + CLV Prediction
Implements MTL training with shared trunk and dual heads

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

from model import create_mtl_model
from data_preprocessing import load_and_preprocess_data


class MTLTrainer:
    """
    Trainer class for Multi-Task Learning (Churn + CLV)
    """
    
    def __init__(self, model, device='cpu', churn_weight=1.0, clv_weight=1.0, **kwargs):
        """
        Initialize MTL trainer
        
        Args:
            model: PyTorch MTL model
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
                mlflow.set_experiment("Churn_Prediction_MTL")
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
        Validate the model
        
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
                clv_predictions.extend(clv_pred.detach().cpu().numpy())
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
        Train the MTL model with early stopping
        
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
        print(f"\nTraining MTL model on {self.device}")
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

        ctx = mlflow.start_run(run_name="MTL_Churn_CLV_Training", nested=True) if self.use_mlflow else DummyContextManager()
        
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
                    "model_type": "MTL",
                    "hidden_dims": str(self.model.hidden_dims),
                    "dropout_rate": self.model.dropout_rate
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
                          f"CLV MSE: {train_clv_mse:.2f}, Val: {val_clv_mse:.2f} | "
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
                
                # Log the PyTorch model artifact
                try:
                    mlflow.pytorch.log_model(self.model, "mtl_churn_clv_classifier")
                    print("MTL model logged successfully to MLflow Registry")
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
    
    def save_model(self, path='mtl_model.pth'):
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
        print(f"MTL model saved to {path}")


def create_clv_targets(customer_df):
    """
    Create CLV targets from customer data
    
    Args:
        customer_df: DataFrame with customer features
        
    Returns:
        np.array: CLV values
    """
    # Use total profit as a proxy for CLV
    # In a real scenario, this could be predicted future value
    clv = customer_df['total_profit'].values
    
    # Apply log transformation to handle skewness
    clv = np.log1p(clv)  # log1p to handle zero values
    
    # Normalize to [0, 1] range for better training
    clv = (clv - clv.min()) / (clv.max() - clv.min())
    
    return clv


def prepare_mtl_data(data_path, test_size=0.2, random_state=42):
    """
    Prepare data for MTL training
    
    Args:
        data_path: Path to data
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        tuple: (X_train, X_test, y_churn_train, y_churn_test, y_clv_train, y_clv_test, preprocessor)
    """
    from data_preprocessing import ChurnDataPreprocessor
    
    # Load and preprocess data
    preprocessor = ChurnDataPreprocessor(data_path)
    preprocessor.load_data()
    customer_df = preprocessor.engineer_churn_label()
    
    # Create CLV targets
    clv_targets = create_clv_targets(customer_df)
    
    # Prepare features
    X, y_churn = preprocessor.prepare_features()
    
    # Split and scale features
    X_train, X_test, y_churn_train, y_churn_test = preprocessor.split_and_scale(
        X, y_churn, test_size, random_state
    )
    
    # Split CLV targets using same indices
    from sklearn.model_selection import train_test_split
    clv_train, clv_test = train_test_split(
        clv_targets, test_size=test_size, random_state=random_state
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_churn_train_tensor = torch.FloatTensor(y_churn_train).unsqueeze(1)
    y_churn_test_tensor = torch.FloatTensor(y_churn_test).unsqueeze(1)
    y_clv_train_tensor = torch.FloatTensor(clv_train).unsqueeze(1)
    y_clv_test_tensor = torch.FloatTensor(clv_test).unsqueeze(1)
    
    return (X_train_tensor, X_test_tensor, 
            y_churn_train_tensor, y_churn_test_tensor,
            y_clv_train_tensor, y_clv_test_tensor,
            preprocessor)


def main():
    """
    Main MTL training function
    """
    import argparse
    parser = argparse.ArgumentParser(description='Train MTL Model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow tracking')
    args = parser.parse_args()

    print("="*80)
    print("Multi-Task Learning (Churn + CLV) Training")
    print("="*80)
    
    # Hyperparameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    HIDDEN_DIMS = [128, 64, 32]
    DROPOUT_RATE = 0.3
    PATIENCE = 15
    WEIGHT_DECAY = 1e-5
    CHURN_WEIGHT = 1.0
    CLV_WEIGHT = 0.5  # Give less weight to CLV initially
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\n" + "="*80)
    print("Data Preprocessing for MTL")
    print("="*80)
    data_path = os.path.join("data", "Business_Analytics_Dataset_10000_Rows.csv")
    if not os.path.exists(data_path):
        data_path = "Business_Analytics_Dataset_10000_Rows.csv"  # Fallback
        
    (X_train, X_test, y_churn_train, y_churn_test,
     y_clv_train, y_clv_test, preprocessor) = prepare_mtl_data(data_path)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Churn targets - Train: {y_churn_train.shape}, Test: {y_churn_test.shape}")
    print(f"CLV targets - Train: {y_clv_train.shape}, Test: {y_clv_test.shape}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_churn_train, y_clv_train)
    test_dataset = TensorDataset(X_test, y_churn_test, y_clv_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create MTL model
    input_dim = X_train.shape[1]
    model = create_mtl_model(input_dim, HIDDEN_DIMS, DROPOUT_RATE)
    
    print("\n" + "="*80)
    print("MTL Model Architecture")
    print("="*80)
    print(model)
    print(f"\nModel Info: {model.get_model_info()}")
    
    # Train model
    print("\n" + "="*80)
    print("MTL Training")
    print("="*80)
    trainer = MTLTrainer(model, device, CHURN_WEIGHT, CLV_WEIGHT, use_mlflow=not args.no_mlflow)
    history = trainer.train(
        train_loader, test_loader, 
        epochs=EPOCHS, 
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Save model
    trainer.save_model('mtl_churn_clv_classifier.pth')
    
    # Save training history
    with open('mtl_training_history.json', 'w') as f:
        # Convert numpy types to Python types
        history_serializable = {}
        for k, v in history.items():
            if isinstance(v, list):
                history_serializable[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
            else:
                history_serializable[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        json.dump(history_serializable, f, indent=4)
    
    print("\nMTL training complete! Model and history saved.")
    
    # Plot training curves
    plot_mtl_training_curves(history)


def plot_mtl_training_curves(history):
    """
    Plot MTL training curves
    
    Args:
        history: Training history dictionary
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_losses'], label='Train Total Loss', linewidth=2)
    axes[0, 0].plot(history['val_losses'], label='Val Total Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss', fontweight='bold')
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
    plt.savefig('mtl_training_curves.png', dpi=300, bbox_inches='tight')
    print("MTL training curves saved to mtl_training_curves.png")
    plt.close()


if __name__ == "__main__":
    main()
