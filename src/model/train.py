"""
Training Script for MLP Churn Classifier
Implements training loop with Adam optimizer, learning rate scheduling, and cross-validation

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import json
import optuna
import mlflow
import mlflow.pytorch
import os

from src.model.model import create_model
from src.data.data_preprocessing import load_and_preprocess_data


class MLPTrainer:
    """
    Trainer class for MLP Classifier with comprehensive training features
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Set up MLflow tracking URI using absolute path so it works regardless of CWD
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mlflow.set_tracking_uri(f"sqlite:///{os.path.join(_project_root, 'mlflow.db')}")
        mlflow.set_experiment("Churn_Prediction")
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * batch_X.size(0)
            predicted = (outputs >= 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """
        Validate the model
        
        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function
            
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Statistics
                running_loss += loss.item() * batch_X.size(0)
                predicted = (outputs >= 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=100, learning_rate=0.001, 
              patience=15, weight_decay=1e-5):
        """
        Train the model with early stopping and learning rate scheduling
        
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
        print(f"\nTraining on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Adam optimizer with weight decay
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler - ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping
        epochs_no_improve = 0
        
        start_time = time.time()
        
        print(f"{'#'*70}")
        
        with mlflow.start_run(run_name="MLP_Churn_Training", nested=True):
            # Log Hyperparameters
            mlflow.log_params({
                "learning_rate": learning_rate,
                "epochs": epochs,
                "patience": patience,
                "weight_decay": weight_decay,
                "hidden_dims": str(self.model.hidden_dims),
                "dropout_rate": self.model.dropout_rate if hasattr(self.model, 'dropout_rate') else 0.0
            })
            
            for epoch in range(epochs):
                # Train
                train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
                
                # Validate
                val_loss, val_acc = self.validate(val_loader, criterion)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Save history
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
                
                # Get current LR for logging and printing
                current_lr = optimizer.param_groups[0]['lr']
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                          f"LR: {current_lr:.6f}")
                    
                # Log metrics to MLflow for the epoch
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": current_lr
                }, step=epoch)
                
                # Early stopping and best model saving
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
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
            mlflow.log_metrics({
                "final_best_val_loss": self.best_val_loss,
                "final_val_accuracy": self.val_accuracies[np.argmin(self.val_losses)], # Accuracy corresponding to best loss
                "training_time_seconds": training_time,
                "epochs_trained": epoch + 1
            })
            
            # Log the PyTorch model artifact
            try:
                mlflow.pytorch.log_model(self.model, "mlp_churn_classifier")
                print("Model logged successfully to MLflow Registry")
            except Exception as e:
                print(f"Error logging model to MLflow: {e}")
        
        print(f"{'#'*70}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'training_time': training_time,
            'epochs_trained': epoch + 1
        }
    
    def save_model(self, path='best_model.pth'):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='best_model.pth'):
        """Load a saved model"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        print(f"Model loaded from {path}")


def cross_validate(X, y, n_folds=5, hidden_dims=[128, 64, 32], 
                   dropout_rate=0.3, epochs=100, batch_size=32, device='cpu'):
    """
    Perform k-fold cross-validation
    
    Args:
        X: Feature tensor
        y: Target tensor
        n_folds: Number of folds
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout rate
        epochs: Training epochs
        batch_size: Batch size
        device: Device to train on
        
    Returns:
        dict: Cross-validation results
    """
    print(f"\n{'='*60}")
    print(f"Starting {n_folds}-Fold Cross-Validation")
    print(f"{'='*60}")
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nFold {fold + 1}/{n_folds}")
        print("-" * 40)
        
        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create model
        input_dim = X.shape[1]
        model = create_model(input_dim, hidden_dims, dropout_rate)
        
        # Train
        trainer = MLPTrainer(model, device)
        history = trainer.train(train_loader, val_loader, epochs=epochs, patience=10)
        
        # Evaluate
        val_loss, val_acc = trainer.validate(val_loader, nn.BCELoss())
        
        fold_results.append({
            'fold': fold + 1,
            'val_accuracy': val_acc,
            'val_loss': val_loss,
            'epochs_trained': history['epochs_trained']
        })
        
        print(f"Fold {fold + 1} Results: Accuracy = {val_acc:.4f}, Loss = {val_loss:.4f}")
    
    # Calculate average metrics
    avg_accuracy = np.mean([r['val_accuracy'] for r in fold_results])
    std_accuracy = np.std([r['val_accuracy'] for r in fold_results])
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results")
    print(f"{'='*60}")
    print(f"Average Accuracy: {avg_accuracy:.4f} (+/- {std_accuracy:.4f})")
    
    return {
        'fold_results': fold_results,
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy
    }


def optimize_hyperparameters(X, y, n_trials=20, device='cpu'):
    """
    Use Optuna to find the best hyperparameters for the MLP model
    """
    from sklearn.model_selection import train_test_split
    
    # Silence optuna INFO logs to keep console clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Split data once for all trials
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    input_dim = X.shape[1]
    
    def objective(trial):
        # Hyperparameter search space requested for Phase 1
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        architecture = trial.suggest_categorical(
            "hidden_dims",
            [
                "64,32",
                "128,64,32",
                "256,128,64",
            ],
        )
        hidden_dims = [int(dim) for dim in architecture.split(",")]
            
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create model
        model = create_model(input_dim, hidden_dims, dropout_rate)
        
        # Train
        trainer = MLPTrainer(model, device)
        # Use smaller epochs and patience for tuning
        history = trainer.train(train_loader, val_loader, epochs=40, 
                                learning_rate=lr, patience=5, 
                                weight_decay=weight_decay)
        
        return trainer.best_val_loss
        
    print(f"\n{'=' * 60}")
    print(f"Starting Optuna Hyperparameter Optimization ({n_trials} trials)")
    print(f"{'=' * 60}")
    
    study = optuna.create_study(direction="minimize")
    
    with mlflow.start_run(run_name="Optuna_Optimization"):
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print("\nOptimization Complete!")
        print(f"Best Validation Loss: {study.best_trial.value:.4f}")
        print("Best Hyperparameters Found:")
        
        best_params = study.best_trial.params
        for key, value in best_params.items():
            print(f"  {key}: {value}")
            
        # Log Best Optuna results to the parent MLflow run
        mlflow.log_params(best_params)
        mlflow.log_metric("best_optuna_val_loss", study.best_trial.value)
        
    best_params['hidden_dims'] = [int(dim) for dim in best_params['hidden_dims'].split(',')]
        
    return best_params


def main():
    """
    Main training function
    """
    print("="*60)
    print("MLP Churn Classifier Training")
    print("="*60)
    
    # Base hyperparameters (overridden by Optuna best trial)
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    HIDDEN_DIMS = [128, 64, 32]
    DROPOUT_RATE = 0.3
    PATIENCE = 15
    WEIGHT_DECAY = 1e-5
    N_TUNING_TRIALS = 20
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\n" + "="*60)
    print("Data Preprocessing")
    print("="*60)
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()

    print("\n" + "="*60)
    print("Hyperparameter Optimization (Optuna)")
    print("="*60)
    best_params = optimize_hyperparameters(
        X_train,
        y_train,
        n_trials=N_TUNING_TRIALS,
        device=device,
    )

    BATCH_SIZE = best_params.get('batch_size', BATCH_SIZE)
    LEARNING_RATE = best_params.get('learning_rate', LEARNING_RATE)
    DROPOUT_RATE = best_params.get('dropout_rate', DROPOUT_RATE)
    WEIGHT_DECAY = best_params.get('weight_decay', WEIGHT_DECAY)
    HIDDEN_DIMS = best_params.get('hidden_dims', HIDDEN_DIMS)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    input_dim = X_train.shape[1]
    model = create_model(input_dim, HIDDEN_DIMS, DROPOUT_RATE)
    
    print("\n" + "="*60)
    print("Model Architecture")
    print("="*60)
    print(model)
    print(f"\nModel Info: {model.get_model_info()}")
    
    # Train model
    print("\n" + "="*60)
    print("Training")
    print("="*60)
    trainer = MLPTrainer(model, device)
    history = trainer.train(
        train_loader, test_loader, 
        epochs=EPOCHS, 
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Save model
    trainer.save_model('mlp_churn_classifier_final.pth')
    
    # Save training history
    with open('training_history.json', 'w') as f:
        # Convert numpy types to Python types
        history_serializable = {k: [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                    for v in vals] if isinstance(vals, list) else vals 
                               for k, vals in history.items()}
        json.dump(history_serializable, f, indent=4)
    
    print("\nTraining complete! Model and history saved.")
    
    # Plot training curves
    plot_training_curves(history)
    

def plot_training_curves(history):
    """
    Plot training and validation curves
    
    Args:
        history: Training history dictionary
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(history['train_losses'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_losses'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(history['train_accuracies'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracies'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved to training_curves.png")
    plt.close()


if __name__ == "__main__":
    main()
