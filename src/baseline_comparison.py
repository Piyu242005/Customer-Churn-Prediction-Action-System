"""
Baseline Model Comparison Module
Compares MLP classifier against various machine learning baselines

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, log_loss
)
import warnings
warnings.filterwarnings('ignore')


class ModelComparator:
    """
    Compare multiple models for churn prediction
    """
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize comparator
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
        """
        # Convert PyTorch tensors to numpy if needed
        self.X_train = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
        self.X_test = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
        self.y_train = y_train.numpy().flatten() if isinstance(y_train, torch.Tensor) else y_train.flatten()
        self.y_test = y_test.numpy().flatten() if isinstance(y_test, torch.Tensor) else y_test.flatten()
        
        self.models = {}
        self.results = {}
    
    def define_baseline_models(self):
        """
        Define baseline models to compare
        
        Returns:
            dict: Dictionary of model instances
        """
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            
            'Random Forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                eval_metric='logloss'
            ),
            
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=10,
                random_state=42
            ),
            
            'SVM (RBF)': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            
            'Naive Bayes': GaussianNB(),
            
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=42
            )
        }
        
        print(f"Defined {len(self.models)} baseline models for comparison")
        return self.models
    
    def train_and_evaluate_model(self, name, model):
        """
        Train and evaluate a single model
        
        Args:
            name (str): Model name
            model: Model instance
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\nTraining {name}...")
        
        # Train
        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        
        # Predict
        start_time = time.time()
        y_pred = model.predict(self.X_test)
        
        # Get probabilities if available
        try:
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        except:
            y_pred_proba = y_pred  # For models without predict_proba
        
        inference_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'model': name,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
            'train_time': train_time,
            'inference_time': inference_time
        }
        
        # Add ROC-AUC if probabilities are available
        try:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(self.y_test, y_pred_proba)
        except:
            metrics['roc_auc'] = None
            metrics['avg_precision'] = None
            
        roc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A"
        
        print(f"✓ {name}: Accuracy={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1_score']:.4f}, "
              f"ROC-AUC={roc_str}")
        
        return metrics
    
    def compare_all_models(self):
        """
        Train and evaluate all baseline models
        
        Returns:
            pd.DataFrame: Comparison results
        """
        print("\n" + "="*60)
        print("BASELINE MODEL COMPARISON")
        print("="*60)
        
        if not self.models:
            self.define_baseline_models()
        
        results_list = []
        
        for name, model in self.models.items():
            try:
                metrics = self.train_and_evaluate_model(name, model)
                results_list.append(metrics)
                self.results[name] = metrics
            except Exception as e:
                print(f"✗ {name} failed: {e}")
        
        # Create DataFrame
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        return results_df
    
    def add_mlp_results(self, mlp_metrics):
        """
        Add MLP results to comparison
        
        Args:
            mlp_metrics (dict): MLP evaluation metrics
        """
        self.results['MLP (Neural Network)'] = mlp_metrics
        print("✓ MLP results added to comparison")
    
    def plot_model_comparison(self, metrics=['accuracy', 'precision', 'recall', 'f1_score'], 
                             save_path=None):
        """
        Plot comparison of models across multiple metrics
        
        Args:
            metrics (list): List of metrics to plot
            save_path (str): Path to save figure
        """
        if not self.results:
            print("Error: No results available. Run compare_all_models first.")
            return
        
        # Create DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):
            if metric in results_df.columns:
                # Drop None values
                plot_data = results_df[[metric]].dropna()
                
                # Sort by metric
                plot_data = plot_data.sort_values(metric, ascending=True)
                
                # Color: MLP in red, others in blue
                colors = ['#e74c3c' if 'MLP' in idx else '#3498db' for idx in plot_data.index]
                
                axes[i].barh(range(len(plot_data)), plot_data[metric], color=colors)
                axes[i].set_yticks(range(len(plot_data)))
                axes[i].set_yticklabels(plot_data.index, fontsize=10)
                axes[i].set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
                axes[i].set_title(f'Model Comparison: {metric.replace("_", " ").title()}', 
                                fontsize=14, fontweight='bold')
                axes[i].grid(axis='x', alpha=0.3)
                
                # Add value labels
                for j, v in enumerate(plot_data[metric]):
                    axes[i].text(v + 0.01, j, f'{v:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_comparison(self, models_dict=None, save_path=None):
        """
        Plot ROC curves for multiple models
        
        Args:
            models_dict (dict): Dictionary of trained models (optional)
            save_path (str): Path to save figure
        """
        plt.figure(figsize=(10, 8))
        
        if models_dict:
            for name, model in models_dict.items():
                try:
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                    
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                    auc = roc_auc_score(self.y_test, y_pred_proba)
                    
                    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', linewidth=2)
                except:
                    pass
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier', alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC comparison saved to {save_path}")
        
        plt.show()
    
    def generate_comparison_report(self, save_path=None):
        """
        Generate comprehensive comparison report
        
        Args:
            save_path (str): Path to save report
        """
        if not self.results:
            print("Error: No results available.")
            return
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON REPORT")
        print("="*60)
        
        print("\nRanking by Accuracy:")
        print("-"*60)
        print(f"{'Rank':<6} {'Model':<25} {'Accuracy':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-"*60)
        
        for i, (model_name, row) in enumerate(results_df.iterrows(), 1):
            roc_auc_str = f"{row['roc_auc']:.4f}" if row['roc_auc'] is not None else "N/A"
            print(f"{i:<6} {model_name:<25} {row['accuracy']:.4f}     "
                  f"{row['f1_score']:.4f}     {roc_auc_str}")
        
        # Best model
        best_model = results_df.index[0]
        best_acc = results_df.iloc[0]['accuracy']
        
        print("\n" + "-"*60)
        print(f"🏆 Best Model: {best_model} (Accuracy: {best_acc:.4f})")
        print("-"*60)
        
        # Detailed metrics
        print("\nDetailed Performance Metrics:")
        print("-"*60)
        print(results_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string())
        
        # Training time comparison
        print("\n" + "-"*60)
        print("Training Time Comparison:")
        print("-"*60)
        for model_name, row in results_df.iterrows():
            print(f"{model_name:<25} {row['train_time']:.2f}s")
        
        print("\n" + "="*60)
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write("MODEL COMPARISON REPORT\n")
                f.write("="*60 + "\n\n")
                f.write(results_df.to_string())
                f.write(f"\n\nBest Model: {best_model}\n")
                f.write(f"Best Accuracy: {best_acc:.4f}\n")
            print(f"✓ Comparison report saved to {save_path}")
        
        return results_df
    
    def plot_metric_heatmap(self, save_path=None):
        """
        Plot heatmap of all metrics across models
        
        Args:
            save_path (str): Path to save figure
        """
        if not self.results:
            print("Error: No results available.")
            return
        
        results_df = pd.DataFrame(self.results).T
        
        # Select only metric columns
        metric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metric_cols = [col for col in metric_cols if col in results_df.columns]
        
        plot_data = results_df[metric_cols].dropna().astype(float)
        
        plt.figure(figsize=(10, max(6, len(plot_data) * 0.5)))
        sns.heatmap(plot_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
        plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Models', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metric heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_training_time_vs_performance(self, save_path=None):
        """
        Plot training time vs performance (accuracy)
        
        Args:
            save_path (str): Path to save figure
        """
        if not self.results:
            print("Error: No results available.")
            return
        
        results_df = pd.DataFrame(self.results).T
        
        plt.figure(figsize=(10, 7))
        
        for model_name, row in results_df.iterrows():
            color = '#e74c3c' if 'MLP' in model_name else '#3498db'
            size = 200 if 'MLP' in model_name else 100
            plt.scatter(row['train_time'], row['accuracy'], s=size, alpha=0.7, 
                       color=color, edgecolors='black', linewidth=1.5)
            plt.annotate(model_name, (row['train_time'], row['accuracy']), 
                        fontsize=9, ha='right', va='bottom')
        
        plt.xlabel('Training Time (seconds)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Training Time vs Model Performance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training time vs performance plot saved to {save_path}")
        
        plt.show()
    
    def create_ensemble(self, top_n=3):
        """
        Create a voting ensemble from top N models
        
        Args:
            top_n (int): Number of top models to include
            
        Returns:
            VotingClassifier: Ensemble model
        """
        from sklearn.ensemble import VotingClassifier
        
        # Get top N models by accuracy
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('accuracy', ascending=False)
        top_models = results_df.head(top_n).index.tolist()
        
        # Filter out MLP (can't be used in sklearn ensemble)
        top_models = [m for m in top_models if 'MLP' not in m][:top_n]
        
        print(f"\nCreating ensemble from top {len(top_models)} models:")
        for model in top_models:
            print(f"  • {model}")
        
        # Create ensemble
        estimators = [(name, self.models[name]) for name in top_models if name in self.models]
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Train ensemble
        print("\nTraining ensemble...")
        start_time = time.time()
        ensemble.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        
        # Evaluate ensemble
        y_pred = ensemble.predict(self.X_test)
        y_pred_proba = ensemble.predict_proba(self.X_test)[:, 1]
        
        ensemble_metrics = {
            'model': f'Ensemble (Top {len(top_models)})',
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'avg_precision': average_precision_score(self.y_test, y_pred_proba),
            'train_time': train_time,
            'inference_time': 0  # Not measured separately
        }
        
        self.results[f'Ensemble (Top {len(top_models)})'] = ensemble_metrics
        
        print(f"\n✓ Ensemble: Accuracy={ensemble_metrics['accuracy']:.4f}, "
              f"F1={ensemble_metrics['f1_score']:.4f}, "
              f"ROC-AUC={ensemble_metrics['roc_auc']:.4f}")
        
        return ensemble, ensemble_metrics


def evaluate_mlp_model(model, X_test, y_test, device='cpu'):
    """
    Evaluate MLP model for comparison
    
    Args:
        model: Trained MLP model
        X_test: Test features (tensor)
        y_test: Test labels (tensor)
        device (str): Device
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.FloatTensor(X_test)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.FloatTensor(y_test)
    
    X_test = X_test.to(device)
    y_test_np = y_test.numpy().flatten()
    
    with torch.no_grad():
        y_pred_proba = model(X_test).cpu().numpy().flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
    
    metrics = {
        'model': 'MLP (Neural Network)',
        'accuracy': accuracy_score(y_test_np, y_pred),
        'precision': precision_score(y_test_np, y_pred, zero_division=0),
        'recall': recall_score(y_test_np, y_pred, zero_division=0),
        'f1_score': f1_score(y_test_np, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test_np, y_pred_proba),
        'avg_precision': average_precision_score(y_test_np, y_pred_proba),
        'train_time': 0,  # Load from training history if available
        'inference_time': 0
    }
    
    return metrics


def run_comprehensive_comparison(X_train, X_test, y_train, y_test, 
                                 mlp_model=None, device='cpu',
                                 save_dir='plots/comparison/'):
    """
    Run comprehensive model comparison pipeline
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        mlp_model: Trained MLP model (optional)
        device (str): Device for MLP
        save_dir (str): Directory to save plots
        
    Returns:
        tuple: (comparator, results_df)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*60)
    
    # Initialize comparator
    comparator = ModelComparator(X_train, X_test, y_train, y_test)
    
    # Define and train baseline models
    comparator.define_baseline_models()
    results_df = comparator.compare_all_models()
    
    # Add MLP if provided
    if mlp_model is not None:
        print("\nEvaluating MLP model...")
        mlp_metrics = evaluate_mlp_model(mlp_model, X_test, y_test, device)
        comparator.add_mlp_results(mlp_metrics)
        
        # Refresh results
        results_df = pd.DataFrame(comparator.results).T
        results_df = results_df.sort_values('accuracy', ascending=False)
    
    # Generate visualizations
    print("\nGenerating comparison visualizations...")
    
    comparator.plot_model_comparison(
        metrics=['accuracy', 'precision', 'recall', 'f1_score'],
        save_path=f'{save_dir}model_comparison.png'
    )
    
    comparator.plot_metric_heatmap(
        save_path=f'{save_dir}metric_heatmap.png'
    )
    
    comparator.plot_training_time_vs_performance(
        save_path=f'{save_dir}time_vs_performance.png'
    )
    
    comparator.plot_roc_comparison(
        models_dict=comparator.models,
        save_path=f'{save_dir}roc_comparison.png'
    )
    
    # Generate report
    comparator.generate_comparison_report(
        save_path=f'{save_dir}comparison_report.txt'
    )
    
    # Create ensemble
    print("\n" + "="*60)
    ensemble, ensemble_metrics = comparator.create_ensemble(top_n=3)
    
    print("\n" + "="*60)
    print("✓ COMPREHENSIVE COMPARISON COMPLETE")
    print(f"✓ All results saved to {save_dir}")
    print("="*60)
    
    return comparator, results_df


def main():
    """
    Main comparison script
    """
    from src.data.data_preprocessing import load_and_preprocess_data
    from src.model.model import MLPClassifier
    
    print("="*60)
    print("BASELINE MODEL COMPARISON")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data_path = "Business_Analytics_Dataset_10000_Rows.csv"
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_path)
    print("✓ Data loaded")
    
    # Load MLP model
    print("\nLoading MLP model...")
    try:
        checkpoint = torch.load('mlp_churn_classifier.pth')
        model = MLPClassifier(input_dim=16, hidden_dims=[128, 64, 32], dropout_rate=0.3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✓ MLP model loaded")
    except:
        print("⚠ MLP model not found, will compare baseline models only")
        model = None
    
    # Run comparison
    comparator, results_df = run_comprehensive_comparison(
        X_train, X_test, y_train, y_test,
        mlp_model=model
    )
    
    print("\n✓ Model comparison complete!")
    print("\nTop 3 Models:")
    print(results_df[['accuracy', 'f1_score', 'roc_auc']].head(3))


if __name__ == "__main__":
    main()
