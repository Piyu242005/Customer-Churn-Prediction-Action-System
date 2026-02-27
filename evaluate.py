"""
Model Evaluation and Visualization Script
Comprehensive evaluation metrics and visualizations for the trained MLP classifier
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, classification_report,
    precision_recall_curve, average_precision_score
)
from model import MLPClassifier
from data_preprocessing import load_and_preprocess_data


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained PyTorch model
        X_test: Test features tensor
        y_test: Test labels tensor
        threshold: Classification threshold
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    
    with torch.no_grad():
        y_pred_proba = model(X_test).numpy()
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Convert tensors to numpy
    y_test_np = y_test.numpy().flatten()
    y_pred_flat = y_pred.flatten()
    y_pred_proba_flat = y_pred_proba.flatten()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test_np, y_pred_flat),
        'precision': precision_score(y_test_np, y_pred_flat, zero_division=0),
        'recall': recall_score(y_test_np, y_pred_flat, zero_division=0),
        'f1_score': f1_score(y_test_np, y_pred_flat, zero_division=0),
        'roc_auc': roc_auc_score(y_test_np, y_pred_proba_flat),
        'avg_precision': average_precision_score(y_test_np, y_pred_proba_flat)
    }
    
    return metrics, y_pred_flat, y_pred_proba_flat


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Active', 'Churned'], 
                yticklabels=['Active', 'Churned'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - MLP Churn Classifier', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.7, f'({cm[i, j]/total*100:.1f}%)', 
                    ha='center', va='center', color='gray', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'MLP Classifier (AUC = {roc_auc:.4f})', color='#e74c3c')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'MLP Classifier (AP = {avg_precision:.4f})', color='#2ecc71')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    
    plt.show()


def plot_threshold_analysis(y_true, y_pred_proba, save_path=None):
    """
    Analyze model performance across different classification thresholds
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save figure
    """
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        accuracies.append(accuracy_score(y_true, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
    plt.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Default Threshold')
    plt.xlabel('Classification Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Across Different Thresholds', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Threshold analysis saved to {save_path}")
    
    plt.show()


def generate_evaluation_report(metrics, y_true, y_pred):
    """
    Generate comprehensive evaluation report
    
    Args:
        metrics: Dictionary of evaluation metrics
        y_true: True labels
        y_pred: Predicted labels
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION REPORT")
    print("="*60)
    
    print("\nOverall Performance Metrics:")
    print("-" * 60)
    print(f"Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1-Score:          {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['avg_precision']:.4f}")
    
    print("\n" + "-" * 60)
    print("Detailed Classification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=['Active', 'Churned']))
    
    print("-" * 60)
    print("Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_true, y_pred)
    print(f"True Negatives:  {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives:  {cm[1, 1]}")
    
    print("\n" + "="*60)


def main():
    """
    Main evaluation function
    """
    print("="*60)
    print("MLP CHURN CLASSIFIER - EVALUATION")
    print("="*60)
    
    # Load model
    print("\nLoading trained model...")
    checkpoint = torch.load('mlp_churn_classifier.pth')
    
    # Recreate model architecture
    input_dim = 16  # Adjust based on your features
    model = MLPClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded successfully!")
    
    # Load data
    print("\nLoading data...")
    data_path = "Business_Analytics_Dataset_10000_Rows.csv"
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_path)
    
    print("✓ Data loaded successfully!")
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Generate report
    y_test_np = y_test.numpy().flatten()
    generate_evaluation_report(metrics, y_test_np, y_pred)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test_np, y_pred, save_path='confusion_matrix.png')
    plot_roc_curve(y_test_np, y_pred_proba, save_path='roc_curve.png')
    plot_precision_recall_curve(y_test_np, y_pred_proba, save_path='precision_recall_curve.png')
    plot_threshold_analysis(y_test_np, y_pred_proba, save_path='threshold_analysis.png')
    
    print("\n" + "="*60)
    print("✓ Evaluation complete! All visualizations saved.")
    print("="*60)


if __name__ == "__main__":
    main()
