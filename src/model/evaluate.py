"""
Model Evaluation and Visualization Script
Comprehensive evaluation metrics and visualizations for the trained MLP classifier

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
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
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, log_loss
)
from sklearn.calibration import calibration_curve, CalibrationDisplay
from src.model.model import MLPClassifier
from src.data.data_preprocessing import load_and_preprocess_data


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


def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, save_path=None):
    """
    Plot calibration curve to assess prediction reliability
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration
        save_path: Path to save figure
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy='uniform')
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='MLP Classifier')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly Calibrated', alpha=0.5)
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve (Reliability Diagram)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration curve saved to {save_path}")
    
    plt.show()


def calculate_business_metrics(y_true, y_pred, y_pred_proba, 
                               churn_cost=500, retention_cost=50, 
                               success_rate=0.3):
    """
    Calculate business impact metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        churn_cost: Cost of losing a customer
        retention_cost: Cost of retention campaign per customer
        success_rate: Success rate of retention campaign
        
    Returns:
        dict: Business metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Without model: All customers churn or no intervention
    baseline_cost = y_true.sum() * churn_cost
    
    # With model:
    # - TP: Correctly predicted churn → retention campaign → some saved
    saved_customers = tp * success_rate
    retention_campaign_cost = (tp + fp) * retention_cost
    unavoidable_churn = fn * churn_cost  # Missed churners
    unsuccessful_retention = (tp * (1 - success_rate)) * churn_cost
    
    total_cost_with_model = retention_campaign_cost + unavoidable_churn + unsuccessful_retention
    
    # Savings
    cost_savings = baseline_cost - total_cost_with_model
    roi = (cost_savings / retention_campaign_cost) * 100 if retention_campaign_cost > 0 else 0
    
    business_metrics = {
        'baseline_cost': baseline_cost,
        'total_cost_with_model': total_cost_with_model,
        'cost_savings': cost_savings,
        'roi_percentage': roi,
        'customers_saved': saved_customers,
        'retention_campaigns': tp + fp,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }
    
    return business_metrics


def plot_business_impact(business_metrics, save_path=None):
    """
    Visualize business impact of the model
    
    Args:
        business_metrics: Dictionary of business metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cost comparison
    categories = ['Without Model\n(Baseline)', 'With Model\n(Predicted)']
    costs = [business_metrics['baseline_cost'], business_metrics['total_cost_with_model']]
    colors = ['#e74c3c', '#27ae60']
    
    axes[0].bar(categories, costs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Total Cost ($)', fontsize=12)
    axes[0].set_title('Cost Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, cost in enumerate(costs):
        axes[0].text(i, cost, f'${cost:,.0f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
    
    # Savings breakdown
    savings = business_metrics['cost_savings']
    roi = business_metrics['roi_percentage']
    
    axes[1].text(0.5, 0.7, f'Cost Savings', ha='center', fontsize=16, fontweight='bold')
    axes[1].text(0.5, 0.5, f'${savings:,.0f}', ha='center', fontsize=32, 
                fontweight='bold', color='#27ae60')
    axes[1].text(0.5, 0.3, f'ROI: {roi:.1f}%', ha='center', fontsize=18, color='#2ecc71')
    axes[1].text(0.5, 0.15, f'Customers Saved: {business_metrics["customers_saved"]:.0f}', 
                ha='center', fontsize=12)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')
    axes[1].set_title('Business Impact', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Business impact plot saved to {save_path}")
    
    plt.show()


def calculate_advanced_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate advanced evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        dict: Advanced metrics
    """
    advanced_metrics = {
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'log_loss': log_loss(y_true, y_pred_proba)
    }
    
    # Calculate specificity and sensitivity
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    advanced_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    advanced_metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    advanced_metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    advanced_metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return advanced_metrics


def plot_error_analysis(y_true, y_pred, y_pred_proba, save_path=None):
    """
    Analyze prediction errors
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save figure
    """
    # Find misclassified samples
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]
    
    # Separate false positives and false negatives
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    fp_proba = y_pred_proba[fp_mask]
    fn_proba = y_pred_proba[fn_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error distribution by prediction confidence
    axes[0].hist(y_pred_proba[errors], bins=20, alpha=0.7, color='#e74c3c', edgecolor='black')
    axes[0].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Number of Errors', fontsize=12)
    axes[0].set_title('Error Distribution by Confidence', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # FP vs FN comparison
    error_types = ['False Positives\n(Predicted Churn)', 'False Negatives\n(Missed Churn)']
    error_counts = [fp_mask.sum(), fn_mask.sum()]
    colors = ['#f39c12', '#e74c3c']
    
    bars = axes[1].bar(error_types, error_counts, color=colors, alpha=0.7, 
                       edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Error Type Analysis', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, error_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, count, f'{count}', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error analysis saved to {save_path}")
    
    plt.show()
    
    # Print error statistics
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    print(f"Total Errors: {errors.sum()} ({errors.sum()/len(y_true)*100:.2f}%)")
    print(f"False Positives: {fp_mask.sum()} (Active predicted as Churned)")
    print(f"False Negatives: {fn_mask.sum()} (Churned predicted as Active)")
    print(f"\nFalse Positive Avg Confidence: {fp_proba.mean():.4f}" if len(fp_proba) > 0 else "")
    print(f"False Negative Avg Confidence: {(1-fn_proba).mean():.4f}" if len(fn_proba) > 0 else "")
    print("="*60)


def plot_prediction_distribution(y_true, y_pred_proba, save_path=None):
    """
    Plot distribution of predicted probabilities for each class
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    # Separate by true class
    active_proba = y_pred_proba[y_true == 0]
    churned_proba = y_pred_proba[y_true == 1]
    
    plt.hist(active_proba, bins=30, alpha=0.6, label='Active Customers (True)', 
            color='#3498db', edgecolor='black')
    plt.hist(churned_proba, bins=30, alpha=0.6, label='Churned Customers (True)', 
            color='#e74c3c', edgecolor='black')
    
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('Predicted Churn Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Distribution by True Class', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction distribution saved to {save_path}")
    
    plt.show()


def generate_comprehensive_report(metrics, advanced_metrics, business_metrics, 
                                 y_true, y_pred, save_path=None):
    """
    Generate comprehensive evaluation report with all metrics
    
    Args:
        metrics: Standard metrics dict
        advanced_metrics: Advanced metrics dict
        business_metrics: Business impact metrics
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save report
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION REPORT")
    print("="*60)
    
    # Standard metrics
    print("\n📊 Standard Performance Metrics:")
    print("-" * 60)
    print(f"Accuracy:              {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:             {metrics['precision']:.4f}")
    print(f"Recall (Sensitivity):  {metrics['recall']:.4f}")
    print(f"F1-Score:              {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:               {metrics['roc_auc']:.4f}")
    print(f"Average Precision:     {metrics['avg_precision']:.4f}")
    
    # Advanced metrics
    print("\n📈 Advanced Metrics:")
    print("-" * 60)
    print(f"Matthews Corr. Coef:   {advanced_metrics['matthews_corrcoef']:.4f}")
    print(f"Cohen's Kappa:         {advanced_metrics['cohen_kappa']:.4f}")
    print(f"Balanced Accuracy:     {advanced_metrics['balanced_accuracy']:.4f}")
    print(f"Log Loss:              {advanced_metrics['log_loss']:.4f}")
    print(f"Specificity:           {advanced_metrics['specificity']:.4f}")
    print(f"Sensitivity:           {advanced_metrics['sensitivity']:.4f}")
    print(f"False Positive Rate:   {advanced_metrics['false_positive_rate']:.4f}")
    print(f"False Negative Rate:   {advanced_metrics['false_negative_rate']:.4f}")
    
    # Business metrics
    print("\n💼 Business Impact Analysis:")
    print("-" * 60)
    print(f"Baseline Cost (No Model):      ${business_metrics['baseline_cost']:,.2f}")
    print(f"Cost with Model:               ${business_metrics['total_cost_with_model']:,.2f}")
    print(f"💰 Cost Savings:               ${business_metrics['cost_savings']:,.2f}")
    print(f"📈 ROI:                        {business_metrics['roi_percentage']:.1f}%")
    print(f"✅ Customers Saved:            {business_metrics['customers_saved']:.0f}")
    print(f"📧 Retention Campaigns:        {business_metrics['retention_campaigns']}")
    
    # Classification details
    print("\n" + "-" * 60)
    print("Detailed Classification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=['Active', 'Churned']))
    
    print("="*60)
    
    # Save report
    if save_path:
        with open(save_path, 'w') as f:
            f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("STANDARD METRICS:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\nADVANCED METRICS:\n")
            for key, value in advanced_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\nBUSINESS IMPACT:\n")
            for key, value in business_metrics.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"\n✓ Comprehensive report saved to {save_path}")


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
    
    # Advanced evaluations
    print("\nGenerating advanced evaluations...")
    plot_calibration_curve(y_test_np, y_pred_proba, save_path='calibration_curve.png')
    plot_prediction_distribution(y_test_np, y_pred_proba, save_path='prediction_distribution.png')
    plot_error_analysis(y_test_np, y_pred, y_pred_proba, save_path='error_analysis.png')
    
    # Calculate advanced metrics
    advanced_metrics = calculate_advanced_metrics(y_test_np, y_pred, y_pred_proba)
    
    # Calculate business metrics
    business_metrics = calculate_business_metrics(
        y_test_np, y_pred, y_pred_proba,
        churn_cost=500,  # $500 cost per churned customer
        retention_cost=50,  # $50 cost per retention campaign
        success_rate=0.3  # 30% success rate
    )
    plot_business_impact(business_metrics, save_path='business_impact.png')
    
    # Generate comprehensive report
    generate_comprehensive_report(
        metrics, advanced_metrics, business_metrics,
        y_test_np, y_pred, save_path='comprehensive_evaluation_report.txt'
    )
    
    print("\n" + "="*60)
    print("✓ Evaluation complete! All visualizations saved.")
    print("="*60)


if __name__ == "__main__":
    main()
