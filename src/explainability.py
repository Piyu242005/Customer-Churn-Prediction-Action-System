"""
Model Explainability Module using SHAP and LIME
Provides interpretability and insights for MLP churn predictions

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')


class ChurnExplainer:
    """
    Model explainability toolkit for churn classifier
    """
    
    def __init__(self, model, feature_names, device='cpu'):
        """
        Initialize explainer
        
        Args:
            model: Trained PyTorch model
            feature_names (list): List of feature names
            device (str): Device model is on
        """
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.model.eval()
        
    def _predict_fn(self, X):
        """
        Prediction function wrapper for SHAP/LIME
        
        Args:
            X (np.array): Input features
            
        Returns:
            np.array: Predicted probabilities
        """
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy()
    
    def compute_shap_values(self, X_background, X_test, max_samples=100):
        """
        Compute SHAP values using DeepExplainer or KernelExplainer
        
        Args:
            X_background (np.array/torch.Tensor): Background data for SHAP
            X_test (np.array/torch.Tensor): Test data to explain
            max_samples (int): Maximum samples to use for background
            
        Returns:
            shap.Explanation: SHAP values object
        """
        print("Computing SHAP values...")
        
        # Convert to numpy if tensor
        if isinstance(X_background, torch.Tensor):
            X_background = X_background.numpy()
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.numpy()
        
        # Limit background samples for efficiency
        if len(X_background) > max_samples:
            indices = np.random.choice(len(X_background), max_samples, replace=False)
            X_background_sample = X_background[indices]
        else:
            X_background_sample = X_background
        
        # Try DeepExplainer first (faster for neural networks)
        try:
            X_bg_tensor = torch.FloatTensor(X_background_sample).to(self.device)
            explainer = shap.DeepExplainer(self.model, X_bg_tensor)
            
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            shap_values = explainer.shap_values(X_test_tensor)
            
            # DeepExplainer returns raw values, wrap in Explanation object
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            print(f"✓ SHAP values computed using DeepExplainer")
            
        except Exception as e:
            print(f"DeepExplainer failed: {e}")
            print("Falling back to KernelExplainer (slower)...")
            
            # Fallback to KernelExplainer
            explainer = shap.KernelExplainer(self._predict_fn, X_background_sample)
            shap_values = explainer.shap_values(X_test[:min(50, len(X_test))])
            
            print(f"✓ SHAP values computed using KernelExplainer")
        
        return shap_values, explainer
    
    def plot_shap_summary(self, shap_values, X_test, plot_type='dot', save_path=None):
        """
        Plot SHAP summary plot
        
        Args:
            shap_values: SHAP values
            X_test (np.array): Test data
            plot_type (str): Type of plot ('dot', 'bar', 'violin')
            save_path (str): Path to save figure
        """
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.numpy()
        
        plt.figure(figsize=(10, 8))
        
        if plot_type == 'dot':
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, 
                            show=False, plot_type='dot')
        elif plot_type == 'bar':
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, 
                            show=False, plot_type='bar')
        elif plot_type == 'violin':
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, 
                            show=False, plot_type='violin')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_shap_waterfall(self, shap_values, X_test, sample_idx=0, save_path=None):
        """
        Plot SHAP waterfall plot for a single prediction
        
        Args:
            shap_values: SHAP values
            X_test (np.array): Test data
            sample_idx (int): Index of sample to explain
            save_path (str): Path to save figure
        """
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.numpy()
        
        # Create explanation object
        base_value = np.mean(self._predict_fn(X_test))
        
        # Create SHAP explanation for single sample
        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=base_value,
            data=X_test[sample_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP waterfall plot saved to {save_path}")
        
        plt.show()
    
    def plot_shap_force(self, shap_values, X_test, explainer, sample_idx=0, save_path=None):
        """
        Plot SHAP force plot for a single prediction
        
        Args:
            shap_values: SHAP values
            X_test (np.array): Test data
            explainer: SHAP explainer object
            sample_idx (int): Index of sample to explain
            save_path (str): Path to save figure
        """
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.numpy()
        
        # Get expected value
        expected_value = np.mean(self._predict_fn(X_test))
        
        # Create force plot
        shap.force_plot(
            expected_value,
            shap_values[sample_idx],
            X_test[sample_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP force plot saved to {save_path}")
        
        plt.show()
    
    def plot_shap_dependence(self, shap_values, X_test, feature_name, 
                            interaction_feature=None, save_path=None):
        """
        Plot SHAP dependence plot for a specific feature
        
        Args:
            shap_values: SHAP values
            X_test (np.array): Test data
            feature_name (str): Feature to analyze
            interaction_feature (str): Feature to show interactions with
            save_path (str): Path to save figure
        """
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.numpy()
        
        feature_idx = self.feature_names.index(feature_name)
        
        if interaction_feature:
            interaction_idx = self.feature_names.index(interaction_feature)
        else:
            interaction_idx = 'auto'
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx, shap_values, X_test,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP dependence plot saved to {save_path}")
        
        plt.show()
    
    def get_top_features_for_prediction(self, shap_values, X_test, sample_idx=0, top_n=5):
        """
        Get top contributing features for a specific prediction
        
        Args:
            shap_values: SHAP values
            X_test (np.array): Test data
            sample_idx (int): Sample index
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Top features with their contributions
        """
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.numpy()
        
        sample_shap = shap_values[sample_idx]
        if isinstance(sample_shap, np.ndarray) and sample_shap.ndim > 1:
            sample_shap = sample_shap.flatten()
            
        sample_features = X_test[sample_idx]
        if isinstance(sample_features, np.ndarray) and sample_features.ndim > 1:
            sample_features = sample_features.flatten()
        
        # Create DataFrame
        feature_contributions = pd.DataFrame({
            'feature': self.feature_names,
            'feature_value': sample_features,
            'shap_value': sample_shap,
            'abs_shap_value': np.abs(sample_shap)
        }).sort_values('abs_shap_value', ascending=False)
        
        return feature_contributions.head(top_n)
    
    def explain_prediction(self, shap_values, X_test, y_test, sample_idx=0):
        """
        Comprehensive explanation for a single prediction
        
        Args:
            shap_values: SHAP values
            X_test (np.array/torch.Tensor): Test features
            y_test (np.array/torch.Tensor): Test labels
            sample_idx (int): Sample to explain
        """
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.numpy()
        if isinstance(y_test, torch.Tensor):
            y_test = y_test.numpy().flatten()
        
        # Get prediction
        prediction_proba = self._predict_fn(X_test[sample_idx:sample_idx+1])[0][0]
        prediction_class = 1 if prediction_proba >= 0.5 else 0
        true_class = int(y_test[sample_idx])
        
        print("\n" + "="*60)
        print(f"EXPLANATION FOR SAMPLE #{sample_idx}")
        print("="*60)
        print(f"True Class:        {'Churned' if true_class == 1 else 'Active'} ({true_class})")
        print(f"Predicted Class:   {'Churned' if prediction_class == 1 else 'Active'} ({prediction_class})")
        print(f"Churn Probability: {prediction_proba:.4f} ({prediction_proba*100:.2f}%)")
        print(f"Prediction:        {'✓ CORRECT' if prediction_class == true_class else '✗ INCORRECT'}")
        
        # Get top contributing features
        top_features = self.get_top_features_for_prediction(shap_values, X_test, sample_idx, top_n=10)
        
        print("\n" + "-"*60)
        print("TOP 10 CONTRIBUTING FEATURES:")
        print("-"*60)
        print(f"{'Rank':<6} {'Feature':<30} {'Value':<12} {'SHAP Impact':<12}")
        print("-"*60)
        
        for i, row in enumerate(top_features.itertuples(), 1):
            impact = "↑ Increases" if row.shap_value > 0 else "↓ Decreases"
            print(f"{i:<6} {row.feature:<30} {row.feature_value:<12.4f} {row.shap_value:>+8.4f} {impact}")
        
        print("="*60)


class LIMEExplainer:
    """
    LIME explainer for MLP classifier
    """
    
    def __init__(self, model, feature_names, class_names=['Active', 'Churned'], device='cpu'):
        """
        Initialize LIME explainer
        
        Args:
            model: Trained PyTorch model
            feature_names (list): Feature names
            class_names (list): Class names
            device (str): Device model is on
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.device = device
        self.model.eval()
    
    def _predict_fn(self, X):
        """
        Prediction function for LIME
        
        Args:
            X (np.array): Input features
            
        Returns:
            np.array: Class probabilities
        """
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor).cpu().numpy()
            # Return both class probabilities
            return np.hstack([1 - outputs, outputs])
    
    def create_explainer(self, X_train):
        """
        Create LIME explainer
        
        Args:
            X_train (np.array): Training data for LIME reference
            
        Returns:
            lime.lime_tabular.LimeTabularExplainer: LIME explainer
        """
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.numpy()
        
        self.explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )
        
        print("✓ LIME explainer created")
        return self.explainer
    
    def explain_instance(self, X_test, sample_idx=0, num_features=10):
        """
        Explain a single prediction using LIME
        
        Args:
            X_test (np.array): Test data
            sample_idx (int): Sample to explain
            num_features (int): Number of top features to show
            
        Returns:
            lime.explanation.Explanation: LIME explanation
        """
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.numpy()
        
        if not hasattr(self, 'explainer'):
            raise ValueError("Create explainer first using create_explainer()")
        
        print(f"Explaining prediction for sample {sample_idx}...")
        
        explanation = self.explainer.explain_instance(
            X_test[sample_idx],
            self._predict_fn,
            num_features=num_features
        )
        
        return explanation
    
    def plot_lime_explanation(self, explanation, save_path=None):
        """
        Plot LIME explanation
        
        Args:
            explanation: LIME explanation object
            save_path (str): Path to save figure
        """
        fig = explanation.as_pyplot_figure()
        plt.title('LIME Feature Importance for Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LIME explanation saved to {save_path}")
        
        plt.show()


def generate_shap_report(model, X_train, X_test, y_test, feature_names, 
                        device='cpu', save_dir='plots/explainability/'):
    """
    Generate comprehensive SHAP analysis report with visualizations
    
    Args:
        model: Trained model
        X_train: Training data
        X_test: Test data
        y_test: Test labels
        feature_names (list): Feature names
        device (str): Device
        save_dir (str): Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING SHAP EXPLAINABILITY REPORT")
    print("="*60)
    
    # Initialize explainer
    explainer = ChurnExplainer(model, feature_names, device)
    
    # Sub-sample for SHAP (too slow on full test set)
    sample_size = min(100, len(X_test))
    if isinstance(X_test, torch.Tensor):
        X_test_sample = X_test[:sample_size].numpy()
        y_test_sample = y_test[:sample_size].numpy() if isinstance(y_test, torch.Tensor) else y_test[:sample_size]
    else:
        X_test_sample = X_test[:sample_size]
        y_test_sample = y_test[:sample_size]
        
    X_train_sample = X_train[:500] if len(X_train) > 500 else X_train
    
    # Compute SHAP values
    shap_values, shap_explainer = explainer.compute_shap_values(X_train_sample, X_test_sample, max_samples=100)
    
    # 1. Summary Plot (Dot)
    print("\n1. Creating SHAP Summary Plot (Dot)...")
    explainer.plot_shap_summary(shap_values, X_test_sample, plot_type='dot', 
                               save_path=f'{save_dir}shap_summary_dot.png')
    
    # 2. Summary Plot (Bar)
    print("\n2. Creating SHAP Summary Plot (Bar)...")
    explainer.plot_shap_summary(shap_values, X_test_sample, plot_type='bar', 
                               save_path=f'{save_dir}shap_summary_bar.png')
    
    # 3. Explain specific predictions
    print("\n3. Explaining Individual Predictions...")
    
    # Find interesting samples
    if isinstance(y_test_sample, torch.Tensor):
        y_test_np = y_test_sample.numpy().flatten()
    else:
        y_test_np = y_test_sample.flatten()
    
    # True positive (correctly predicted churn)
    tp_indices = np.where((y_test_np == 1) & 
                         (explainer._predict_fn(X_test_sample if isinstance(X_test_sample, np.ndarray) 
                          else X_test_sample.numpy()).flatten() >= 0.5))[0]
    if len(tp_indices) > 0:
        print(f"\n   a) True Positive (Churned, Correctly Predicted):")
        explainer.explain_prediction(shap_values, X_test_sample, y_test_np, sample_idx=tp_indices[0])
        explainer.plot_shap_waterfall(shap_values, X_test_sample, sample_idx=tp_indices[0],
                                     save_path=f'{save_dir}shap_waterfall_true_positive.png')
    
    # False positive (incorrectly predicted as churn)
    fp_indices = np.where((y_test_np == 0) & 
                         (explainer._predict_fn(X_test_sample if isinstance(X_test_sample, np.ndarray) 
                          else X_test_sample.numpy()).flatten() >= 0.5))[0]
    if len(fp_indices) > 0:
        print(f"\n   b) False Positive (Active, Incorrectly Predicted as Churned):")
        explainer.explain_prediction(shap_values, X_test_sample, y_test_np, sample_idx=fp_indices[0])
        explainer.plot_shap_waterfall(shap_values, X_test_sample, sample_idx=fp_indices[0],
                                     save_path=f'{save_dir}shap_waterfall_false_positive.png')
    
    # 4. Dependence plots for top features
    print("\n4. Creating SHAP Dependence Plots for Top Features...")
    
    # Get top 3 most important features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_feature_indices = np.argsort(mean_abs_shap)[-3:][::-1]
    
    for idx in top_feature_indices:
        feature_name = feature_names[idx]
        print(f"   - Dependence plot for {feature_name}")
        explainer.plot_shap_dependence(
            shap_values, X_test_sample, feature_name,
            save_path=f'{save_dir}shap_dependence_{feature_name.replace(" ", "_")}.png'
        )
    
    print("\n" + "="*60)
    print("✓ SHAP REPORT GENERATION COMPLETE")
    print(f"✓ All plots saved to {save_dir}")
    print("="*60)
    
    return explainer, shap_values


def generate_lime_report(model, X_train, X_test, y_test, feature_names, 
                        device='cpu', num_samples=5, save_dir='plots/explainability/'):
    """
    Generate LIME analysis report
    
    Args:
        model: Trained model
        X_train: Training data
        X_test: Test data
        y_test: Test labels
        feature_names (list): Feature names
        device (str): Device
        num_samples (int): Number of samples to explain
        save_dir (str): Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING LIME EXPLAINABILITY REPORT")
    print("="*60)
    
    # Initialize LIME explainer
    lime_explainer = LIMEExplainer(model, feature_names, device=device)
    
    if isinstance(X_train, torch.Tensor):
        X_train_np = X_train.numpy()
    else:
        X_train_np = X_train
    
    if isinstance(X_test, torch.Tensor):
        X_test_np = X_test.numpy()
    else:
        X_test_np = X_test
    
    # Create explainer
    lime_explainer.create_explainer(X_train_np)
    
    # Explain random samples
    sample_indices = np.random.choice(len(X_test_np), min(num_samples, len(X_test_np)), 
                                     replace=False)
    
    print(f"\nExplaining {len(sample_indices)} random predictions...")
    
    for i, idx in enumerate(sample_indices):
        print(f"\nSample {i+1}/{len(sample_indices)} (Index: {idx})")
        explanation = lime_explainer.explain_instance(X_test_np, sample_idx=idx, num_features=10)
        lime_explainer.plot_lime_explanation(explanation, 
                                            save_path=f'{save_dir}lime_explanation_sample_{idx}.png')
    
    print("\n" + "="*60)
    print("✓ LIME REPORT GENERATION COMPLETE")
    print(f"✓ All plots saved to {save_dir}")
    print("="*60)
    
    return lime_explainer


def generate_business_insights(shap_values, X_test, feature_names, save_path=None):
    """
    Generate business insights from SHAP analysis
    
    Args:
        shap_values: SHAP values
        X_test: Test data
        feature_names (list): Feature names
        save_path (str): Path to save report
    """
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.numpy()
    
    print("\n" + "="*60)
    print("BUSINESS INSIGHTS FROM MODEL EXPLAINABILITY")
    print("="*60)
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Average_Impact': mean_abs_shap,
        'Impact_Percentage': mean_abs_shap / mean_abs_shap.sum() * 100
    }).sort_values('Average_Impact', ascending=False)
    
    print("\n" + "-"*60)
    print("KEY DRIVERS OF CUSTOMER CHURN:")
    print("-"*60)
    
    for i, row in enumerate(importance_df.head(10).itertuples(), 1):
        print(f"{i}. {row.Feature}")
        print(f"   Impact Score: {row.Average_Impact:.4f} ({row.Impact_Percentage:.2f}% of total)")
    
    # Analyze feature directions
    print("\n" + "-"*60)
    print("FEATURE IMPACT DIRECTION:")
    print("-"*60)
    
    for feature in importance_df.head(5)['Feature']:
        feature_idx = feature_names.index(feature)
        feature_shap = shap_values[:, feature_idx]
        
        positive_impact = (feature_shap > 0).sum()
        negative_impact = (feature_shap < 0).sum()
        
        print(f"\n{feature}:")
        print(f"   Increases churn risk: {positive_impact} samples ({positive_impact/len(feature_shap)*100:.1f}%)")
        print(f"   Decreases churn risk: {negative_impact} samples ({negative_impact/len(feature_shap)*100:.1f}%)")
    
    # Save report
    if save_path:
        with open(save_path, 'w') as f:
            f.write("BUSINESS INSIGHTS REPORT\n")
            f.write("="*60 + "\n\n")
            f.write("TOP CHURN DRIVERS:\n")
            f.write("-"*60 + "\n")
            for i, row in enumerate(importance_df.head(10).itertuples(), 1):
                f.write(f"{i}. {row.Feature}: {row.Average_Impact:.4f} ({row.Impact_Percentage:.2f}%)\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("RECOMMENDATIONS:\n")
            f.write("-"*60 + "\n")
            f.write("1. Focus retention efforts on customers with high-risk feature values\n")
            f.write("2. Monitor top 5 features closely for early churn indicators\n")
            f.write("3. Implement targeted interventions based on feature impacts\n")
        
        print(f"\n✓ Business insights saved to {save_path}")
    
    print("\n" + "="*60)
    
    return importance_df


def main():
    """
    Main explainability analysis
    """
    from data_preprocessing import load_and_preprocess_data
    from model import MLPClassifier
    
    print("="*60)
    print("MODEL EXPLAINABILITY ANALYSIS")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load('mlp_churn_classifier.pth')
    model = MLPClassifier(input_dim=16, hidden_dims=[128, 64, 32], dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded")
    
    # Load data
    print("\nLoading data...")
    data_path = "Business_Analytics_Dataset_10000_Rows.csv"
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_path)
    feature_names = preprocessor.get_feature_names()
    print(f"✓ Data loaded: {len(feature_names)} features")
    
    # Generate SHAP report
    explainer, shap_values = generate_shap_report(
        model, X_train, X_test, y_test, feature_names
    )
    
    # Generate business insights
    generate_business_insights(
        shap_values, X_test, feature_names, 
        save_path='business_insights_report.txt'
    )
    
    # Generate LIME report (optional - slower)
    # generate_lime_report(model, X_train, X_test, y_test, feature_names, num_samples=3)
    
    print("\n✓ Explainability analysis complete!")


if __name__ == "__main__":
    main()
