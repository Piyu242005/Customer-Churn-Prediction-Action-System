"""
Advanced Feature Engineering Module
Implements feature selection (RFE, feature importance) and SMOTE for class imbalance

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Advanced feature engineering including selection and balancing
    """
    
    def __init__(self, feature_names=None):
        """
        Initialize feature engineer
        
        Args:
            feature_names (list): List of feature names
        """
        self.feature_names = feature_names
        self.selected_features = None
        self.feature_importance = None
        self.smote = None
        
    def calculate_feature_importance(self, X, y, method='random_forest'):
        """
        Calculate feature importance using various methods
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
            method (str): Method to use ('random_forest', 'mutual_info', 'f_classif')
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        print(f"Calculating feature importance using {method}...")
        
        if method == 'random_forest':
            # Use Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            importance = rf.feature_importances_
            
        elif method == 'mutual_info':
            # Mutual Information
            importance = mutual_info_classif(X, y, random_state=42)
            
        elif method == 'f_classif':
            # ANOVA F-statistic
            importance = SelectKBest(f_classif, k='all').fit(X, y).scores_
            importance = importance / importance.sum()  # Normalize
        
        # Create DataFrame
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame({
                'feature': [f'Feature_{i}' for i in range(X.shape[1])],
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        print(f"\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        return importance_df
    
    def plot_feature_importance(self, top_n=None, save_path=None):
        """
        Plot feature importance
        
        Args:
            top_n (int): Number of top features to show (None = all)
            save_path (str): Path to save figure
        """
        if self.feature_importance is None:
            print("Error: Feature importance not calculated. Run calculate_feature_importance first.")
            return
        
        importance_df = self.feature_importance.copy()
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        plt.figure(figsize=(10, max(6, len(importance_df) * 0.3)))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title('Feature Importance Ranking', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def recursive_feature_elimination(self, X, y, n_features_to_select=10, step=1):
        """
        Perform Recursive Feature Elimination (RFE)
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
            n_features_to_select (int): Number of features to select
            step (int): Number of features to remove at each iteration
            
        Returns:
            tuple: (X_selected, selected_feature_indices, selected_feature_names)
        """
        print(f"\nPerforming Recursive Feature Elimination...")
        print(f"Selecting {n_features_to_select} features from {X.shape[1]}")
        
        # Use Random Forest as the estimator
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        # RFE
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
        rfe.fit(X, y)
        
        # Get selected features
        selected_indices = np.where(rfe.support_)[0]
        X_selected = X[:, selected_indices]
        
        if self.feature_names:
            selected_names = [self.feature_names[i] for i in selected_indices]
            
            print(f"\nSelected Features ({len(selected_names)}):")
            for i, (name, rank) in enumerate(zip(selected_names, rfe.ranking_[selected_indices]), 1):
                print(f"  {i}. {name} (rank: {rank})")
        else:
            selected_names = [f'Feature_{i}' for i in selected_indices]
        
        self.selected_features = selected_names
        
        return X_selected, selected_indices, selected_names
    
    def apply_smote(self, X, y, sampling_strategy='auto', method='smote', random_state=42):
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
            sampling_strategy: Sampling strategy for SMOTE
            method (str): Balancing method ('smote', 'adasyn', 'smote_tomek')
            random_state (int): Random seed
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        print(f"\nApplying {method.upper()} for class balancing...")
        print(f"Original class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        if method == 'smote':
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'adasyn':
            sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        print(f"Resampled class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
        print(f"Original samples: {len(y)}, Resampled samples: {len(y_resampled)}")
        
        self.smote = sampler
        
        return X_resampled, y_resampled
    
    def select_k_best_features(self, X, y, k=10, score_func='f_classif'):
        """
        Select K best features using statistical tests
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
            k (int): Number of features to select
            score_func (str): Scoring function ('f_classif' or 'mutual_info')
            
        Returns:
            tuple: (X_selected, selected_indices, selected_names, scores)
        """
        print(f"\nSelecting {k} best features using {score_func}...")
        
        if score_func == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        elif score_func == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        else:
            raise ValueError(f"Unknown score function: {score_func}")
        
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        scores = selector.scores_
        
        if self.feature_names:
            selected_names = [self.feature_names[i] for i in selected_indices]
            
            # Create DataFrame with scores
            feature_scores = pd.DataFrame({
                'feature': selected_names,
                'score': scores[selected_indices]
            }).sort_values('score', ascending=False)
            
            print("\nSelected Features with Scores:")
            print(feature_scores)
        else:
            selected_names = [f'Feature_{i}' for i in selected_indices]
        
        return X_selected, selected_indices, selected_names, scores
    
    def analyze_class_imbalance(self, y):
        """
        Analyze class imbalance in the dataset
        
        Args:
            y (np.array): Target labels
            
        Returns:
            dict: Class distribution statistics
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        imbalance_stats = {
            'class_counts': dict(zip(unique, counts)),
            'class_proportions': dict(zip(unique, counts / total)),
            'imbalance_ratio': max(counts) / min(counts),
            'minority_class': unique[np.argmin(counts)],
            'majority_class': unique[np.argmax(counts)]
        }
        
        print("\n" + "="*60)
        print("CLASS IMBALANCE ANALYSIS")
        print("="*60)
        print(f"Total Samples: {total}")
        for cls, count in imbalance_stats['class_counts'].items():
            proportion = imbalance_stats['class_proportions'][cls]
            print(f"Class {cls}: {count} samples ({proportion*100:.2f}%)")
        print(f"Imbalance Ratio: {imbalance_stats['imbalance_ratio']:.2f}:1")
        print("="*60)
        
        return imbalance_stats
    
    def plot_class_distribution(self, y, y_resampled=None, save_path=None):
        """
        Plot class distribution before and after resampling
        
        Args:
            y (np.array): Original target labels
            y_resampled (np.array): Resampled target labels (optional)
            save_path (str): Path to save figure
        """
        if y_resampled is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Original distribution
            unique, counts = np.unique(y, return_counts=True)
            axes[0].bar(['Active (0)', 'Churned (1)'], counts, color=['#3498db', '#e74c3c'])
            axes[0].set_title('Original Class Distribution', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Count', fontsize=12)
            for i, count in enumerate(counts):
                axes[0].text(i, count, f'{count}\n({count/len(y)*100:.1f}%)', 
                           ha='center', va='bottom', fontsize=11)
            
            # Resampled distribution
            unique_r, counts_r = np.unique(y_resampled, return_counts=True)
            axes[1].bar(['Active (0)', 'Churned (1)'], counts_r, color=['#3498db', '#e74c3c'])
            axes[1].set_title('After SMOTE Resampling', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Count', fontsize=12)
            for i, count in enumerate(counts_r):
                axes[1].text(i, count, f'{count}\n({count/len(y_resampled)*100:.1f}%)', 
                           ha='center', va='bottom', fontsize=11)
            
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            unique, counts = np.unique(y, return_counts=True)
            ax.bar(['Active (0)', 'Churned (1)'], counts, color=['#3498db', '#e74c3c'])
            ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12)
            for i, count in enumerate(counts):
                ax.text(i, count, f'{count}\n({count/len(y)*100:.1f}%)', 
                       ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        plt.show()


class FeatureSelector:
    """
    Comprehensive feature selection toolkit
    """
    
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self.selection_results = {}
    
    def compare_selection_methods(self, X, y, k=10):
        """
        Compare multiple feature selection methods
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
            k (int): Number of features to select
            
        Returns:
            dict: Results from different methods
        """
        print("\n" + "="*60)
        print("COMPARING FEATURE SELECTION METHODS")
        print("="*60)
        
        results = {}
        
        # Method 1: Random Forest Feature Importance
        print("\n1. Random Forest Feature Importance:")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        rf_indices = np.argsort(rf_importance)[-k:][::-1]
        results['random_forest'] = {
            'indices': rf_indices,
            'scores': rf_importance[rf_indices],
            'features': [self.feature_names[i] for i in rf_indices] if self.feature_names else rf_indices
        }
        print(f"   Top {k} features: {results['random_forest']['features']}")
        
        # Method 2: Mutual Information
        print("\n2. Mutual Information:")
        mi_selector = SelectKBest(mutual_info_classif, k=k)
        mi_selector.fit(X, y)
        mi_indices = mi_selector.get_support(indices=True)
        results['mutual_info'] = {
            'indices': mi_indices,
            'scores': mi_selector.scores_[mi_indices],
            'features': [self.feature_names[i] for i in mi_indices] if self.feature_names else mi_indices
        }
        print(f"   Top {k} features: {results['mutual_info']['features']}")
        
        # Method 3: ANOVA F-test
        print("\n3. ANOVA F-test:")
        f_selector = SelectKBest(f_classif, k=k)
        f_selector.fit(X, y)
        f_indices = f_selector.get_support(indices=True)
        results['f_classif'] = {
            'indices': f_indices,
            'scores': f_selector.scores_[f_indices],
            'features': [self.feature_names[i] for i in f_indices] if self.feature_names else f_indices
        }
        print(f"   Top {k} features: {results['f_classif']['features']}")
        
        # Method 4: RFE with Random Forest
        print("\n4. Recursive Feature Elimination (RFE):")
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), 
                  n_features_to_select=k, step=1)
        rfe.fit(X, y)
        rfe_indices = np.where(rfe.support_)[0]
        results['rfe'] = {
            'indices': rfe_indices,
            'ranking': rfe.ranking_[rfe_indices],
            'features': [self.feature_names[i] for i in rfe_indices] if self.feature_names else rfe_indices
        }
        print(f"   Top {k} features: {results['rfe']['features']}")
        
        self.selection_results = results
        
        # Find consensus features (appears in at least 3 methods)
        all_features = set()
        feature_counts = {}
        
        for method_name, method_results in results.items():
            features = method_results['features']
            all_features.update(features)
            for feat in features:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
        
        consensus_features = [feat for feat, count in feature_counts.items() if count >= 3]
        
        print(f"\n{'='*60}")
        print(f"Consensus Features (appearing in ≥3 methods): {len(consensus_features)}")
        print(f"{'='*60}")
        for feat in consensus_features:
            print(f"  • {feat} (in {feature_counts[feat]}/4 methods)")
        
        return results, consensus_features
    
    def plot_feature_selection_comparison(self, save_path=None):
        """
        Visualize feature selection results from different methods
        
        Args:
            save_path (str): Path to save figure
        """
        if not self.selection_results:
            print("Error: Run compare_selection_methods first.")
            return
        
        # Count how many times each feature is selected
        feature_counts = {}
        all_features = set()
        
        for method_name, method_results in self.selection_results.items():
            features = method_results['features']
            all_features.update(features)
            for feat in features:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
        
        # Create DataFrame
        df_counts = pd.DataFrame({
            'feature': list(feature_counts.keys()),
            'selection_count': list(feature_counts.values())
        }).sort_values('selection_count', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, max(6, len(df_counts) * 0.3)))
        colors = ['#27ae60' if x >= 3 else '#f39c12' if x == 2 else '#95a5a6' 
                  for x in df_counts['selection_count']]
        sns.barplot(data=df_counts, y='feature', x='selection_count', palette=colors)
        plt.title('Feature Selection Consensus Across Methods', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Methods Selecting This Feature', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.axvline(x=3, color='red', linestyle='--', linewidth=1, alpha=0.5, 
                   label='Consensus Threshold (3+)')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature selection comparison saved to {save_path}")
        
        plt.show()


def apply_advanced_feature_engineering(X_train, X_test, y_train, feature_names=None, 
                                      use_smote=True, use_feature_selection=True,
                                      n_features=None, smote_method='smote'):
    """
    Apply advanced feature engineering pipeline
    
    Args:
        X_train (np.array): Training features
        X_test (np.array): Test features
        y_train (np.array): Training labels
        feature_names (list): Feature names
        use_smote (bool): Whether to apply SMOTE
        use_feature_selection (bool): Whether to do feature selection
        n_features (int): Number of features to select (None = keep all)
        smote_method (str): SMOTE method to use
        
    Returns:
        tuple: (X_train_processed, X_test_processed, y_train_processed, 
                feature_engineer, selected_feature_indices)
    """
    print("\n" + "="*60)
    print("ADVANCED FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    engineer = FeatureEngineer(feature_names)
    selected_indices = None
    
    # Analyze class imbalance
    engineer.analyze_class_imbalance(y_train)
    
    # Apply SMOTE if requested
    if use_smote:
        X_train_balanced, y_train_balanced = engineer.apply_smote(
            X_train, y_train, method=smote_method
        )
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Feature selection
    if use_feature_selection and n_features:
        # Calculate feature importance
        importance_df = engineer.calculate_feature_importance(
            X_train_balanced, y_train_balanced, method='random_forest'
        )
        
        # Select top N features
        top_indices = importance_df.head(n_features).index.tolist()
        
        # Get actual column indices
        if feature_names:
            selected_indices = [feature_names.index(importance_df.iloc[i]['feature']) 
                              for i in range(min(n_features, len(importance_df)))]
        else:
            selected_indices = list(range(min(n_features, X_train.shape[1])))
        
        X_train_processed = X_train_balanced[:, selected_indices]
        X_test_processed = X_test[:, selected_indices]
        
        print(f"\nFeature selection: {X_train.shape[1]} → {X_train_processed.shape[1]} features")
    else:
        X_train_processed = X_train_balanced
        X_test_processed = X_test
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Final Training Shape: {X_train_processed.shape}")
    print(f"Final Test Shape: {X_test_processed.shape}")
    print(f"Final Target Shape: {len(y_train_balanced)}")
    
    return X_train_processed, X_test_processed, y_train_balanced, engineer, selected_indices


if __name__ == "__main__":
    """Demo of feature engineering capabilities"""
    from src.data.data_preprocessing import load_and_preprocess_data
    
    print("Feature Engineering Demo")
    print("="*60)
    
    # Load data
    data_path = "Business_Analytics_Dataset_10000_Rows.csv"
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_path)
    
    # Convert tensors to numpy
    X_train_np = X_train.numpy()
    X_test_np = X_test.numpy()
    y_train_np = y_train.numpy().flatten()
    y_test_np = y_test.numpy().flatten()
    
    # Get feature names
    feature_names = preprocessor.get_feature_names()
    
    # Initialize feature engineer
    engineer = FeatureEngineer(feature_names)
    
    # Analyze class imbalance
    engineer.analyze_class_imbalance(y_train_np)
    
    # Calculate feature importance
    importance_df = engineer.calculate_feature_importance(X_train_np, y_train_np, method='random_forest')
    engineer.plot_feature_importance(top_n=10, save_path='feature_importance.png')
    
    # Compare selection methods
    selector = FeatureSelector(feature_names)
    results, consensus = selector.compare_selection_methods(X_train_np, y_train_np, k=10)
    selector.plot_feature_selection_comparison(save_path='feature_selection_comparison.png')
    
    # Apply SMOTE
    X_train_balanced, y_train_balanced = engineer.apply_smote(X_train_np, y_train_np)
    engineer.plot_class_distribution(y_train_np, y_train_balanced, 
                                    save_path='class_distribution.png')
    
    print("\n✓ Feature engineering demo complete!")
