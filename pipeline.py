"""
End-to-End ML Pipeline for Churn Prediction
Automates the complete workflow: preprocessing → training → evaluation → deployment

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

import os
import sys
import argparse
import json
import joblib
from datetime import datetime
import torch
import numpy as np

# Import project modules
from data_preprocessing import load_and_preprocess_data
from model import create_model
from train import MLPTrainer
from evaluate import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve, 
    plot_precision_recall_curve, plot_threshold_analysis,
    plot_calibration_curve, plot_prediction_distribution,
    plot_error_analysis, calculate_advanced_metrics,
    calculate_business_metrics, generate_comprehensive_report
)
from feature_engineering import apply_advanced_feature_engineering, FeatureEngineer
from baseline_comparison import run_comprehensive_comparison
from explainability import generate_shap_report

from torch.utils.data import TensorDataset, DataLoader


class ChurnPredictionPipeline:
    """
    End-to-end pipeline for churn prediction
    """
    
    def __init__(self, config):
        """
        Initialize pipeline
        
        Args:
            config (dict): Pipeline configuration
        """
        self.config = config
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir = f"outputs/run_{self.timestamp}"
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.output_dir}/reports", exist_ok=True)
        os.makedirs(f"{self.output_dir}/explainability", exist_ok=True)
        os.makedirs(f"{self.output_dir}/comparison", exist_ok=True)
        
        print("="*70)
        print("CHURN PREDICTION PIPELINE INITIALIZED")
        print("="*70)
        print(f"Output Directory: {self.output_dir}")
        print(f"Timestamp: {self.timestamp}")
        print("="*70)
    
    def step_1_load_data(self):
        """Step 1: Load and preprocess data"""
        print("\n" + "="*70)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("="*70)
        
        data_path = self.config.get('data_path', 'Business_Analytics_Dataset_10000_Rows.csv')
        test_size = self.config.get('test_size', 0.2)
        
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(
            data_path, test_size=test_size
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.preprocessor = preprocessor
        self.feature_names = preprocessor.get_feature_names()
        
        # Save preprocessing artifacts
        joblib.dump(preprocessor.scaler, f"{self.output_dir}/models/scaler.pkl")
        joblib.dump(self.feature_names, f"{self.output_dir}/models/feature_names.pkl")
        joblib.dump(preprocessor.label_encoders, f"{self.output_dir}/models/label_encoders.pkl")
        
        print(f"✓ Data loaded: Train={X_train.shape}, Test={X_test.shape}")
        print(f"✓ Preprocessing artifacts saved to {self.output_dir}/models/")
    
    def step_2_feature_engineering(self):
        """Step 2: Advanced feature engineering"""
        print("\n" + "="*70)
        print("STEP 2: ADVANCED FEATURE ENGINEERING")
        print("="*70)
        
        use_smote = self.config.get('use_smote', True)
        use_feature_selection = self.config.get('use_feature_selection', False)
        n_features = self.config.get('n_features_to_select', None)
        
        if use_smote or use_feature_selection:
            X_train_np = self.X_train.numpy()
            X_test_np = self.X_test.numpy()
            y_train_np = self.y_train.numpy().flatten()
            
            X_train_eng, X_test_eng, y_train_eng, engineer, selected_indices = \
                apply_advanced_feature_engineering(
                    X_train_np, X_test_np, y_train_np,
                    feature_names=self.feature_names,
                    use_smote=use_smote,
                    use_feature_selection=use_feature_selection,
                    n_features=n_features
                )
            
            # Convert back to tensors
            self.X_train = torch.FloatTensor(X_train_eng)
            self.X_test = torch.FloatTensor(X_test_eng)
            self.y_train = torch.FloatTensor(y_train_eng).unsqueeze(1)
            
            # Update feature names if selection was done
            if selected_indices:
                self.feature_names = [self.feature_names[i] for i in selected_indices]
            
            # Save feature engineering plots
            engineer.plot_feature_importance(
                top_n=10, 
                save_path=f"{self.output_dir}/plots/feature_importance.png"
            )
            
            print(f"✓ Feature engineering complete")
        else:
            print("⊘ Skipping feature engineering (disabled in config)")
    
    def step_3_train_model(self):
        """Step 3: Train MLP model"""
        print("\n" + "="*70)
        print("STEP 3: MODEL TRAINING")
        print("="*70)
        
        # Configuration
        batch_size = self.config.get('batch_size', 32)
        epochs = self.config.get('epochs', 100)
        learning_rate = self.config.get('learning_rate', 0.001)
        hidden_dims = self.config.get('hidden_dims', [128, 64, 32])
        dropout_rate = self.config.get('dropout_rate', 0.3)
        patience = self.config.get('patience', 15)
        weight_decay = self.config.get('weight_decay', 1e-5)
        device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        test_dataset = TensorDataset(self.X_test, self.y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Create model
        input_dim = self.X_train.shape[1]
        self.model = create_model(input_dim, hidden_dims, dropout_rate)
        
        print(f"\nModel Architecture:")
        print(self.model)
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Train
        trainer = MLPTrainer(self.model, device)
        history = trainer.train(
            train_loader, test_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience,
            weight_decay=weight_decay
        )
        
        self.trainer = trainer
        self.history = history
        
        # Save model
        model_path = f"{self.output_dir}/models/mlp_churn_classifier.pth"
        trainer.save_model(model_path)
        
        # Save training history
        history_path = f"{self.output_dir}/reports/training_history.json"
        with open(history_path, 'w') as f:
            history_serializable = {k: [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                       for v in vals] if isinstance(vals, list) else float(vals)
                                   for k, vals in history.items()}
            json.dump(history_serializable, f, indent=4)
        
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Training history saved to {history_path}")
    
    def step_4_evaluate_model(self):
        """Step 4: Comprehensive model evaluation"""
        print("\n" + "="*70)
        print("STEP 4: MODEL EVALUATION")
        print("="*70)
        
        # Evaluate
        metrics, y_pred, y_pred_proba = evaluate_model(self.model, self.X_test, self.y_test)
        
        y_test_np = self.y_test.numpy().flatten()
        
        print("\nPerformance Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Generate all plots
        print("\nGenerating evaluation plots...")
        
        plot_dir = f"{self.output_dir}/plots"
        
        plot_confusion_matrix(y_test_np, y_pred, 
                            save_path=f"{plot_dir}/confusion_matrix.png")
        plot_roc_curve(y_test_np, y_pred_proba, 
                      save_path=f"{plot_dir}/roc_curve.png")
        plot_precision_recall_curve(y_test_np, y_pred_proba,
                                   save_path=f"{plot_dir}/precision_recall_curve.png")
        plot_threshold_analysis(y_test_np, y_pred_proba,
                               save_path=f"{plot_dir}/threshold_analysis.png")
        plot_calibration_curve(y_test_np, y_pred_proba,
                              save_path=f"{plot_dir}/calibration_curve.png")
        plot_prediction_distribution(y_test_np, y_pred_proba,
                                    save_path=f"{plot_dir}/prediction_distribution.png")
        plot_error_analysis(y_test_np, y_pred, y_pred_proba,
                          save_path=f"{plot_dir}/error_analysis.png")
        
        # Calculate advanced metrics
        advanced_metrics = calculate_advanced_metrics(y_test_np, y_pred, y_pred_proba)
        
        # Calculate business metrics
        business_metrics = calculate_business_metrics(
            y_test_np, y_pred, y_pred_proba,
            churn_cost=self.config.get('churn_cost', 500),
            retention_cost=self.config.get('retention_cost', 50),
            success_rate=self.config.get('retention_success_rate', 0.3)
        )
        
        # Generate comprehensive report
        generate_comprehensive_report(
            metrics, advanced_metrics, business_metrics,
            y_test_np, y_pred,
            save_path=f"{self.output_dir}/reports/evaluation_report.txt"
        )
        
        self.results['evaluation'] = {
            'metrics': metrics,
            'advanced_metrics': advanced_metrics,
            'business_metrics': business_metrics
        }
        
        print(f"✓ Evaluation plots saved to {plot_dir}/")
        print(f"✓ Evaluation report saved")
    
    def step_5_baseline_comparison(self):
        """Step 5: Compare with baseline models"""
        print("\n" + "="*70)
        print("STEP 5: BASELINE MODEL COMPARISON")
        print("="*70)
        
        if self.config.get('skip_baseline_comparison', False):
            print("⊘ Baseline comparison skipped (disabled in config)")
            return
        
        comparator, results_df = run_comprehensive_comparison(
            self.X_train, self.X_test, self.y_train, self.y_test,
            mlp_model=self.model,
            device=self.config.get('device', 'cpu'),
            save_dir=f"{self.output_dir}/comparison/"
        )
        
        self.results['comparison'] = {
            'results_df': results_df.to_dict(),
            'best_model': results_df.index[0],
            'best_accuracy': float(results_df.iloc[0]['accuracy'])
        }
        
        print(f"✓ Baseline comparison complete")
        print(f"✓ Comparison results saved to {self.output_dir}/comparison/")
    
    def step_6_explainability(self):
        """Step 6: Generate model explainability reports"""
        print("\n" + "="*70)
        print("STEP 6: MODEL EXPLAINABILITY (SHAP)")
        print("="*70)
        
        if self.config.get('skip_explainability', False):
            print("⊘ Explainability analysis skipped (disabled in config)")
            return
        
        try:
            explainer, shap_values = generate_shap_report(
                self.model,
                self.X_train,
                self.X_test,
                self.y_test,
                self.feature_names,
                device=self.config.get('device', 'cpu'),
                save_dir=f"{self.output_dir}/explainability/"
            )
            
            print(f"✓ Explainability report complete")
            print(f"✓ SHAP plots saved to {self.output_dir}/explainability/")
        
        except Exception as e:
            print(f"⚠ Explainability analysis failed: {e}")
            print("  (This is optional and won't affect model performance)")
    
    def step_7_generate_summary(self):
        """Step 7: Generate pipeline summary"""
        print("\n" + "="*70)
        print("STEP 7: GENERATING PIPELINE SUMMARY")
        print("="*70)
        
        summary = {
            'pipeline_run': {
                'timestamp': self.timestamp,
                'config': self.config,
                'output_directory': self.output_dir
            },
            'data': {
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'num_features': self.X_train.shape[1],
                'feature_names': self.feature_names
            },
            'model': {
                'architecture': 'MLP',
                'hidden_dims': self.config.get('hidden_dims', [128, 64, 32]),
                'dropout_rate': self.config.get('dropout_rate', 0.3),
                'total_parameters': sum(p.numel() for p in self.model.parameters())
            },
            'training': {
                'epochs': self.history.get('epochs_trained', 0),
                'training_time': self.history.get('training_time', 0),
                'final_train_loss': float(self.history['train_losses'][-1]),
                'final_val_loss': float(self.history['val_losses'][-1]),
                'final_train_acc': float(self.history['train_accuracies'][-1]),
                'final_val_acc': float(self.history['val_accuracies'][-1])
            },
            'results': self.results
        }
        
        # Save summary
        summary_path = f"{self.output_dir}/pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        print(f"✓ Pipeline summary saved to {summary_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*70)
        print(f"Total Training Time: {summary['training']['training_time']:.2f}s")
        print(f"Epochs Trained: {summary['training']['epochs']}")
        print(f"Final Validation Accuracy: {summary['training']['final_val_acc']:.4f}")
        
        if 'evaluation' in self.results:
            eval_metrics = self.results['evaluation']['metrics']
            print(f"\nTest Set Performance:")
            print(f"  Accuracy:  {eval_metrics['accuracy']:.4f}")
            print(f"  F1-Score:  {eval_metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {eval_metrics['roc_auc']:.4f}")
        
        if 'comparison' in self.results:
            print(f"\nBest Model: {self.results['comparison']['best_model']}")
            print(f"Best Accuracy: {self.results['comparison']['best_accuracy']:.4f}")
        
        print("="*70)
        
        return summary
    
    def run(self):
        """Execute complete pipeline"""
        print("\n" + "#"*70)
        print("EXECUTING END-TO-END CHURN PREDICTION PIPELINE")
        print("#"*70)
        
        try:
            # Step 1: Load data
            self.step_1_load_data()
            
            # Step 2: Feature engineering
            if self.config.get('enable_feature_engineering', True):
                self.step_2_feature_engineering()
            
            # Step 3: Train model
            self.step_3_train_model()
            
            # Step 4: Evaluate model
            self.step_4_evaluate_model()
            
            # Step 5: Baseline comparison
            if self.config.get('enable_baseline_comparison', True):
                self.step_5_baseline_comparison()
            
            # Step 6: Explainability
            if self.config.get('enable_explainability', True):
                self.step_6_explainability()
            
            # Step 7: Generate summary
            summary = self.step_7_generate_summary()
            
            print("\n" + "#"*70)
            print("✓ PIPELINE EXECUTION COMPLETE")
            print("#"*70)
            print(f"\n📁 All outputs saved to: {self.output_dir}")
            print("\nGenerated Files:")
            print(f"  • Model: {self.output_dir}/models/mlp_churn_classifier.pth")
            print(f"  • Plots: {self.output_dir}/plots/")
            print(f"  • Reports: {self.output_dir}/reports/")
            print(f"  • Summary: {self.output_dir}/pipeline_summary.json")
            
            return summary
        
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def create_default_config():
    """Create default pipeline configuration"""
    return {
        # Data
        'data_path': 'Business_Analytics_Dataset_10000_Rows.csv',
        'test_size': 0.2,
        
        # Feature Engineering
        'enable_feature_engineering': True,
        'use_smote': True,
        'use_feature_selection': False,
        'n_features_to_select': None,
        
        # Model Architecture
        'hidden_dims': [128, 64, 32],
        'dropout_rate': 0.3,
        
        # Training
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Pipeline Options
        'enable_baseline_comparison': True,
        'enable_explainability': True,
        'skip_baseline_comparison': False,
        'skip_explainability': False,
        
        # Business Metrics
        'churn_cost': 500,
        'retention_cost': 50,
        'retention_success_rate': 0.3
    }


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Run end-to-end churn prediction pipeline')
    
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick mode (skip baseline comparison and explainability)')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE balancing')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Apply command line overrides
    if args.quick:
        config['skip_baseline_comparison'] = True
        config['skip_explainability'] = True
        print("⚡ Quick mode enabled (skipping comparison and explainability)")
    
    if args.no_smote:
        config['use_smote'] = False
        print("⊘ SMOTE disabled")
    
    if args.epochs:
        config['epochs'] = args.epochs
        print(f"⚙ Epochs set to {args.epochs}")
    
    # Save config
    os.makedirs('configs', exist_ok=True)
    config_path = f"configs/pipeline_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"📄 Config saved to {config_path}")
    
    # Run pipeline
    pipeline = ChurnPredictionPipeline(config)
    summary = pipeline.run()
    
    if summary:
        print("\n🎉 SUCCESS! Pipeline completed successfully.")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
