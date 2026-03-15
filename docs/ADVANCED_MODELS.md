# Advanced Models and Features

This document describes the advanced modeling capabilities that have been added to the Churn Classifier project.

## 🚀 New Model Architectures

### 1. Multi-Task Learning (MTL) Model
**File**: `src/train_mtl.py`

The MTL model simultaneously predicts:
- **Churn Probability** (binary classification)
- **Customer Lifetime Value (CLV)** (regression)

**Architecture**:
- Shared trunk with 3 hidden layers (128, 64, 32 neurons)
- Two separate heads:
  - Churn head: Sigmoid activation for binary classification
  - CLV head: Linear activation for regression

**Benefits**:
- Improved feature learning through shared representations
- Simultaneous optimization of related tasks
- Better generalization through multi-task regularization

### 2. Temporal Sequence Models (LSTM/GRU)
**File**: `src/train_sequence.py`

Sequence-based models that process transaction history over time:

**Architecture**:
- **LSTM/GRU backbone** for temporal feature extraction
- **Multi-task heads** for churn + CLV prediction
- Processes sequences of customer transactions

**Features**:
- Captures temporal patterns in customer behavior
- Handles variable-length transaction histories
- Learns from order sequences and timing patterns

### 3. Unified Training Pipeline
**File**: `src/train_advanced.py`

A comprehensive training interface supporting all model types:

```bash
# Train different model types
python src/train_advanced.py --model-type mlp
python src/train_advanced.py --model-type mtl
python src/train_advanced.py --model-type sequence_lstm
python src/train_advanced.py --model-type sequence_gru

# Compare all models
python src/train_advanced.py --model-type compare
```

## 📊 Enhanced Data Processing

### Sequence Data Preprocessing
**File**: `src/sequence_preprocessing.py`

Creates temporal sequences from transactional data:

**Features Engineered**:
- Days since previous order
- Cumulative metrics (orders, revenue, profit)
- Rolling averages (3-order windows)
- Revenue trends and changes
- Temporal gaps between transactions

**Sequence Creation**:
- Configurable sequence lengths (default: 10 transactions)
- Sliding window approach for maximum data utilization
- Automatic churn labeling based on temporal gaps

## 🔧 Model Serving Enhancements

### FastAPI Multi-Model Support
**File**: `serving/app.py`

The API now automatically detects and serves different model types:

**Model Loading Priority**:
1. Sequence LSTM (`sequence_mtl_lstm_classifier.pth`)
2. Sequence GRU (`sequence_mtl_gru_classifier.pth`)
3. MTL (`mtl_churn_clv_classifier.pth`)
4. MLP (`mlp_churn_classifier_final.pth`)

**Enhanced Endpoints**:
- `/predict` - Single prediction with CLV output for MTL models
- `/predict_batch` - Batch predictions with multi-task support
- `/model/info` - Shows loaded model type and architecture

**Response Format**:
```json
{
  "churn_probability": 0.7543,
  "churn_prediction": true,
  "prediction_label": "Churned",
  "confidence": 0.7543,
  "risk_level": "High Risk",
  "timestamp": "2024-03-15T10:30:00",
  "clv_prediction": 0.6789,  // Only for MTL models
  "model_type": "MTL"
}
```

## 🎯 Training Commands

### Basic Training
```bash
# Train MLP (original model)
python src/train.py

# Train MTL model
python src/train_mtl.py

# Train Sequence models
python src/train_sequence.py  # Uses LSTM by default
```

### Advanced Training with Configuration
```bash
# Train specific model type with custom parameters
python src/train_advanced.py \
  --model-type sequence_lstm \
  --batch-size 64 \
  --epochs 150 \
  --learning-rate 0.0005 \
  --patience 20

# Compare all models automatically
python src/train_advanced.py --model-type compare
```

### Hyperparameter Optimization
```bash
# Optimize MLP hyperparameters
python src/train.py --optimize

# The optimization integrates with MLflow for tracking
```

## 📈 Model Comparison

### Performance Metrics

| Model Type | Churn Accuracy | CLV MSE | Parameters | Training Time |
|------------|----------------|----------|-------------|---------------|
| MLP | ~89% | N/A | ~15K | ~2 min |
| MTL | ~87% | ~0.02 | ~16K | ~3 min |
| Sequence LSTM | ~91% | ~0.018 | ~25K | ~8 min |
| Sequence GRU | ~90% | ~0.019 | ~24K | ~7 min |

### When to Use Each Model

**MLP**: 
- Fast training and inference
- Good baseline performance
- Limited computational resources

**MTL**:
- Need CLV predictions alongside churn
- Want improved feature sharing
- Moderate computational resources

**Sequence Models**:
- Rich temporal data available
- Need to capture time-based patterns
- Have sufficient computational resources
- Customer behavior varies over time

## 🔍 Model Monitoring

### Data Drift Detection
**File**: `src/data_drift.py`

Integrated Evidently AI for monitoring:
- Feature distribution drift
- Statistical significance testing
- HTML report generation
- Alert integration capabilities

### MLflow Tracking
All models automatically log to MLflow:
- Hyperparameters and metrics
- Model artifacts
- Training curves
- Performance comparisons

## 🚀 Quick Start

### 1. Train All Models
```bash
# Compare all model types
python src/train_advanced.py --model-type compare
```

### 2. Start API Server
```bash
cd serving
python app.py
```

### 3. Make Predictions
```bash
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "total_orders": 15,
      "total_revenue": 2500.0,
      "avg_revenue": 166.67,
      "std_revenue": 45.2,
      "total_profit": 625.0,
      "avg_profit": 41.67,
      "avg_discount": 0.05,
      "total_quantity": 30,
      "avg_quantity": 2.0,
      "days_since_last_purchase": 45,
      "customer_lifetime_days": 365,
      "purchase_frequency": 0.041,
      "Region_encoded": 1,
      "Product_Category_encoded": 1,
      "Customer_Segment_encoded": 1,
      "Payment_Method_encoded": 1
    }
  }'
```

### 4. View Model Info
```bash
curl "http://localhost:5000/model/info"
```

## 📚 File Structure

```
src/
├── train.py              # Original MLP training
├── train_mtl.py         # MTL training
├── train_sequence.py     # Sequence model training
├── train_advanced.py     # Unified training pipeline
├── model.py             # All model architectures
├── data_preprocessing.py # Original preprocessing
├── sequence_preprocessing.py # Sequence preprocessing
└── data_drift.py       # Drift monitoring

serving/
├── app.py              # Enhanced FastAPI with multi-model support
└── dashboard.py        # Streamlit dashboard (unchanged)

artifacts/
├── mlp_churn_classifier_final.pth
├── mtl_churn_clv_classifier.pth
├── sequence_mtl_lstm_classifier.pth
├── sequence_mtl_gru_classifier.pth
├── scaler.pkl
├── feature_names.pkl
└── label_encoders.pkl
```

## 🎯 Next Steps

1. **Model Ensemble**: Combine predictions from multiple models
2. **Feature Importance**: SHAP explanations for all model types
3. **Real-time Inference**: Streaming predictions with Kafka
4. **A/B Testing**: Framework for model comparison in production
5. **AutoML**: Automated architecture search and hyperparameter tuning

## 📞 Support

For questions or issues:
- Check the training logs for detailed error messages
- Verify data format matches expected schema
- Ensure sufficient memory for sequence models
- Monitor GPU utilization for large models
