# Neural Network Churn Classifier (MLP)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Multilayer Perceptron (MLP) classifier for customer churn prediction achieving **89% test accuracy** and outperforming baseline Logistic Regression by **11%**.

## 🎯 Project Overview

Designed and trained a deep learning model using PyTorch on a 10,000-record customer dataset to predict customer churn. The project demonstrates advanced feature engineering, neural network architecture design, and optimization techniques including ReLU activation, dropout regularization, and Adam optimizer with learning rate scheduling.

### Key Results
- ✅ **Test Accuracy: 89%**
- ✅ **11% improvement** over baseline Logistic Regression (78%)
- ✅ Robust cross-validation performance
- ✅ High generalization across different data splits

## 📊 Performance Metrics

| Metric | MLP Classifier | Logistic Regression | Improvement |
|--------|---------------|---------------------|-------------|
| **Accuracy** | **89%** | 78% | **+11%** |
| Precision | 0.87 | 0.75 | +12% |
| Recall | 0.85 | 0.72 | +13% |
| F1-Score | 0.86 | 0.73 | +13% |
| ROC-AUC | 0.92 | 0.82 | +10% |

## 🏗️ Architecture

### MLP Model Structure
```
Input Layer (16 features)
    ↓
Hidden Layer 1 (128 neurons) → ReLU → Dropout(0.3)
    ↓
Hidden Layer 2 (64 neurons) → ReLU → Dropout(0.3)
    ↓
Hidden Layer 3 (32 neurons) → ReLU → Dropout(0.3)
    ↓
Output Layer (1 neuron) → Sigmoid
```

### Key Technical Components

- **Activation Function:** ReLU (Rectified Linear Unit) for non-linearity and fast convergence
- **Regularization:** Dropout (30%) to prevent overfitting
- **Optimizer:** Adam with weight decay (L2 regularization)
- **Learning Rate Scheduler:** ReduceLROnPlateau (dynamic learning rate adjustment)
- **Loss Function:** Binary Cross-Entropy (BCE)
- **Weight Initialization:** Xavier/Glorot initialization

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Development:** Jupyter Notebook

## 📁 Project Structure

```
Neural Network Churn Classifier (MLP)/
│
├── mlp_churn_classifier.ipynb      # Main Jupyter Notebook (comprehensive analysis)
├── Business_Analytics_Dataset_10000_Rows.csv  # Dataset
│
├── model.py                         # MLP model architecture
├── data_preprocessing.py            # Data loading and feature engineering
├── train.py                         # Training script with optimizer & scheduling
├── evaluate.py                      # Evaluation metrics and visualizations
│
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Piyu242005/neural-network-churn.git
cd neural-network-churn
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Option 1: Jupyter Notebook (Recommended for Experimentation)

Launch Jupyter Notebook for interactive exploration:

```bash
jupyter notebook mlp_churn_classifier.ipynb
```

The notebook contains:
- Data exploration and visualization
- Feature engineering process
- Model architecture details
- Training with learning curves
- Comprehensive evaluation
- Baseline comparison
- Cross-validation analysis

### Option 2: Python Scripts

#### Train the model:
```bash
python train.py
```

This will:
- Load and preprocess the data
- Train the MLP classifier
- Save the trained model as `mlp_churn_classifier.pth`
- Generate training curves

#### Evaluate the model:
```bash
python evaluate.py
```

This will:
- Load the trained model
- Generate comprehensive evaluation metrics
- Create visualization plots (confusion matrix, ROC curve, etc.)

### Using the Trained Model

```python
import torch
from model import MLPClassifier

# Load trained model
checkpoint = torch.load('mlp_churn_classifier.pth')
model = MLPClassifier(input_dim=16, hidden_dims=[128, 64, 32], dropout_rate=0.3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(new_data)
```

## 📈 Feature Engineering

The dataset is transactional, so churn labels were engineered from customer behavior:

### Engineered Features

**Numerical Features:**
- Total orders
- Total revenue, average revenue, revenue std
- Total profit, average profit
- Average discount rate
- Total quantity, average quantity
- Days since last purchase
- Customer lifetime (days)
- Purchase frequency

**Categorical Features (encoded):**
- Region
- Product category
- Customer segment  
- Payment method

### Churn Definition

A customer is classified as **churned** if:
- No purchase in the last **90 days**, OR
- Total profit in the **bottom 30%**, OR
- Fewer than **3 orders** AND no activity in **60 days**

## 🎓 Model Training Details

### Hyperparameters

```python
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
HIDDEN_DIMS = [128, 64, 32]
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15
```

### Training Features

- **Early Stopping:** Prevents overfitting by stopping when validation loss plateaus
- **Learning Rate Scheduling:** ReduceLROnPlateau reduces LR by 50% after 5 epochs of no improvement
- **Cross-Validation:** 5-fold CV to ensure robust generalization
- **Batch Training:** Mini-batch gradient descent with batch size 32

## 📊 Results & Visualizations

### Training Curves
![Training Curves](training_curves.png)

### Confusion Matrix
Shows the model's prediction accuracy across both classes (Active vs Churned customers).

### ROC Curve
Demonstrates excellent discriminative ability with AUC = 0.92.

### Feature Importance
Analysis reveals which features contribute most to churn predictions.

## 🔍 Key Insights

1. **Days since last purchase** is the strongest predictor of churn
2. **Total profit** and **purchase frequency** are also highly predictive
3. The MLP captures complex non-linear relationships better than logistic regression
4. Dropout regularization significantly improves generalization
5. Learning rate scheduling stabilizes training and improves convergence

## 🚀 Future Enhancements

- [ ] Implement LSTM for temporal pattern analysis
- [ ] Add SMOTE for class imbalance handling
- [ ] Deploy model as REST API using Flask/FastAPI
- [ ] Experiment with ensemble methods
- [ ] A/B testing in production environment
- [ ] Real-time prediction dashboard

## 📝 Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Piyu**  
- GitHub: [@Piyu242005](https://github.com/Piyu242005)
- Project Link: [neural-network-churn](https://github.com/Piyu242005/neural-network-churn)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Scikit-learn for preprocessing and evaluation tools
- The open-source community for inspiration and resources

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
