# Quick Start Guide - MLP Churn Classifier

## 🚀 Get Up and Running in 3 Minutes

### Step 1: Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### Step 2: Choose Your Approach

#### A. Interactive Notebook (Recommended for Learning)

```bash
# Launch Jupyter
jupyter notebook mlp_churn_classifier.ipynb
```

Then run all cells to:
- ✅ Explore the data
- ✅ Engineer churn labels
- ✅ Train the MLP model
- ✅ Compare with baseline
- ✅ Visualize results

#### B. Command Line Scripts (For Quick Training)

```bash
# Train the model
python train.py

# Evaluate the model
python evaluate.py
```

### Step 3: View Results

After training, you'll have:
- `mlp_churn_classifier.pth` - Trained model weights
- `training_curves.png` - Loss and accuracy plots
- `confusion_matrix.png` - Classification performance
- `roc_curve.png` - ROC analysis

## 📊 Expected Results

```
Test Accuracy: ~89%
Training Time: ~2-3 minutes (CPU) / ~30 seconds (GPU)
```

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `mlp_churn_classifier.ipynb` | **Start here!** Complete analysis |
| `train.py` | Train model from command line |
| `evaluate.py` | Generate evaluation metrics |
| `model.py` | MLP architecture definition |
| `data_preprocessing.py` | Feature engineering |
| `config.py` | Hyperparameter configuration |

## 🔧 Customization

### Change Model Architecture

Edit `config.py`:
```python
MODEL_CONFIG = {
    'hidden_dims': [256, 128, 64],  # Deeper network
    'dropout_rate': 0.4,  # More regularization
}
```

### Adjust Training

Edit `config.py`:
```python
TRAINING_CONFIG = {
    'batch_size': 64,  # Larger batches
    'learning_rate': 0.0001,  # Slower learning
    'epochs': 150  # More training
}
```

## 🐛 Troubleshooting

**Issue:** CUDA out of memory
- **Solution:** Reduce batch_size in `config.py`

**Issue:** Model not converging
- **Solution:** Try lower learning rate (0.0001)

**Issue:** Overfitting
- **Solution:** Increase dropout_rate to 0.4-0.5

## 📚 Next Steps

1. ✅ Run the notebook to understand the full pipeline
2. ✅ Experiment with hyperparameters in `config.py`
3. ✅ Try different model architectures
4. ✅ Analyze feature importance
5. ✅ Test on your own data

## 🎓 Learning Resources

- **PyTorch Docs:** https://pytorch.org/docs/
- **Neural Networks:** http://neuralnetworksanddeeplearning.com/
- **Churn Prediction:** Related research papers and articles

## 💡 Pro Tips

1. **Start with the notebook** - It has detailed explanations
2. **Use GPU if available** - 5-10x faster training
3. **Monitor validation loss** - Early stopping prevents overfitting
4. **Cross-validate** - Ensures robust performance
5. **Save your best model** - Use checkpointing

---

**Need help?** Open an issue on GitHub: [neural-network-churn](https://github.com/Piyu242005/neural-network-churn)
