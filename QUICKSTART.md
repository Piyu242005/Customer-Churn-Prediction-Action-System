# Quick Start Guide - MLP Churn Classifier

## 🚀 Get Up and Running in VS Code (Step-by-Step)

### Step 1: Open Project in VS Code
1. Open Visual Studio Code.
2. Go to **File > Open Folder...** and select the dataset folder/workspace.

### Step 2: Setup Python Environment
1. Open a new terminal in VS Code (`Ctrl+Shift+` ` or **Terminal > New Terminal**).
2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Train the Model (Optional if `.pth` exists)
Run the training script to generate the model weights:
```bash
python train.py
```

### Step 4: Start the API and Dashboard
Open **two** separate terminals in VS Code to run the backend API and frontend dashboard simultaneously.

**Terminal 1 (Flask API):**
```bash
python app.py
```
*The API will start running at `http://127.0.0.1:5000`*

**Terminal 2 (Streamlit Dashboard):**
```bash
streamlit run dashboard.py
```
*The dashboard will automatically open in your browser (usually at `http://localhost:8501`)*

---

### Alternative: Interactive Learning

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
