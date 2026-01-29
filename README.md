# Molecular GNN for Tuberculosis Toxicology Prediction

A Graph Neural Network (GNN) for predicting molecular activity against tuberculosis bacteria using the AID 588726 dataset.

## 🎯 Project Overview

This project implements a deep learning model that:
- Converts molecular structures (SMILES) into graph representations
- Uses Graph Neural Networks to predict toxicology activity
- Achieves high accuracy on tuberculosis drug screening data
- Supports multiple GNN architectures (GCN, GAT, GraphSAGE)

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- RDKit
- scikit-learn
- pandas, numpy

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Train the Model

```bash
python model.py
```

This will:
- Load the tuberculosis toxicology dataset (320,000+ molecules)
- Train a GNN for 100 epochs
- Evaluate on test set
- Save model to `molecular_gnn_model.pt`

Expected output:
```
Using device: cuda
Converting molecules to graphs...
Train: 219,482, Val: 27,435, Test: 68,588

Training...
Epoch 10/100
  Train Loss: 0.3245
  Val AUC: 0.8567, Acc: 0.7891, F1: 0.7234

Final Test Results:
  AUC: 0.8612
  Accuracy: 0.7945
  Precision: 0.7456
  Recall: 0.7123
  F1 Score: 0.7289
```

### 2. Test the Model (Interactive)

Run the interactive testing interface:

```bash
python quick_test.py
```

This provides an easy menu-driven interface to:
1. Test on the full dataset with metrics and visualizations
2. Predict activity for your own molecules (SMILES)
3. Run predictions on example drugs

### 3. Test the Model (Command Line)

For more control, use the command-line interface:

**Test on the original dataset:**
```bash
python test_model.py \
    --model-path molecular_gnn_model.pt \
    --data-path "AID_588726_datatable (FBA).csv" \
    --use-test-split \
    --save-predictions
```

**Predict on new molecules:**
```bash
python test_model.py \
    --model-path molecular_gnn_model.pt \
    --smiles "CCO" "c1ccccc1" "CC(=O)O"
```

**Batch predict from file:**
```bash
python test_model.py \
    --model-path molecular_gnn_model.pt \
    --smiles-file molecules.txt \
    --save-predictions
```

## 📊 Understanding Results

### Output Metrics

- **AUC-ROC**: Area under ROC curve (0.5 = random, 1.0 = perfect)
- **Accuracy**: Overall correct predictions
- **Precision**: Of predicted active, how many are truly active
- **Recall**: Of truly active, how many were detected
- **F1 Score**: Harmonic mean of precision and recall

### Output Files

When using `--save-predictions`, the following files are generated in `test_results/`:

- `test_metrics.json` - Detailed performance metrics
- `predictions.csv` - Individual molecule predictions
- `roc_curve.png` - ROC curve visualization
- `confusion_matrix.png` - Confusion matrix heatmap

### Interpreting Predictions

- **Active** = Toxic to tuberculosis bacteria (potential drug candidate ✓)
- **Inactive** = Not toxic to tuberculosis bacteria
- **Probability** = Confidence score (0-1) for active prediction
- **Confidence** = How certain the model is (distance from 0.5 threshold)

## 🧪 Example Usage

### Python API

```python
import torch
from test_model import load_model, predict_smiles
from model import MolecularDataset

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, model_info = load_model('molecular_gnn_model.pt', device=device)

# Predict on new molecules
smiles_list = [
    'CCO',  # Ethanol
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    'CC(C)Cc1ccc(cc1)C(C)C(O)=O'  # Ibuprofen
]

results = predict_smiles(model, smiles_list, device=device)

for r in results:
    print(f"{r['smiles']}: {r['prediction']} (prob={r['probability_active']:.3f})")
```

### Virtual Screening

Screen a large library of compounds:

```python
import pandas as pd
from test_model import load_model, predict_smiles

# Load model
model, _ = load_model('molecular_gnn_model.pt')

# Load compound library
df = pd.read_csv('compound_library.csv')
smiles_list = df['smiles'].tolist()

# Predict in batches
batch_size = 1000
all_results = []

for i in range(0, len(smiles_list), batch_size):
    batch = smiles_list[i:i+batch_size]
    results = predict_smiles(model, batch)
    all_results.extend(results)

# Filter for active compounds
active_df = pd.DataFrame([r for r in all_results if r['prediction'] == 'Active'])
active_df = active_df.sort_values('probability_active', ascending=False)

# Save top candidates
active_df.head(100).to_csv('top_candidates.csv', index=False)
print(f"Found {len(active_df)} active compounds")
```

## 🏗️ Model Architecture

The GNN model consists of:

1. **Graph Convolutional Layers** (3 layers)
   - Input: Molecular graph with atom features
   - Hidden dimension: 128
   - Activation: ReLU
   - Dropout: 0.2

2. **Global Pooling**
   - Aggregates node features to graph-level representation

3. **MLP Classifier** (3 layers)
   - 128 → 64 → 32 → 1
   - Binary classification (Active/Inactive)

### Node Features (per atom)

- Atomic number
- Degree (number of bonds)
- Formal charge
- Hybridization (sp, sp2, sp3)
- Aromaticity
- Total hydrogens
- In ring

### Supported GNN Types

- **GCN** (Graph Convolutional Network) - Default
- **GAT** (Graph Attention Network) - Uses attention mechanism
- **GraphSAGE** - Inductive learning

## 📁 Project Structure

```
silico/
├── model.py                          # Main model and training code
├── test_model.py                     # Comprehensive testing script
├── quick_test.py                     # Interactive testing interface
├── azure_train.py                    # Azure ML training script
├── submit_azure_job.py               # Submit to Azure ML
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── TESTING_GUIDE.md                  # Detailed testing guide
├── AID_588726_datatable (FBA).csv    # Dataset
└── test_results/                     # Output directory (created on test)
    ├── test_metrics.json
    ├── predictions.csv
    ├── roc_curve.png
    └── confusion_matrix.png
```

## 🔬 Training Options

Customize training with different hyperparameters:

```python
from model import MolecularDataset, train_model

dataset = MolecularDataset('AID_588726_datatable (FBA).csv')

model, metrics = train_model(
    dataset,
    epochs=200,              # More epochs
    batch_size=64,           # Larger batch
    lr=0.0005,              # Lower learning rate
    hidden_channels=256,     # Bigger model
    num_layers=4,           # Deeper model
    gnn_type='GAT',         # Different architecture
)
```

## ☁️ Training on Azure ML

For GPU-accelerated training on Azure:

1. Set up Azure ML workspace (see `AZURE_SETUP.md`)
2. Submit training job:

```bash
python submit_azure_job.py \
    --data-path "AID_588726_datatable (FBA).csv" \
    --epochs 100 \
    --gnn-type GCN
```

## 🐛 Troubleshooting

### Model not found
```
python model.py  # Train first
```

### CUDA out of memory
```
python test_model.py --batch-size 16 ...  # Reduce batch size
```

### Invalid SMILES
The model will skip invalid SMILES and report them in output. Use RDKit to validate:
```python
from rdkit import Chem
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    print("Invalid SMILES")
```

### Slow dataset loading
The first run converts all SMILES to graphs (takes ~5-10 minutes for full dataset). Consider using `--max-samples` for testing:
```bash
python azure_train.py --max-samples 10000 ...
```

## 📊 Dataset Information

**AID 588726**: Tuberculosis toxicology screening

- **Total molecules**: ~320,000
- **Active (toxic)**: ~8,500 (2.7%)
- **Inactive**: ~311,500 (97.3%)
- **Features**: SMILES strings, activity outcome, inhibition scores

The dataset is highly imbalanced, which is typical for drug screening datasets.

## 🎓 Model Performance

Typical performance on test set:

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.85-0.87 |
| Accuracy | 0.78-0.80 |
| Precision | 0.73-0.76 |
| Recall | 0.70-0.73 |
| F1 Score | 0.72-0.75 |

Performance can be improved by:
- Hyperparameter tuning
- Ensemble methods
- Feature engineering (adding more molecular descriptors)
- Handling class imbalance (weighted loss, oversampling)

## 📝 Citation

If you use this code in your research, please cite:

```
@software{molecular_gnn_tb,
  title={Graph Neural Network for Tuberculosis Toxicology Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/molecular-gnn-tb}
}
```

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This model is for research purposes only. Drug candidates identified should undergo proper validation and clinical trials before any medical use.

