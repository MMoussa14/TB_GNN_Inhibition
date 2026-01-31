# TB_GNN_Inhibition

Uses original Graph Neural Network (GNN) model with the addition of inhibition and weighted loss.

## 📝 Citation

Credits go to Zidane Virani for the majority of the model's framework. 

Inhibition and weighted loss have been added to support the original model.

## 🎯 Project Overview

This project implements a deep learning model that:
- Converts molecular structures (SMILES) into graph representations
- Uses Graph Neural Networks to predict toxicology activity
- Applies weighted loss to reduce false negatives
- Records normalised inhibition values
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
python InhModel.py
```

This will:
- Load the tuberculosis toxicology dataset (320,000+ molecules)
- Train a GNN for 100 epochs
- Evaluate on test set
- Save model to `model_exp2_weighted_2x.pt`

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
