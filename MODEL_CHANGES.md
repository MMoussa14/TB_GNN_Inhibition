# Molecular GNN — Program Additions (Original vs New)

A changelog describing what the new model/program adds or changes compared to the original.

## 🎯 Overview

The original program predicts activity (Active/Inactive) from molecular datasets.

The new program updates the pipeline to a multi-task / dual-head model that predicts:

- **Activity** (binary classification; logits)
- **Inhibition** (continuous score; normalized to **0–1**, trained as regression)

## ✅ What's New (All Implementation Changes)

### 1) Model outputs: single head → dual head

**Original**
- One MLP output layer: `fc3` producing a single logit for activity.

**New**
- Shared MLP trunk (`fc1`, `fc2`) plus two separate output heads:
  - `fc_activity`: activity logits (classification)
  - `fc_inhibition`: inhibition prediction (regression; output passed through sigmoid)

**Implication**
- `forward()` returns two tensors now:  
  `activity_out, inhibition_out` instead of a single `out`.

### 2) Forward pass changed: inhibition head uses sigmoid

**Original**
- `return x` where `x` is raw logits from `fc3`.

**New**
- Returns:
  - `activity_out = self.fc_activity(x)` *(raw logits)*
  - `inhibition_out = torch.sigmoid(self.fc_inhibition(x))` *(forces 0–1 range)*

**Implication**
- Inhibition is explicitly constrained to `[0, 1]` at inference time.

### 3) Dataset: inhibition values are now normalized + clipped

**Original**
- If `Inhibition at 1 uM` exists:
  - stores raw values (missing filled with 0)
  - no scaling/normalization

**New**
- If `Inhibition at 1 uM` exists:
  - reads raw values and **normalizes** them using fixed min/max:
    - `min_score = -65.54`
    - `max_score = 2011.49`
  - applies: `(raw - min) / (max - min)`
  - then **clips to [0, 1]**
  - prints a confirmation line with min/max of the tensor

**Implication**
- The regression target is consistently scaled to `[0, 1]` to match the sigmoid output.

### 4) Training loss: single objective → weighted multi-objective

**Original**
- `train_epoch()` uses:
  - `criterion = BCEWithLogitsLoss()`
  - loss computed on the single output vs `data.y`

**New**
- `train_epoch()` uses **two criteria**:
  - `criterion_activity = BCEWithLogitsLoss(...)`
  - `criterion_inhibition = MSELoss()`
- Computes:
  - `loss_activity` always
  - `loss_inhibition` only when inhibition scores exist and **only on active molecules**
- Combines losses with weight `alpha`:
  - `loss = alpha * loss_activity + (1 - alpha) * loss_inhibition`

**Implications**
- Multi-task learning (classification + regression).
- Regression is masked so it contributes only for active molecules.

### 5) Inhibition loss masking: regression only on active compounds

**Original**
- No inhibition regression task exists.

**New**
- In `train_epoch()`:
  - Creates `active_mask = data.y == 1`
  - If there are actives in the batch, computes regression loss only on those indices.
  - Otherwise uses `torch.tensor(0.0)`.

**Implication**
- The model is not penalized for inhibition predictions on inactive compounds.

### 6) Evaluate: returns extra regression metrics when available

**Original**
- `evaluate()` reports classification metrics:
  - AUC, Accuracy, Precision, Recall, F1

**New**
- Still reports the same classification metrics, plus optionally:
  - `inhibition_mse`
  - `inhibition_mae`
- These are computed only when scores are present and only on active molecules.

**Implication**
- Evaluation now covers both tasks; regression metrics are conditional.

### 7) Training API expanded: new knobs for imbalance + multitask weighting

**Original `train_model()` args**
- `epochs, batch_size, lr, hidden_channels, num_layers, gnn_type, test_size, val_size`

**New `train_model()` adds**
- `alpha=0.5` → controls classification vs regression weighting
- `pos_weight=None` → enables class imbalance handling for activity loss

**Implication**
- You can tune task trade-offs and address label imbalance explicitly.

### 8) Weighted classification loss support (pos_weight)

**Original**
- Always uses `BCEWithLogitsLoss()` without `pos_weight`.

**New**
- If `pos_weight` is provided:
  - constructs `pos_weight_tensor = torch.tensor([pos_weight], device=device)`
  - uses `BCEWithLogitsLoss(pos_weight=pos_weight_tensor)`
  - prints confirmation

**Implication**
- Helps with highly imbalanced datasets (common in screening tasks).

## 🧾 Summary of Changes

- **Original**: single-task activity classification
- **New**: dual-task activity classification + inhibition regression
- Adds:
  - normalization of inhibition targets
  - dual-head architecture
  - multi-objective loss with `alpha`
  - optional class imbalance weighting with `pos_weight`
  - extra evaluation metrics (MSE/MAE) on active compounds only

## 📁 Where the Differences Live

- **Model (`MolecularGNN`)**
  - Added: `fc_activity`, `fc_inhibition`
  - Changed: `forward()` returns two outputs; inhibition uses sigmoid

- **Dataset (`MolecularDataset`)**
  - Added: inhibition normalization + clipping + print

- **Training (`train_epoch`, `train_model`)**
  - Added: dual losses, active masking, weighted combination (`alpha`)
  - Added: optional `pos_weight` for BCE

- **Evaluation (`evaluate`)**
  - Added: inhibition prediction collection + MSE/MAE on actives

- **Main / Saving**
  - Changed: CSV path expectation + checkpoint filename
