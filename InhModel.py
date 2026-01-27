import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, SAGEConv
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')


class MolecularGNN(nn.Module):
    """
    Graph Neural Network for molecular property prediction.
    Supports multiple GNN architectures: GCN, GAT, GraphSAGE
    """
    def __init__(self, num_node_features, hidden_channels=128, num_layers=3,
                 dropout=0.2, gnn_type='GCN', num_heads=4):
        super(MolecularGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type

        # Build GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        if gnn_type == 'GCN':
            self.convs.append(GCNConv(num_node_features, hidden_channels))
        elif gnn_type == 'GAT':
            self.convs.append(GATConv(num_node_features, hidden_channels // num_heads,
                                     heads=num_heads, dropout=dropout))
        elif gnn_type == 'GraphSAGE':
            self.convs.append(SAGEConv(num_node_features, hidden_channels))

        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            if gnn_type == 'GCN':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            elif gnn_type == 'GAT':
                self.convs.append(GATConv(hidden_channels, hidden_channels // num_heads,
                                         heads=num_heads, dropout=dropout))
            elif gnn_type == 'GraphSAGE':
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))

            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # MLP for final prediction (shared layers)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, hidden_channels // 4)

        # Two separate output heads
        self.fc_activity = nn.Linear(hidden_channels // 4, 1) # Binary classification
        self.fc_inhibition = nn.Linear(hidden_channels // 4, 1) # Inhibition prediction

    def forward(self, x, edge_index, batch):
        # GNN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, batch)

        # MLP layers (shared representation)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Dual outputs
        activity_out = self.fc_activity(x)  # Binary classification logits
        inhibition_out = torch.sigmoid(self.fc_inhibition(x))  # Regression (0-1 range)

        return activity_out, inhibition_out


def smiles_to_graph(smiles):
    """
    Convert SMILES string to PyTorch Geometric graph representation.

    Args:
        smiles: SMILES string representing a molecule

    Returns:
        Data object with node features and edge indices
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features: atom type, degree, formal charge, hybridization, aromaticity
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),  # Atomic number
            atom.GetDegree(),     # Degree
            atom.GetFormalCharge(),  # Formal charge
            int(atom.GetHybridization()),  # Hybridization
            int(atom.GetIsAromatic()),  # Aromaticity
            atom.GetTotalNumHs(),  # Total hydrogens
            int(atom.IsInRing()),  # In ring
        ]
        node_features.append(features)

    node_features = torch.tensor(node_features, dtype=torch.float)

    # Edge indices (bonds)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # Undirected graph

    if len(edge_indices) == 0:
        # Handle single-atom molecules
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=node_features, edge_index=edge_index)


class MolecularDataset(Dataset):
    """
    PyTorch Geometric Dataset for molecular data from CSV file.
    """
    def __init__(self, csv_file, transform=None, pre_transform=None):
        super(MolecularDataset, self).__init__(None, transform, pre_transform)

        # Read CSV
        df = pd.read_csv(csv_file, skiprows=[1, 2, 3, 4])  # Skip metadata rows

        # Clean data
        df = df.dropna(subset=['PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME'])

        # Convert activity to binary (Active=1, Inactive=0)
        df['label'] = (df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active').astype(int)

        # Store SMILES and labels
        self.smiles_list = df['PUBCHEM_EXT_DATASOURCE_SMILES'].tolist()
        self.labels = torch.tensor(df['label'].values, dtype=torch.float)

        # Store activity scores if available
        if 'Inhibition at 1 uM' in df.columns:
            raw_scores = df['Inhibition at 1 uM'].fillna(0).values
            
            # Normalise scores to 0-1 range
            # Original range: -65.54 to 2011.49
            min_score = -65.54
            max_score = 2011.49
            normalized_scores = (raw_scores - min_score) / (max_score - min_score)
            
            # Clip to ensure values are in [0, 1]
            import numpy as np
            normalized_scores = np.clip(normalized_scores, 0, 1)
            
            self.scores = torch.tensor(normalized_scores, dtype=torch.float)
            print(f"✓ Normalized inhibition scores - Min: {self.scores.min():.4f}, Max: {self.scores.max():.4f}")
        else:
            self.scores = None

        # Convert SMILES to graphs
        self.graphs = []
        valid_indices = []

        print(f"Converting {len(self.smiles_list)} molecules to graphs...")
        for idx, smiles in enumerate(self.smiles_list):
            if idx % 10000 == 0:
                print(f"Processed {idx}/{len(self.smiles_list)} molecules")

            graph = smiles_to_graph(smiles)
            if graph is not None:
                self.graphs.append(graph)
                valid_indices.append(idx)

        # Keep only valid samples
        self.labels = self.labels[valid_indices]
        if self.scores is not None:
            self.scores = self.scores[valid_indices]

        print(f"Successfully converted {len(self.graphs)}/{len(self.smiles_list)} molecules")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        graph = self.graphs[idx].clone()
        graph.y = self.labels[idx]
        if self.scores is not None:
            graph.score = self.scores[idx]
        return graph


def train_epoch(model, loader, optimizer, criterion_activity, criterion_inhibition, device, alpha=0.5):
    """Train for one epoch with dual objectives
    
    Args:
        alpha: Weight for activity loss (1-alpha for inhibition loss)
               alpha=0.5 means equal weight to both tasks
    """
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass - now returns two outputs
        activity_out, inhibition_out = model(data.x, data.edge_index, data.batch)
        
        # Activity loss (binary classification)
        loss_activity = criterion_activity(activity_out.squeeze(), data.y)
        
        # Inhibition loss (regression) - only for active compounds
        if hasattr(data, 'score'):
            # Only compute inhibition loss for active molecules (where inhibition matters)
            active_mask = data.y == 1
            if active_mask.sum() > 0:
                loss_inhibition = criterion_inhibition(
                    inhibition_out.squeeze()[active_mask],
                    data.score[active_mask]
                )
            else:
                loss_inhibition = torch.tensor(0.0, device=device)
        else:
            loss_inhibition = torch.tensor(0.0, device=device)
        
        # Combined loss (weighted)
        loss = alpha * loss_activity + (1 - alpha) * loss_inhibition

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """Evaluate model with dual outputs"""
    model.eval()
    activity_predictions = []
    inhibition_predictions = []
    labels = []
    scores = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            activity_out, inhibition_out = model(data.x, data.edge_index, data.batch)

            # Activity predictions (apply sigmoid for probabilities)
            activity_predictions.extend(torch.sigmoid(activity_out).cpu().numpy())
            labels.extend(data.y.cpu().numpy())
            
            # Inhibition predictions (already in 0-1 range from sigmoid in forward)
            inhibition_predictions.extend(inhibition_out.cpu().numpy())
            
            # Store true inhibition scores if available
            if hasattr(data, 'score'):
                scores.extend(data.score.cpu().numpy())

    activity_predictions = np.array(activity_predictions).flatten()
    inhibition_predictions = np.array(inhibition_predictions).flatten()
    labels = np.array(labels)

    # Calculate classification metrics
    auc = roc_auc_score(labels, activity_predictions)
    binary_preds = (activity_predictions > 0.5).astype(int)
    acc = accuracy_score(labels, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, binary_preds, average='binary')

    # Calculate inhibition metrics (if we have ground truth scores)
    metrics = {
        'auc': auc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    if len(scores) > 0:
        scores = np.array(scores)
        # MSE for inhibition predictions (only on active molecules)
        active_mask = labels == 1
        if active_mask.sum() > 0:
            inhibition_mse = np.mean((inhibition_predictions[active_mask] - scores[active_mask]) ** 2)
            inhibition_mae = np.mean(np.abs(inhibition_predictions[active_mask] - scores[active_mask]))
            metrics['inhibition_mse'] = inhibition_mse
            metrics['inhibition_mae'] = inhibition_mae

    return metrics


def train_model(dataset, epochs=100, batch_size=32, lr=0.001,
                hidden_channels=128, num_layers=3, gnn_type='GCN',
                test_size=0.2, val_size=0.1, alpha=0.5, pos_weight=None):
    """
    Complete training pipeline for molecular GNN

    Args:
        dataset: MolecularDataset instance
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        hidden_channels: Hidden dimension size
        num_layers: Number of GNN layers
        gnn_type: Type of GNN ('GCN', 'GAT', 'GraphSAGE')
        test_size: Test set proportion
        val_size: Validation set proportion
    """
    from torch_geometric.loader import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Split dataset
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size/(1-test_size),
                                          random_state=42)

    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Initialize model
    num_node_features = dataset[0].x.shape[1]
    model = MolecularGNN(num_node_features, hidden_channels=hidden_channels,
                        num_layers=num_layers, gnn_type=gnn_type).to(device)

    print(f"\nModel architecture ({gnn_type}):")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Two separate loss functions
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], device=device)
        criterion_activity = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"Using weighted loss with pos_weight={pos_weight}")
    else:
        criterion_activity = nn.BCEWithLogitsLoss()
    criterion_inhibition = nn.MSELoss()  # For regression

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                        factor=0.5, patience=10)

    # Training loop
    best_val_auc = 0
    best_model_state = None

    print("\nStarting training...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, 
                                criterion_activity, criterion_inhibition, 
                                device, alpha=alpha)  # alpha controls task weighting
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_metrics['auc'])

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                f"F1: {val_metrics['f1']:.4f}", end='')
    
            # Print inhibition metrics if available
            if 'inhibition_mse' in val_metrics:
                print(f", Inhib MSE: {val_metrics['inhibition_mse']:.4f}")
            else:
                print()

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "="*60)
    print("Final Test Results:")
    print("Classification Metrics:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")

    if 'inhibition_mse' in test_metrics:
        print("\nInhibition Prediction Metrics (Active Compounds Only):")
        print(f"  MSE: {test_metrics['inhibition_mse']:.4f}")
        print(f"  MAE: {test_metrics['inhibition_mae']:.4f}")

    print("="*60)

    return model, test_metrics


if __name__ == "__main__":
    # Example usage - UPDATE THIS PATH
    csv_file = "C:\Users\georgemoussa\Downloads\Python313\AID_588726_datatable (FBA).csv" # FIle is now in the same directory

    print("Loading and preprocessing dataset...")
    dataset = MolecularDataset(csv_file)

    print("\nTraining GNN model...")
    model, metrics = train_model(
        dataset,
        epochs=100,
        batch_size=32,
        lr=0.001,
        hidden_channels=128,
        num_layers=3,
        gnn_type='GCN',  # Try 'GAT' or 'GraphSAGE' for different architectures
    )

    # Save model
    torch.save(model.state_dict(), 'molecular_gnn_model.pt')
    print("\nModel saved to 'molecular_gnn_model.pt'")
