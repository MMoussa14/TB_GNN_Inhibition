"""
Test script for trained Molecular GNN models
Supports testing on datasets and individual molecule predictions
UPDATED: Supports dual-output models (activity + inhibition)
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_recall_fscore_support,
                            confusion_matrix, classification_report, roc_curve, mean_squared_error,
                            mean_absolute_error)
import matplotlib.pyplot as plt
import seaborn as sns

# Import from model.py
from InhModel import MolecularGNN, MolecularDataset, smiles_to_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from rdkit import Chem


def load_model(model_path, model_info_path=None, device='cpu'):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to saved model weights (.pt file)
        model_info_path: Path to model_info.json (optional)
        device: Device to load model on
    
    Returns:
        Loaded model and model info dict
    """
    print(f"Loading model from {model_path}...")
    
    # Try to load model info
    if model_info_path and Path(model_info_path).exists():
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        print(f"Model info loaded: {model_info}")
    else:
        # Use default values
        print("Warning: model_info.json not found, using default architecture")
        model_info = {
            'num_node_features': 7,
            'hidden_channels': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'gnn_type': 'GCN'
        }
    
    # Initialize model
    model = MolecularGNN(
        num_node_features=model_info['num_node_features'],
        hidden_channels=model_info['hidden_channels'],
        num_layers=model_info['num_layers'],
        dropout=model_info.get('dropout', 0.2),
        gnn_type=model_info.get('gnn_type', 'GCN')
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model, model_info


def test_on_dataset(model, dataset, test_indices=None, batch_size=32, device='cpu',
                   save_predictions=False, output_dir='./test_results'):
    """
    Test model on a dataset and compute comprehensive metrics
    Supports dual-output models (activity + inhibition)
    
    Args:
        model: Trained model
        dataset: MolecularDataset instance
        test_indices: Optional list of indices to test on (if None, uses all data)
        batch_size: Batch size for testing
        device: Device to run on
        save_predictions: Whether to save predictions to file
        output_dir: Directory to save results
    
    Returns:
        Dictionary of metrics and predictions
    """
    print(f"\nTesting on dataset with {len(dataset)} samples...")
    
    # Create test loader
    if test_indices is not None:
        test_dataset = [dataset[i] for i in test_indices]
    else:
        test_dataset = dataset
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate
    model.eval()
    predictions = []
    probabilities = []
    labels = []
    inhibitions = []
    true_inhibitions = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # Model returns (activity_logits, inhibition_values) for dual-output models
            model_output = model(data.x, data.edge_index, data.batch)
            
            # Check if dual output (tuple/list) or single output (tensor)
            if isinstance(model_output, (tuple, list)):
                activity_out, inhibition_out = model_output
                # Activity predictions
                probs = torch.sigmoid(activity_out).cpu().numpy().flatten()
                # Inhibition predictions
                inhibs = inhibition_out.cpu().numpy().flatten()
                inhibitions.extend(inhibs)
                
                # Get true inhibition scores if available
                if hasattr(data, 'inhibition_score'):
                    true_inhibitions.extend(data.inhibition_score.cpu().numpy())
            else:
                # Single output model (backward compatibility)
                probs = torch.sigmoid(model_output).cpu().numpy().flatten()
            
            probabilities.extend(probs)
            predictions.extend((probs > 0.5).astype(int))
            labels.extend(data.y.cpu().numpy())
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    labels = np.array(labels)
    
    # Calculate classification metrics
    auc = roc_auc_score(labels, probabilities)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions,
                                                                average='binary', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics = {
        'auc': float(auc),
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    # Calculate inhibition metrics if available
    if len(inhibitions) > 0:
        inhibitions = np.array(inhibitions)
        active_mask = labels == 1
        
        if active_mask.sum() > 0:
            active_inhibitions = inhibitions[active_mask]
            metrics['inhibition_mean'] = float(active_inhibitions.mean())
            metrics['inhibition_std'] = float(active_inhibitions.std())
            metrics['inhibition_min'] = float(active_inhibitions.min())
            metrics['inhibition_max'] = float(active_inhibitions.max())
            
            # If true inhibition scores available, calculate MSE/MAE
            if len(true_inhibitions) > 0:
                true_inhibitions = np.array(true_inhibitions)
                active_true_inhibitions = true_inhibitions[active_mask]
                metrics['inhibition_mse'] = float(mean_squared_error(active_true_inhibitions, active_inhibitions))
                metrics['inhibition_mae'] = float(mean_absolute_error(active_true_inhibitions, active_inhibitions))
    
    # Print results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Total samples: {len(labels)}")
    print(f"Active (positive): {int(labels.sum())} ({labels.mean()*100:.1f}%)")
    print(f"Inactive (negative): {int(len(labels) - labels.sum())} ({(1-labels.mean())*100:.1f}%)")
    print("\nClassification Metrics:")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    
    # Print inhibition metrics if available
    if 'inhibition_mean' in metrics:
        print("\nInhibition Prediction Metrics (Active Compounds):")
        print(f"  Mean: {metrics['inhibition_mean']:.4f}")
        print(f"  Std:  {metrics['inhibition_std']:.4f}")
        print(f"  Min:  {metrics['inhibition_min']:.4f}")
        print(f"  Max:  {metrics['inhibition_max']:.4f}")
        if 'inhibition_mse' in metrics:
            print(f"  MSE:  {metrics['inhibition_mse']:.4f}")
            print(f"  MAE:  {metrics['inhibition_mae']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"  True Positives:  {tp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print("="*70 + "\n")
    
    # Generate classification report
    print("Detailed Classification Report:")
    print(classification_report(labels, predictions, target_names=['Inactive', 'Active'], zero_division=0))
    
    # Save results if requested
    if save_predictions:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = output_path / 'test_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics saved to {metrics_file}")
        
        # Save predictions
        pred_data = {
            'true_label': labels,
            'predicted_label': predictions,
            'probability_active': probabilities
        }
        if len(inhibitions) > 0:
            pred_data['predicted_inhibition'] = inhibitions
        if len(true_inhibitions) > 0:
            pred_data['true_inhibition'] = true_inhibitions
            
        pred_df = pd.DataFrame(pred_data)
        pred_file = output_path / 'predictions.csv'
        pred_df.to_csv(pred_file, index=False)
        print(f"✓ Predictions saved to {pred_file}")
        
        # Plot ROC curve
        plot_roc_curve(labels, probabilities, output_path)
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, output_path)
        
        # Plot inhibition distribution if available
        if len(inhibitions) > 0:
            plot_inhibition_distribution(labels, inhibitions, output_path)
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'probabilities': probabilities,
        'labels': labels,
        'inhibitions': inhibitions if len(inhibitions) > 0 else None
    }


def predict_smiles(model, smiles_list, device='cpu'):
    """
    Make predictions on a list of SMILES strings
    Supports dual-output models (activity + inhibition)
    
    Args:
        model: Trained model
        smiles_list: List of SMILES strings
        device: Device to run on
    
    Returns:
        List of predictions (dicts with smiles, probability, prediction, inhibition)
    """
    print(f"\nMaking predictions on {len(smiles_list)} molecules...")
    
    results = []
    model.eval()
    
    with torch.no_grad():
        for smiles in smiles_list:
            # Convert to graph
            graph = smiles_to_graph(smiles)
            
            if graph is None:
                results.append({
                    'smiles': smiles,
                    'valid': False,
                    'prediction': None,
                    'probability': None,
                    'error': 'Invalid SMILES'
                })
                continue
            
            # Add batch dimension
            graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
            graph = graph.to(device)
            
            # Predict
            model_output = model(graph.x, graph.edge_index, graph.batch)
            
            # Check if dual output
            if isinstance(model_output, (tuple, list)):
                activity_out, inhibition_out = model_output
                prob = torch.sigmoid(activity_out).item()
                inhibition = inhibition_out.item()
            else:
                prob = torch.sigmoid(model_output).item()
                inhibition = None
            
            pred = 'Active' if prob > 0.5 else 'Inactive'
            
            result = {
                'smiles': smiles,
                'valid': True,
                'prediction': pred,
                'probability_active': prob,
                'confidence': abs(prob - 0.5) * 2  # Confidence score 0-1
            }
            
            if inhibition is not None:
                result['inhibition_strength'] = inhibition
            
            results.append(result)
    
    return results


def plot_roc_curve(labels, probabilities, output_dir):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})', color='#2E86AB')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Molecular GNN', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'roc_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to {output_path}")
    plt.close()


def plot_confusion_matrix(cm, output_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Inactive', 'Active'],
                yticklabels=['Inactive', 'Active'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix - Molecular GNN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_path}")
    plt.close()


def plot_inhibition_distribution(labels, inhibitions, output_dir):
    """Plot distribution of predicted inhibition scores"""
    inhibitions = np.array(inhibitions)
    labels = np.array(labels)
    
    active_inhibitions = inhibitions[labels == 1]
    inactive_inhibitions = inhibitions[labels == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(inactive_inhibitions, bins=50, alpha=0.6, label='Inactive', color='#A23B72')
    plt.hist(active_inhibitions, bins=50, alpha=0.6, label='Active', color='#06A77D')
    plt.xlabel('Predicted Inhibition Strength', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Predicted Inhibition Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'inhibition_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Inhibition distribution saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test trained Molecular GNN (supports dual-output models)')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to saved model (.pt file)')
    parser.add_argument('--model-info', type=str, default=None,
                       help='Path to model_info.json')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to CSV file for testing')
    parser.add_argument('--smiles', type=str, nargs='+', default=None,
                       help='SMILES strings to predict')
    parser.add_argument('--smiles-file', type=str, default=None,
                       help='File with SMILES strings (one per line)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--output-dir', type=str, default='./test_results',
                       help='Output directory for results')
    parser.add_argument('--use-test-split', action='store_true',
                       help='Use same test split as training (20%% of data)')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save predictions and generate visualizations')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("MOLECULAR GNN - TEST SCRIPT")
    print("="*70)
    print(f"Using device: {device}")
    
    # Load model
    model, model_info = load_model(args.model_path, args.model_info, device)
    
    # Test on dataset
    if args.data_path:
        dataset = MolecularDataset(args.data_path)
        
        if args.use_test_split:
            # Use same split as training (last 20%)
            indices = list(range(len(dataset)))
            _, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
            print(f"Using test split with {len(test_indices)} samples")
        else:
            test_indices = None
        
        results = test_on_dataset(
            model, dataset, test_indices, 
            batch_size=args.batch_size,
            device=device,
            save_predictions=args.save_predictions,
            output_dir=args.output_dir
        )
        
        if args.save_predictions:
            print("\n" + "="*70)
            print("RESULTS SUMMARY")
            print("="*70)
            print(f"All results saved to: {args.output_dir}/")
            print("\nGenerated files:")
            print("  • test_metrics.json - Complete metrics")
            print("  • predictions.csv - All predictions")
            print("  • roc_curve.png - ROC curve visualization")
            print("  • confusion_matrix.png - Confusion matrix heatmap")
            if results['inhibitions'] is not None:
                print("  • inhibition_distribution.png - Inhibition score distribution")
            print("="*70)
    
    # Test on individual SMILES
    elif args.smiles:
        results = predict_smiles(model, args.smiles, device)
        
        print("\nPredictions:")
        print("="*70)
        for r in results:
            if r['valid']:
                print(f"SMILES: {r['smiles']}")
                print(f"  Prediction: {r['prediction']}")
                print(f"  Probability (Active): {r['probability_active']:.4f}")
                print(f"  Confidence: {r['confidence']:.4f}")
                if 'inhibition_strength' in r:
                    print(f"  Inhibition Strength: {r['inhibition_strength']:.4f}")
                print()
            else:
                print(f"SMILES: {r['smiles']}")
                print(f"  Error: {r['error']}")
                print()
        
        # Save if requested
        if args.save_predictions:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            pred_file = output_path / 'smiles_predictions.json'
            with open(pred_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Predictions saved to {pred_file}")
    
    # Test on SMILES file
    elif args.smiles_file:
        with open(args.smiles_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        
        results = predict_smiles(model, smiles_list, device)
        
        # Create DataFrame and save
        df = pd.DataFrame(results)
        print("\nPredictions Summary:")
        print(df)
        
        if args.save_predictions:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            pred_file = output_path / 'smiles_predictions.csv'
            df.to_csv(pred_file, index=False)
            print(f"\n✓ Predictions saved to {pred_file}")
    
    else:
        print("\nError: Must specify either --data-path, --smiles, or --smiles-file")
        parser.print_help()
        return


if __name__ == "__main__":
    main()
