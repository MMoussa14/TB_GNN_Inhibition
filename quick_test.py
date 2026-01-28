#!/usr/bin/env python3
"""
Quick Test Script for Molecular GNN
Simple interface for testing your trained model
"""

import sys
import torch
from pathlib import Path
from test_model import load_model, predict_smiles, test_on_dataset
from model import MolecularDataset
from sklearn.model_selection import train_test_split


def main():
    print("="*70)
    print("Molecular GNN - Quick Test")
    print("="*70)
    
    # Check if model exists
    model_path = Path("model_exp2_weighted_2x.pt")
    if not model_path.exists():
        print("\n❌ Error: Model not found!")
        print("Please train the model first by running:")
        print("  python model.py")
        return
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Loading model...")
    print(f"  Device: {device}")
    
    try:
        model, model_info = load_model(str(model_path), device=device)
        print(f"  Architecture: {model_info.get('gnn_type', 'GCN')}")
        print(f"  Parameters: {model_info.get('total_parameters', 'Unknown')}")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return
    
    # Main menu
    while True:
        print("\n" + "="*70)
        print("What would you like to test?")
        print("="*70)
        print("1. Test on tuberculosis dataset (test split)")
        print("2. Predict activity for specific molecules (SMILES)")
        print("3. Example predictions (demo molecules)")
        print("4. Exit")
        print()
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            test_dataset(model, device)
        elif choice == "2":
            test_custom_smiles(model, device)
        elif choice == "3":
            test_examples(model, device)
        elif choice == "4":
            print("\nExiting. Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Please enter 1-4.")


def test_dataset(model, device):
    """Test on the tuberculosis dataset"""
    data_path = Path("AID_588726_datatable (FBA).csv")
    
    if not data_path.exists():
        print("\n❌ Error: Dataset not found!")
        print(f"Looking for: {data_path}")
        return
    
    print("\n" + "="*70)
    print("Testing on Tuberculosis Dataset")
    print("="*70)
    print("\nLoading dataset... (this may take a minute)")
    
    try:
        dataset = MolecularDataset(str(data_path))
        
        # Use same test split as training
        indices = list(range(len(dataset)))
        _, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
        
        print(f"\nDataset loaded:")
        print(f"  Total molecules: {len(dataset)}")
        print(f"  Test split: {len(test_indices)} molecules")
        
        # Run test
        results = test_on_dataset(
            model, 
            dataset, 
            test_indices,
            batch_size=32,
            device=device,
            save_predictions=True,
            output_dir='./test_results'
        )
        
        print("\n✓ Testing complete! Results saved to ./test_results/")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


def test_custom_smiles(model, device):
    """Test on user-provided SMILES"""
    print("\n" + "="*70)
    print("Predict Molecular Activity")
    print("="*70)
    print("\nEnter SMILES strings (one per line, empty line to finish):")
    print("Example: CCO (ethanol)")
    print()
    
    smiles_list = []
    while True:
        smiles = input("SMILES > ").strip()
        if not smiles:
            break
        smiles_list.append(smiles)
    
    if not smiles_list:
        print("\n❌ No SMILES entered.")
        return
    
    print(f"\n🔬 Predicting activity for {len(smiles_list)} molecules...\n")
    
    try:
        results = predict_smiles(model, smiles_list, device=device)
        
        print("="*70)
        print("RESULTS")
        print("="*70)
        
        for r in results:
            if r['valid']:
                activity_emoji = "🔴" if r['prediction'] == 'Active' else "🟢"
                print(f"\n{activity_emoji} SMILES: {r['smiles']}")
                print(f"   Prediction: {r['prediction']}")
                print(f"   Probability (Active): {r['probability_active']:.4f}")
                print(f"   Confidence: {r['confidence']:.2%}")
            else:
                print(f"\n❌ SMILES: {r['smiles']}")
                print(f"   Error: {r['error']}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")


def test_examples(model, device):
    """Test on example molecules"""
    print("\n" + "="*70)
    print("Example Predictions")
    print("="*70)
    
    examples = [
        ("CCO", "Ethanol"),
        ("c1ccccc1", "Benzene"),
        ("CC(=O)O", "Acetic acid"),
        ("CC(C)Cc1ccc(cc1)C(C)C(O)=O", "Ibuprofen"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
        ("CC(C)NCC(COc1ccccc1)O", "Propranolol (beta blocker)"),
    ]
    
    smiles_list = [s for s, _ in examples]
    names = [n for _, n in examples]
    
    print(f"\n🔬 Testing {len(examples)} example molecules...\n")
    
    try:
        results = predict_smiles(model, smiles_list, device=device)
        
        print("="*70)
        print("RESULTS")
        print("="*70)
        
        for r, name in zip(results, names):
            if r['valid']:
                activity_emoji = "🔴" if r['prediction'] == 'Active' else "🟢"
                print(f"\n{activity_emoji} {name}")
                print(f"   SMILES: {r['smiles']}")
                print(f"   Prediction: {r['prediction']}")
                print(f"   Probability (Active): {r['probability_active']:.4f}")
                print(f"   Confidence: {r['confidence']:.2%}")
            else:
                print(f"\n❌ {name}")
                print(f"   Error: {r['error']}")
        
        print("\n" + "="*70)
        print("\nNote: 'Active' means potentially toxic to tuberculosis bacteria")
        print("      'Inactive' means not toxic to tuberculosis bacteria")
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)

