#!/usr/bin/env python3
"""
Setup Verification Script
Checks if all dependencies and files are ready for testing
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    missing = []
    
    packages = [
        ('torch', 'PyTorch'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('rdkit', 'RDKit'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
    ]
    
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT INSTALLED")
            missing.append(package_name)
    
    return missing

def check_files():
    """Check if required files exist"""
    print("\nChecking files...")
    
    required_files = [
        ('InhModel.py', 'Model definition and training'),
        ('test_model.py', 'Testing script'),
        ('quick_test.py', 'Interactive testing'),
        ('requirements.txt', 'Dependencies'),
        ('AID_588726_datatable (FBA).csv', 'Dataset'),
    ]
    
    optional_files = [
        ('model_exp2_weighted_2x.pt', 'Trained model'),
        ('model_info.json', 'Model metadata'),
    ]
    
    missing_required = []
    
    for filename, description in required_files:
        if Path(filename).exists():
            print(f"  ✓ {filename} - {description}")
        else:
            print(f"  ✗ {filename} - NOT FOUND")
            missing_required.append(filename)
    
    print("\nOptional files:")
    for filename, description in optional_files:
        if Path(filename).exists():
            print(f"  ✓ {filename} - {description}")
        else:
            print(f"  ⚠ {filename} - Not found (will be created after training)")
    
    return missing_required

def check_cuda():
    """Check CUDA availability"""
    print("\nChecking GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
        else:
            print(f"  ⚠ CUDA not available - will use CPU")
            print(f"    (Training and testing will be slower)")
    except ImportError:
        print(f"  ✗ Cannot check - PyTorch not installed")

def main():
    print("="*70)
    print("Molecular GNN - Setup Verification")
    print("="*70)
    print()
    
    # Check dependencies
    missing_deps = check_dependencies()
    
    # Check files
    missing_files = check_files()
    
    # Check CUDA
    check_cuda()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if missing_deps:
        print("\n❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall missing dependencies:")
        print("  pip install -r requirements.txt")
        return False
    
    if missing_files:
        print("\n❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    print("\n✓ All dependencies installed")
    print("✓ All required files present")
    
    # Check if model is trained
    if not Path("model_exp2_weighted_2x.pt").exists():
        print("\n⚠ Model not trained yet")
        print("\nNext steps:")
        print("  1. Train the model:")
        print("     python model.py")
        print("\n  2. Test the model:")
        print("     python quick_test.py")
    else:
        print("\n✓ Model is trained and ready!")
        print("\nYou can now:")
        print("  • Run interactive testing: python quick_test.py")
        print("  • Run CLI testing: python test_model.py --help")
        print("  • Use Python API: see README.md")
    
    print("\n" + "="*70)
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

