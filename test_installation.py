#!/usr/bin/env python3
"""
Test script to verify the installation and basic functionality of the 3D Molecular Diffusion Model.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric {torch_geometric.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch Geometric import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from rdkit import Chem
        print("✓ RDKit")
    except ImportError as e:
        print(f"⚠ RDKit import failed: {e} (optional)")
    
    try:
        import wandb
        print("✓ WandB")
    except ImportError as e:
        print(f"⚠ WandB import failed: {e} (optional)")
    
    try:
        from omegaconf import OmegaConf
        print("✓ OmegaConf")
    except ImportError as e:
        print(f"✗ OmegaConf import failed: {e}")
        return False
    
    return True

def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        return True
    else:
        print("⚠ CUDA not available - will use CPU")
        return False

def test_model_components():
    """Test model components."""
    print("\nTesting model components...")
    
    try:
        from model import DiffusionModel
        from diffusion import DiffusionScheduler, DiffusionLoss
        from dataset import QM9Dataset
        
        print("✓ Model components imported successfully")
        
        # Test model creation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DiffusionModel(
            node_dim=1,
            hidden_dim=64,  # Smaller for testing
            num_layers=3,
            max_atoms=10
        ).to(device)
        
        print("✓ Model created successfully")
        
        # Test diffusion scheduler
        scheduler = DiffusionScheduler(
            num_timesteps=100,
            schedule_type='linear'
        )
        print("✓ Diffusion scheduler created successfully")
        
        # Test loss function
        loss_fn = DiffusionLoss(loss_type='mse', atom_loss_weight=0.1)
        print("✓ Loss function created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model component test failed: {e}")
        return False

def test_model_forward_pass():
    """Test model forward pass."""
    print("\nTesting model forward pass...")
    
    try:
        from model import DiffusionModel
        from torch_geometric.data import Data, Batch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a simple model
        model = DiffusionModel(
            node_dim=1,
            hidden_dim=32,
            num_layers=2,
            max_atoms=5
        ).to(device)
        
        # Create dummy data
        batch_size = 2
        num_nodes = 5
        
        x = torch.randint(1, 6, (num_nodes, 1), dtype=torch.long).to(device)
        pos = torch.randn(num_nodes, 3).to(device)
        edge_index = torch.randint(0, num_nodes, (2, 20)).to(device)
        batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes // batch_size).to(device)
        t = torch.rand(batch_size).to(device)
        
        data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            noise_pred, atom_logits = model(data, t)
        
        print(f"✓ Forward pass successful")
        print(f"  Noise prediction shape: {noise_pred.shape}")
        print(f"  Atom logits shape: {atom_logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        return False

def test_diffusion_process():
    """Test diffusion process."""
    print("\nTesting diffusion process...")
    
    try:
        from diffusion import DiffusionScheduler
        
        scheduler = DiffusionScheduler(
            num_timesteps=100,
            schedule_type='linear'
        )
        
        # Test noise addition
        batch_size = 2
        num_atoms = 5
        
        original_coords = torch.randn(batch_size, num_atoms, 3)
        timesteps = torch.randint(0, 100, (batch_size,))
        
        noised_coords, noise = scheduler.add_noise(original_coords, timesteps)
        
        print(f"✓ Noise addition successful")
        print(f"  Original coords shape: {original_coords.shape}")
        print(f"  Noised coords shape: {noised_coords.shape}")
        print(f"  Noise shape: {noise.shape}")
        
        # Test noise removal
        denoised_coords = scheduler.remove_noise(noised_coords, noise, timesteps)
        print(f"✓ Noise removal successful")
        print(f"  Denoised coords shape: {denoised_coords.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Diffusion process test failed: {e}")
        return False

def test_dataset():
    """Test dataset creation (without actual data)."""
    print("\nTesting dataset...")
    
    try:
        from dataset import QM9Dataset
        
        # Create dataset with dummy directory
        dataset = QM9Dataset(data_dir="dummy_dir", max_atoms=10)
        print("✓ Dataset class created successfully")
        
        # Test that it handles empty directory gracefully
        print(f"  Dataset length: {len(dataset)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False

def create_sample_data():
    """Create sample XYZ files for testing."""
    print("\nCreating sample data...")
    
    try:
        os.makedirs("test_data", exist_ok=True)
        
        # Create a simple methane molecule
        methane_xyz = """5
Methane
C 0.000000 0.000000 0.000000
H 1.089000 0.000000 0.000000
H -0.363000 1.033000 0.000000
H -0.363000 -0.516500 0.895000
H -0.363000 -0.516500 -0.895000"""
        
        with open("test_data/methane.xyz", "w") as f:
            f.write(methane_xyz)
        
        # Create a water molecule
        water_xyz = """3
Water
O 0.000000 0.000000 0.000000
H 0.957000 0.000000 0.000000
H -0.240000 0.927000 0.000000"""
        
        with open("test_data/water.xyz", "w") as f:
            f.write(water_xyz)
        
        print("✓ Sample data created in test_data/")
        return True
        
    except Exception as e:
        print(f"✗ Sample data creation failed: {e}")
        return False

def test_with_sample_data():
    """Test with actual sample data."""
    print("\nTesting with sample data...")
    
    try:
        from dataset import QM9Dataset
        
        dataset = QM9Dataset(data_dir="test_data", max_atoms=10)
        print(f"✓ Dataset loaded with {len(dataset)} molecules")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample keys: {list(sample.keys())}")
            print(f"  Sample x shape: {sample.x.shape}")
            print(f"  Sample pos shape: {sample.pos.shape}")
            print(f"  Sample num_atoms: {sample.num_atoms}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample data test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("3D Molecular Diffusion Model - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Model Components", test_model_components),
        ("Model Forward Pass", test_model_forward_pass),
        ("Diffusion Process", test_diffusion_process),
        ("Dataset", test_dataset),
        ("Sample Data Creation", create_sample_data),
        ("Sample Data Test", test_with_sample_data),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\nNext steps:")
        print("1. Place your QM9 dataset in the 'data/' directory")
        print("2. Run training: ./launch_training.sh")
        print("3. Generate molecules: python inference.py --model_path checkpoints/best_model.pth")
    else:
        print("⚠ Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check CUDA installation if using GPU")
        print("3. Verify PyTorch and PyTorch Geometric versions are compatible")
    
    # Cleanup
    if os.path.exists("test_data"):
        import shutil
        shutil.rmtree("test_data")

if __name__ == "__main__":
    main()