#!/usr/bin/env python3
"""
Example workflow demonstrating the complete 3D Molecular Diffusion Model pipeline.
This script shows how to:
1. Load and process QM9 data
2. Train a diffusion model
3. Generate new molecules
4. Validate the generated molecules
"""

import torch
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path

def create_sample_dataset():
    """Create a small sample dataset for demonstration."""
    print("Creating sample dataset...")
    
    # Create sample molecules
    molecules = [
        # Methane
        """5
Methane
C 0.000000 0.000000 0.000000
H 1.089000 0.000000 0.000000
H -0.363000 1.033000 0.000000
H -0.363000 -0.516500 0.895000
H -0.363000 -0.516500 -0.895000""",
        
        # Water
        """3
Water
O 0.000000 0.000000 0.000000
H 0.957000 0.000000 0.000000
H -0.240000 0.927000 0.000000""",
        
        # Ammonia
        """4
Ammonia
N 0.000000 0.000000 0.000000
H 1.012000 0.000000 0.000000
H -0.337000 0.947000 0.000000
H -0.337000 -0.473500 0.820000""",
        
        # Carbon dioxide
        """3
Carbon Dioxide
C 0.000000 0.000000 0.000000
O 1.160000 0.000000 0.000000
O -1.160000 0.000000 0.000000""",
        
        # Methanol
        """6
Methanol
C 0.000000 0.000000 0.000000
O 1.420000 0.000000 0.000000
H -0.354000 1.033000 0.000000
H -0.354000 -0.516500 0.895000
H -0.354000 -0.516500 -0.895000
H 1.774000 0.000000 0.000000"""
    ]
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, "sample_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save molecules
    for i, mol in enumerate(molecules):
        filename = os.path.join(data_dir, f"molecule_{i+1:03d}.xyz")
        with open(filename, 'w') as f:
            f.write(mol)
    
    print(f"Sample dataset created with {len(molecules)} molecules")
    return data_dir, temp_dir

def demonstrate_data_loading(data_dir):
    """Demonstrate data loading functionality."""
    print("\n" + "="*50)
    print("DEMONSTRATION: Data Loading")
    print("="*50)
    
    from dataset import QM9Dataset, create_dataloader
    
    # Create dataset
    dataset = QM9Dataset(data_dir=data_dir, max_atoms=10)
    print(f"Dataset size: {len(dataset)}")
    
    # Load a sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Atomic numbers shape: {sample.x.shape}")
    print(f"Coordinates shape: {sample.pos.shape}")
    print(f"Number of atoms: {sample.num_atoms}")
    print(f"Mask shape: {sample.mask.shape}")
    
    # Create dataloader
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=True)
    
    # Iterate through a batch
    for batch in dataloader:
        print(f"\nBatch info:")
        print(f"  Number of graphs: {batch.num_graphs}")
        print(f"  Total nodes: {batch.x.size(0)}")
        print(f"  Total edges: {batch.edge_index.size(1)}")
        break
    
    return dataset

def demonstrate_model_architecture():
    """Demonstrate model architecture."""
    print("\n" + "="*50)
    print("DEMONSTRATION: Model Architecture")
    print("="*50)
    
    from model import DiffusionModel
    from torch_geometric.data import Data, Batch
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionModel(
        node_dim=1,
        hidden_dim=64,  # Smaller for demonstration
        num_layers=3,
        max_atoms=10
    ).to(device)
    
    print(f"Model created on device: {device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
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
    
    print(f"Forward pass successful:")
    print(f"  Noise prediction shape: {noise_pred.shape}")
    print(f"  Atom logits shape: {atom_logits.shape}")
    
    return model

def demonstrate_diffusion_process():
    """Demonstrate diffusion process."""
    print("\n" + "="*50)
    print("DEMONSTRATION: Diffusion Process")
    print("="*50)
    
    from diffusion import DiffusionScheduler, DiffusionLoss
    
    # Create scheduler
    scheduler = DiffusionScheduler(
        num_timesteps=100,  # Smaller for demonstration
        schedule_type='linear'
    )
    
    print(f"Diffusion scheduler created with {scheduler.num_timesteps} timesteps")
    
    # Test noise addition
    batch_size = 2
    num_atoms = 5
    
    original_coords = torch.randn(batch_size, num_atoms, 3)
    timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,))
    
    print(f"Original coordinates shape: {original_coords.shape}")
    print(f"Timesteps: {timesteps}")
    
    # Add noise
    noised_coords, noise = scheduler.add_noise(original_coords, timesteps)
    print(f"Noised coordinates shape: {noised_coords.shape}")
    print(f"Noise shape: {noise.shape}")
    
    # Test noise removal
    denoised_coords = scheduler.remove_noise(noised_coords, noise, timesteps)
    print(f"Denoised coordinates shape: {denoised_coords.shape}")
    
    # Test loss function
    loss_fn = DiffusionLoss(loss_type='mse', atom_loss_weight=0.1)
    
    predicted_noise = torch.randn_like(noise)
    predicted_atoms = torch.randn(batch_size, num_atoms, 5)
    target_atoms = torch.randint(0, 5, (batch_size, num_atoms))
    mask = torch.ones(batch_size, num_atoms, dtype=torch.bool)
    
    total_loss, coord_loss, atom_loss = loss_fn(
        predicted_noise, noise, predicted_atoms, target_atoms, mask
    )
    
    print(f"Loss computation successful:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Coordinate loss: {coord_loss.item():.4f}")
    print(f"  Atom loss: {atom_loss.item():.4f}")
    
    return scheduler, loss_fn

def demonstrate_training_step(model, scheduler, loss_fn, dataset):
    """Demonstrate a single training step."""
    print("\n" + "="*50)
    print("DEMONSTRATION: Training Step")
    print("="*50)
    
    from dataset import create_dataloader
    import torch.optim as optim
    
    # Create dataloader
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=True)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training step
    model.train()
    for batch in dataloader:
        device = next(model.parameters()).device
        batch = batch.to(device)
        
        # Get batch info
        batch_size = batch.num_graphs
        num_atoms = batch.pos.size(0) // batch_size
        
        # Sample timesteps
        timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
        
        # Get original data
        original_coords = batch.pos.view(batch_size, num_atoms, 3)
        atom_types = batch.x.view(batch_size, num_atoms).squeeze(-1)
        mask = batch.mask.view(batch_size, num_atoms)
        
        # Add noise
        noised_coords, noise = scheduler.add_noise(original_coords, timesteps)
        batch.pos = noised_coords.view(-1, 3)
        
        # Forward pass
        optimizer.zero_grad()
        predicted_noise, predicted_atoms = model(batch, timesteps)
        
        # Reshape predictions
        predicted_noise = predicted_noise.view(batch_size, num_atoms, 3)
        predicted_atoms = predicted_atoms.view(batch_size, num_atoms, -1)
        
        # Compute loss
        loss, coord_loss, atom_loss = loss_fn(
            predicted_noise, noise, predicted_atoms, atom_types, mask
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Training step completed:")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Coordinate loss: {coord_loss.item():.4f}")
        print(f"  Atom loss: {atom_loss.item():.4f}")
        break
    
    return model

def demonstrate_inference(model, scheduler):
    """Demonstrate molecule generation."""
    print("\n" + "="*50)
    print("DEMONSTRATION: Molecule Generation")
    print("="*50)
    
    from diffusion import DiffusionSampler
    
    # Create sampler
    sampler = DiffusionSampler(scheduler, num_inference_steps=20)  # Fewer steps for demo
    
    device = next(model.parameters()).device
    
    # Generate molecules
    print("Generating molecules...")
    with torch.no_grad():
        coords, atom_types = sampler.sample(
            model=model,
            batch_size=2,
            num_atoms=5,
            device=device
        )
    
    print(f"Generated molecules:")
    print(f"  Coordinates shape: {coords.shape}")
    print(f"  Atom types shape: {atom_types.shape}")
    
    # Show some details
    for i in range(coords.size(0)):
        print(f"  Molecule {i+1}:")
        print(f"    Atom types: {atom_types[i].cpu().numpy()}")
        print(f"    Center of mass: {coords[i].mean(dim=0).cpu().numpy()}")
    
    return coords, atom_types

def demonstrate_validation(coords, atom_types):
    """Demonstrate molecule validation."""
    print("\n" + "="*50)
    print("DEMONSTRATION: Molecule Validation")
    print("="*50)
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        print("RDKit validation available")
        
        # Convert to numpy
        coords_np = coords.cpu().numpy()
        atom_types_np = atom_types.cpu().numpy()
        
        # Atomic number mapping
        atomic_numbers = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
        
        for i in range(coords_np.shape[0]):
            print(f"\nValidating molecule {i+1}:")
            
            # Create RDKit molecule
            mol = Chem.RWMol()
            
            # Add atoms
            atom_indices = []
            for atom_type in atom_types_np[i]:
                atom_symbol = atomic_numbers.get(atom_type.item(), 'H')
                atom = mol.AddAtom(Chem.Atom(atom_symbol))
                atom_indices.append(atom)
            
            # Add bonds based on distance
            for j in range(len(atom_indices)):
                for k in range(j + 1, len(atom_indices)):
                    dist = np.linalg.norm(coords_np[i, j] - coords_np[i, k])
                    if dist < 1.8:  # Bond threshold
                        mol.AddBond(atom_indices[j], atom_indices[k], Chem.BondType.SINGLE)
            
            mol = mol.GetMol()
            
            try:
                Chem.SanitizeMol(mol)
                smiles = Chem.MolToSmiles(mol)
                print(f"  Valid molecule: {smiles}")
                print(f"  Number of atoms: {mol.GetNumAtoms()}")
                print(f"  Number of bonds: {mol.GetNumBonds()}")
            except:
                print(f"  Invalid molecule")
                
    except ImportError:
        print("RDKit not available - skipping validation")

def main():
    """Run the complete demonstration."""
    print("="*60)
    print("3D Molecular Diffusion Model - Complete Workflow Demo")
    print("="*60)
    
    # Create sample dataset
    data_dir, temp_dir = create_sample_dataset()
    
    try:
        # Demonstrate each component
        dataset = demonstrate_data_loading(data_dir)
        model = demonstrate_model_architecture()
        scheduler, loss_fn = demonstrate_diffusion_process()
        model = demonstrate_training_step(model, scheduler, loss_fn, dataset)
        coords, atom_types = demonstrate_inference(model, scheduler)
        demonstrate_validation(coords, atom_types)
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThis demonstration shows:")
        print("✓ Data loading and preprocessing")
        print("✓ Model architecture and forward pass")
        print("✓ Diffusion process (noising/denoising)")
        print("✓ Training step with loss computation")
        print("✓ Molecule generation")
        print("✓ Chemical validation")
        
        print("\nNext steps:")
        print("1. Prepare your QM9 dataset in the 'data/' directory")
        print("2. Run full training: ./launch_training.sh")
        print("3. Generate molecules: python inference.py --model_path checkpoints/best_model.pth")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        print("Please check the error messages above and ensure all dependencies are installed.")
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary files")

if __name__ == "__main__":
    main()