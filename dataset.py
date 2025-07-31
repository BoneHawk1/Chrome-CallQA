import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from ase.io import read
import pandas as pd
from typing import List, Tuple, Optional
import glob


class QM9Dataset(Dataset):
    """
    Custom PyTorch Dataset for QM9 molecular data.
    Loads .xyz files and extracts atomic numbers, coordinates, and properties.
    """
    
    def __init__(self, data_dir: str, max_atoms: int = 29, transform=None):
        """
        Initialize the QM9 dataset.
        
        Args:
            data_dir: Directory containing .xyz files
            max_atoms: Maximum number of atoms per molecule (QM9 max is 29)
            transform: Optional transform to apply to the data
        """
        self.data_dir = data_dir
        self.max_atoms = max_atoms
        self.transform = transform
        
        # Find all .xyz files in the directory
        self.xyz_files = glob.glob(os.path.join(data_dir, "*.xyz"))
        self.xyz_files.sort()  # Ensure consistent ordering
        
        # Atomic number mapping for common elements in QM9
        self.atomic_numbers = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9
        }
        
        print(f"Found {len(self.xyz_files)} molecular files")
    
    def __len__(self):
        return len(self.xyz_files)
    
    def __getitem__(self, idx):
        """
        Load a single molecule from its .xyz file.
        
        Returns:
            Data object with:
            - x: Atomic numbers (node features)
            - pos: 3D coordinates
            - num_atoms: Number of atoms in the molecule
        """
        xyz_file = self.xyz_files[idx]
        
        try:
            # Read the .xyz file using ASE
            atoms = read(xyz_file)
            
            # Extract atomic numbers and positions
            atomic_numbers = []
            positions = []
            
            for atom in atoms:
                atomic_numbers.append(self.atomic_numbers.get(atom.symbol, 1))  # Default to H if unknown
                positions.append(atom.position)
            
            # Convert to tensors
            atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
            positions = torch.tensor(positions, dtype=torch.float32)
            
            # Pad to max_atoms if necessary
            num_atoms = len(atomic_numbers)
            if num_atoms < self.max_atoms:
                # Pad with zeros
                padding_size = self.max_atoms - num_atoms
                atomic_numbers = torch.cat([atomic_numbers, torch.zeros(padding_size, dtype=torch.long)])
                positions = torch.cat([positions, torch.zeros(padding_size, 3, dtype=torch.float32)])
            
            # Create mask for valid atoms
            mask = torch.arange(self.max_atoms) < num_atoms
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=atomic_numbers.unsqueeze(-1),  # Add feature dimension
                pos=positions,
                num_atoms=num_atoms,
                mask=mask,
                file_path=xyz_file
            )
            
            if self.transform:
                data = self.transform(data)
            
            return data
            
        except Exception as e:
            print(f"Error loading {xyz_file}: {e}")
            # Return a dummy molecule if loading fails
            return self._create_dummy_molecule()
    
    def _create_dummy_molecule(self):
        """Create a dummy molecule for error handling."""
        atomic_numbers = torch.zeros(self.max_atoms, dtype=torch.long)
        positions = torch.zeros(self.max_atoms, 3, dtype=torch.float32)
        mask = torch.zeros(self.max_atoms, dtype=torch.bool)
        
        return Data(
            x=atomic_numbers.unsqueeze(-1),
            pos=positions,
            num_atoms=0,
            mask=mask,
            file_path="dummy"
        )


def create_dataloader(
    dataset: QM9Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the QM9 dataset.
    
    Args:
        dataset: QM9Dataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance
    """
    
    def collate_fn(batch):
        """Custom collate function to handle variable-sized molecules."""
        return Batch.from_data_list(batch)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def create_distributed_dataloader(
    dataset: QM9Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    world_size: int = 1,
    rank: int = 0
) -> DataLoader:
    """
    Create a distributed DataLoader for multi-GPU training.
    
    Args:
        dataset: QM9Dataset instance
        batch_size: Batch size per GPU
        num_workers: Number of worker processes per GPU
        pin_memory: Whether to pin memory for faster GPU transfer
        world_size: Total number of GPUs
        rank: Current GPU rank
    
    Returns:
        DataLoader instance with DistributedSampler
    """
    from torch.utils.data.distributed import DistributedSampler
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    def collate_fn(batch):
        """Custom collate function to handle variable-sized molecules."""
        return Batch.from_data_list(batch)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    # Test the dataset
    dataset = QM9Dataset("data/", max_atoms=29)
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys}")
    print(f"Atomic numbers shape: {sample.x.shape}")
    print(f"Positions shape: {sample.pos.shape}")
    print(f"Number of atoms: {sample.num_atoms}")
    print(f"Mask shape: {sample.mask.shape}")