import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm
import json

from model import DiffusionModel
from diffusion import DiffusionScheduler, DiffusionSampler

# RDKit imports for chemical validation
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Chemical validation will be skipped.")


class MoleculeGenerator:
    """
    Class for generating molecules using the trained diffusion model.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: torch.device,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 1.0):
        """
        Initialize the molecule generator.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
        """
        self.device = device
        self.guidance_scale = guidance_scale
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Create diffusion scheduler and sampler
        self.scheduler = DiffusionScheduler(
            num_timesteps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            schedule_type='linear'
        ).to(device)
        
        self.sampler = DiffusionSampler(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps
        )
        
        # Atomic number to symbol mapping
        self.atomic_numbers = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained diffusion model."""
        # Create model instance
        model = DiffusionModel(
            node_dim=1,
            hidden_dim=128,
            num_layers=6,
            max_atoms=29
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle both DDP and regular model state dicts
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Remove 'module.' prefix if it exists (from DDP)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_state_dict[key[7:]] = value
                else:
                    new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def generate_molecules(self,
                          num_molecules: int,
                          num_atoms: int,
                          atom_types: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate molecules using the diffusion model.
        
        Args:
            num_molecules: Number of molecules to generate
            num_atoms: Number of atoms per molecule
            atom_types: Optional fixed atom types [num_molecules, num_atoms]
        
        Returns:
            Tuple of (coordinates, atom_types)
        """
        with torch.no_grad():
            coords, atoms = self.sampler.sample(
                model=self.model,
                batch_size=num_molecules,
                num_atoms=num_atoms,
                device=self.device,
                atom_types=atom_types,
                guidance_scale=self.guidance_scale
            )
        
        return coords, atoms
    
    def save_molecules_to_xyz(self,
                             coords: torch.Tensor,
                             atom_types: torch.Tensor,
                             output_dir: str,
                             prefix: str = "generated") -> List[str]:
        """
        Save generated molecules to .xyz files.
        
        Args:
            coords: Generated coordinates [num_molecules, num_atoms, 3]
            atom_types: Generated atom types [num_molecules, num_atoms]
            output_dir: Directory to save .xyz files
            prefix: Prefix for filenames
        
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        num_molecules = coords.size(0)
        
        for i in range(num_molecules):
            # Convert to numpy
            mol_coords = coords[i].cpu().numpy()
            mol_atoms = atom_types[i].cpu().numpy()
            
            # Create .xyz file content
            xyz_content = f"{mol_atoms.shape[0]}\n"
            xyz_content += f"Generated molecule {i+1}\n"
            
            for j, (atom_type, coord) in enumerate(zip(mol_atoms, mol_coords)):
                atom_symbol = self.atomic_numbers.get(atom_type.item(), 'H')
                xyz_content += f"{atom_symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n"
            
            # Save to file
            filename = f"{prefix}_molecule_{i+1:04d}.xyz"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(xyz_content)
            
            saved_files.append(filepath)
        
        return saved_files
    
    def validate_molecules(self, coords: torch.Tensor, atom_types: torch.Tensor) -> List[dict]:
        """
        Validate generated molecules using RDKit.
        
        Args:
            coords: Generated coordinates [num_molecules, num_atoms, 3]
            atom_types: Generated atom types [num_molecules, num_atoms]
        
        Returns:
            List of validation results
        """
        if not RDKIT_AVAILABLE:
            return [{"valid": False, "error": "RDKit not available"} for _ in range(coords.size(0))]
        
        validation_results = []
        num_molecules = coords.size(0)
        
        for i in range(num_molecules):
            mol_coords = coords[i].cpu().numpy()
            mol_atoms = atom_types[i].cpu().numpy()
            
            try:
                # Create RDKit molecule
                mol = self._create_rdkit_molecule(mol_coords, mol_atoms)
                
                if mol is None:
                    validation_results.append({
                        "valid": False,
                        "error": "Failed to create molecule"
                    })
                    continue
                
                # Perform validation checks
                validation_result = self._validate_molecule(mol)
                validation_results.append(validation_result)
                
            except Exception as e:
                validation_results.append({
                    "valid": False,
                    "error": str(e)
                })
        
        return validation_results
    
    def _create_rdkit_molecule(self, coords: np.ndarray, atom_types: np.ndarray) -> Optional[Chem.Mol]:
        """Create an RDKit molecule from coordinates and atom types."""
        try:
            # Create molecule
            mol = Chem.RWMol()
            
            # Add atoms
            atom_indices = []
            for atom_type in atom_types:
                atom_symbol = self.atomic_numbers.get(atom_type.item(), 'H')
                atom = mol.AddAtom(Chem.Atom(atom_symbol))
                atom_indices.append(atom)
            
            # Add bonds based on distance
            for i in range(len(atom_indices)):
                for j in range(i + 1, len(atom_indices)):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    
                    # Simple bond detection based on distance
                    if dist < 1.8:  # Typical bond length threshold
                        mol.AddBond(atom_indices[i], atom_indices[j], Chem.BondType.SINGLE)
            
            # Convert to molecule
            mol = mol.GetMol()
            
            # Clean up the molecule
            Chem.SanitizeMol(mol)
            
            return mol
            
        except Exception as e:
            print(f"Error creating molecule: {e}")
            return None
    
    def _validate_molecule(self, mol: Chem.Mol) -> dict:
        """Validate a single molecule using RDKit."""
        try:
            # Basic validation
            validation_result = {
                "valid": True,
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "molecular_weight": rdMolDescriptors.CalcExactMolWt(mol),
                "smiles": Chem.MolToSmiles(mol),
                "errors": []
            }
            
            # Check for common issues
            if mol.GetNumAtoms() == 0:
                validation_result["valid"] = False
                validation_result["errors"].append("No atoms")
            
            if mol.GetNumBonds() == 0:
                validation_result["errors"].append("No bonds")
            
            # Check for disconnected fragments
            fragments = Chem.GetMolFrags(mol, asMols=True)
            if len(fragments) > 1:
                validation_result["errors"].append(f"Disconnected fragments: {len(fragments)}")
            
            # Check for unusual bond lengths (basic check)
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                begin_pos = mol.GetConformer().GetAtomPosition(begin_idx)
                end_pos = mol.GetConformer().GetAtomPosition(end_idx)
                bond_length = np.linalg.norm([begin_pos.x - end_pos.x, 
                                            begin_pos.y - end_pos.y, 
                                            begin_pos.z - end_pos.z])
                
                if bond_length > 3.0:  # Unusually long bond
                    validation_result["errors"].append(f"Long bond: {bond_length:.2f} Å")
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    def generate_and_save(self,
                         num_molecules: int,
                         num_atoms: int,
                         output_dir: str,
                         validate: bool = True) -> dict:
        """
        Generate molecules and save them to files.
        
        Args:
            num_molecules: Number of molecules to generate
            num_atoms: Number of atoms per molecule
            output_dir: Directory to save generated molecules
            validate: Whether to validate molecules using RDKit
        
        Returns:
            Dictionary with generation results
        """
        print(f"Generating {num_molecules} molecules with {num_atoms} atoms each...")
        
        # Generate molecules
        coords, atom_types = self.generate_molecules(num_molecules, num_atoms)
        
        # Save to .xyz files
        saved_files = self.save_molecules_to_xyz(coords, atom_types, output_dir)
        
        # Validate molecules if requested
        validation_results = None
        if validate:
            print("Validating generated molecules...")
            validation_results = self.validate_molecules(coords, atom_types)
        
        # Compile results
        results = {
            "num_generated": num_molecules,
            "num_atoms": num_atoms,
            "saved_files": saved_files,
            "validation_results": validation_results
        }
        
        if validation_results:
            valid_count = sum(1 for r in validation_results if r.get("valid", False))
            results["valid_molecules"] = valid_count
            results["validity_rate"] = valid_count / num_molecules
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Generate molecules using trained diffusion model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='generated_molecules', help='Output directory')
    parser.add_argument('--num_molecules', type=int, default=10, help='Number of molecules to generate')
    parser.add_argument('--num_atoms', type=int, default=10, help='Number of atoms per molecule')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of denoising steps')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Classifier-free guidance scale')
    parser.add_argument('--validate', action='store_true', help='Validate molecules using RDKit')
    parser.add_argument('--save_results', type=str, default=None, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create generator
    generator = MoleculeGenerator(
        model_path=args.model_path,
        device=device,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale
    )
    
    # Generate molecules
    results = generator.generate_and_save(
        num_molecules=args.num_molecules,
        num_atoms=args.num_atoms,
        output_dir=args.output_dir,
        validate=args.validate
    )
    
    # Print results
    print(f"\nGeneration completed!")
    print(f"Generated {results['num_generated']} molecules")
    print(f"Saved to: {args.output_dir}")
    
    if results.get('validation_results'):
        print(f"Valid molecules: {results['valid_molecules']}/{results['num_generated']}")
        print(f"Validity rate: {results['validity_rate']:.2%}")
        
        # Print some validation details
        for i, result in enumerate(results['validation_results'][:5]):  # Show first 5
            if result.get('valid'):
                print(f"  Molecule {i+1}: Valid - {result.get('smiles', 'N/A')}")
            else:
                print(f"  Molecule {i+1}: Invalid - {result.get('error', 'Unknown error')}")
    
    # Save results to JSON if requested
    if args.save_results:
        # Convert tensors to lists for JSON serialization
        json_results = {
            "num_generated": results["num_generated"],
            "num_atoms": results["num_atoms"],
            "saved_files": results["saved_files"],
            "valid_molecules": results.get("valid_molecules", 0),
            "validity_rate": results.get("validity_rate", 0.0)
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {args.save_results}")


if __name__ == "__main__":
    main()