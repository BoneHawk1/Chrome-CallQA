import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class DiffusionScheduler:
    """
    Manages the diffusion process schedule and noise levels.
    """
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 schedule_type: str = 'linear'):
        """
        Initialize the diffusion scheduler.
        
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
            schedule_type: Type of noise schedule ('linear' or 'cosine')
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        
        # Compute noise schedule
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Pre-compute values for efficiency
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-compute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Pre-compute values for reverse process
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Pre-compute posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int) -> torch.Tensor:
        """
        Cosine beta schedule as proposed in the DDPM paper.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, 
                  original_coords: torch.Tensor, 
                  timesteps: torch.Tensor,
                  noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to coordinates according to the diffusion schedule.
        
        Args:
            original_coords: Original 3D coordinates [batch_size, num_atoms, 3]
            timesteps: Diffusion timesteps [batch_size]
            noise: Optional pre-computed noise
        
        Returns:
            Tuple of (noised_coords, noise)
        """
        if noise is None:
            noise = torch.randn_like(original_coords)
        
        # Get noise schedule values for the given timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        
        # Add noise according to the diffusion equation
        noised_coords = sqrt_alphas_cumprod_t * original_coords + sqrt_one_minus_alphas_cumprod_t * noise
        
        return noised_coords, noise
    
    def remove_noise(self,
                    noised_coords: torch.Tensor,
                    predicted_noise: torch.Tensor,
                    timesteps: torch.Tensor) -> torch.Tensor:
        """
        Remove predicted noise from noised coordinates.
        
        Args:
            noised_coords: Noised coordinates [batch_size, num_atoms, 3]
            predicted_noise: Predicted noise [batch_size, num_atoms, 3]
            timesteps: Current timesteps [batch_size]
        
        Returns:
            Denoised coordinates
        """
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[timesteps].view(-1, 1, 1)
        
        # Remove noise according to the reverse diffusion equation
        denoised_coords = sqrt_recip_alphas_cumprod_t * noised_coords - sqrt_recipm1_alphas_cumprod_t * predicted_noise
        
        return denoised_coords


class DiffusionLoss(nn.Module):
    """
    Loss function for diffusion model training.
    """
    
    def __init__(self, loss_type: str = 'mse', atom_loss_weight: float = 0.1):
        """
        Initialize the diffusion loss.
        
        Args:
            loss_type: Type of loss ('mse' or 'huber')
            atom_loss_weight: Weight for atom type prediction loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.atom_loss_weight = atom_loss_weight
        
        if loss_type == 'mse':
            self.coord_loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'huber':
            self.coord_loss_fn = nn.HuberLoss(reduction='none', delta=1.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.atom_loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self,
                predicted_noise: torch.Tensor,
                target_noise: torch.Tensor,
                predicted_atoms: torch.Tensor,
                target_atoms: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the diffusion loss.
        
        Args:
            predicted_noise: Model's predicted noise [batch_size, num_atoms, 3]
            target_noise: Target noise [batch_size, num_atoms, 3]
            predicted_atoms: Predicted atom type logits [batch_size, num_atoms, num_types]
            target_atoms: Target atom types [batch_size, num_atoms]
            mask: Mask for valid atoms [batch_size, num_atoms]
        
        Returns:
            Tuple of (total_loss, coord_loss, atom_loss)
        """
        # Coordinate loss (noise prediction)
        coord_loss = self.coord_loss_fn(predicted_noise, target_noise)  # [batch_size, num_atoms, 3]
        coord_loss = coord_loss.mean(dim=-1)  # [batch_size, num_atoms]
        
        # Apply mask to coordinate loss
        coord_loss = coord_loss * mask.float()
        coord_loss = coord_loss.sum() / (mask.float().sum() + 1e-8)
        
        # Atom type loss
        atom_loss = self.atom_loss_fn(
            predicted_atoms.view(-1, predicted_atoms.size(-1)),
            target_atoms.view(-1)
        )  # [batch_size * num_atoms]
        atom_loss = atom_loss.view(predicted_atoms.size(0), -1)  # [batch_size, num_atoms]
        
        # Apply mask to atom loss
        atom_loss = atom_loss * mask.float()
        atom_loss = atom_loss.sum() / (mask.float().sum() + 1e-8)
        
        # Total loss
        total_loss = coord_loss + self.atom_loss_weight * atom_loss
        
        return total_loss, coord_loss, atom_loss


class DiffusionSampler:
    """
    Handles the reverse diffusion process for generating molecules.
    """
    
    def __init__(self, scheduler: DiffusionScheduler, num_inference_steps: int = 50):
        """
        Initialize the diffusion sampler.
        
        Args:
            scheduler: Diffusion scheduler
            num_inference_steps: Number of denoising steps for inference
        """
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        
        # Create timestep schedule for inference
        self.inference_timesteps = torch.linspace(
            0, scheduler.num_timesteps - 1, num_inference_steps, dtype=torch.long
        )
    
    def sample(self,
               model: nn.Module,
               batch_size: int,
               num_atoms: int,
               device: torch.device,
               atom_types: Optional[torch.Tensor] = None,
               guidance_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate molecules using the trained diffusion model.
        
        Args:
            model: Trained diffusion model
            batch_size: Number of molecules to generate
            num_atoms: Number of atoms per molecule
            device: Device to run inference on
            atom_types: Optional fixed atom types [batch_size, num_atoms]
            guidance_scale: Scale for classifier-free guidance
        
        Returns:
            Tuple of (generated_coords, generated_atoms)
        """
        model.eval()
        
        # Start with random noise
        coords = torch.randn(batch_size, num_atoms, 3, device=device)
        
        # Initialize atom types if not provided
        if atom_types is None:
            atom_types = torch.randint(1, 6, (batch_size, num_atoms), device=device)
        
        # Generate edge indices for the graph
        edge_index = self._create_fully_connected_edges(num_atoms, device)
        
        # Denoising loop
        for i, t in enumerate(self.inference_timesteps.flip(0)):  # Reverse order
            t = t.repeat(batch_size)
            
            # Create batch data
            batch_data = self._create_batch_data(coords, atom_types, edge_index, batch_size, num_atoms)
            
            with torch.no_grad():
                # Predict noise
                predicted_noise, predicted_atoms = model(batch_data, t)
                
                # Apply classifier-free guidance if needed
                if guidance_scale > 1.0:
                    # Generate unconditional prediction
                    unconditional_data = self._create_batch_data(
                        coords, 
                        torch.zeros_like(atom_types), 
                        edge_index, 
                        batch_size, 
                        num_atoms
                    )
                    uncond_noise, _ = model(unconditional_data, t)
                    
                    # Interpolate between conditional and unconditional
                    predicted_noise = uncond_noise + guidance_scale * (predicted_noise - uncond_noise)
                
                # Remove noise
                coords = self.scheduler.remove_noise(coords, predicted_noise, t)
                
                # Update atom types (optional)
                if i % 10 == 0:  # Update atom types every 10 steps
                    atom_types = torch.argmax(predicted_atoms, dim=-1)
        
        return coords, atom_types
    
    def _create_fully_connected_edges(self, num_atoms: int, device: torch.device) -> torch.Tensor:
        """Create fully connected edge indices for a molecule."""
        edges = []
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long, device=device).t()
    
    def _create_batch_data(self, 
                          coords: torch.Tensor, 
                          atom_types: torch.Tensor, 
                          edge_index: torch.Tensor,
                          batch_size: int,
                          num_atoms: int):
        """Create batch data for the model."""
        from torch_geometric.data import Data, Batch
        
        batch_list = []
        for i in range(batch_size):
            data = Data(
                x=atom_types[i].unsqueeze(-1),
                pos=coords[i],
                edge_index=edge_index
            )
            batch_list.append(data)
        
        return Batch.from_data_list(batch_list)


if __name__ == "__main__":
    # Test the diffusion components
    scheduler = DiffusionScheduler(num_timesteps=100, schedule_type='linear')
    loss_fn = DiffusionLoss(loss_type='mse', atom_loss_weight=0.1)
    
    # Test noise addition
    batch_size = 2
    num_atoms = 10
    
    original_coords = torch.randn(batch_size, num_atoms, 3)
    timesteps = torch.randint(0, 100, (batch_size,))
    
    noised_coords, noise = scheduler.add_noise(original_coords, timesteps)
    print(f"Noised coordinates shape: {noised_coords.shape}")
    print(f"Noise shape: {noise.shape}")
    
    # Test loss computation
    predicted_noise = torch.randn_like(noise)
    predicted_atoms = torch.randn(batch_size, num_atoms, 5)
    target_atoms = torch.randint(0, 5, (batch_size, num_atoms))
    mask = torch.ones(batch_size, num_atoms, dtype=torch.bool)
    
    total_loss, coord_loss, atom_loss = loss_fn(
        predicted_noise, noise, predicted_atoms, target_atoms, mask
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Coordinate loss: {coord_loss.item():.4f}")
    print(f"Atom loss: {atom_loss.item():.4f}")