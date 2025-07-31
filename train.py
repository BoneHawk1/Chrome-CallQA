import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import logging
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf
import random
import numpy as np
import math

from dataset import QM9Dataset, create_distributed_dataloader
from model import DiffusionModel
from diffusion import DiffusionScheduler, DiffusionLoss


def setup_logging(rank: int, log_dir: str = "logs"):
    """Setup logging for distributed training."""
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def setup_distributed(rank: int, world_size: int, port: str = "12355"):
    """Setup distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training environment."""
    dist.destroy_process_group()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_edge_index(num_atoms: int, device: torch.device) -> torch.Tensor:
    """Create fully connected edge indices for molecular graphs."""
    edges = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                edges.append([i, j])
    return torch.tensor(edges, dtype=torch.long, device=device).t()


def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler._LRScheduler,
                loss_fn: nn.Module,
                diffusion_scheduler: DiffusionScheduler,
                device: torch.device,
                rank: int,
                epoch: int) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Progress bar only for rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        batch = batch.to(device)
        
        # Get batch size and number of atoms
        batch_size = batch.num_graphs
        num_atoms = batch.pos.size(0) // batch_size
        
        # Create edge indices for the batch
        edge_index = create_edge_index(num_atoms, device)
        
        # Sample random timesteps
        timesteps = torch.randint(0, diffusion_scheduler.num_timesteps, 
                                (batch_size,), device=device)
        
        # Get original coordinates and atom types
        original_coords = batch.pos.view(batch_size, num_atoms, 3)
        atom_types = batch.x.view(batch_size, num_atoms).squeeze(-1)
        mask = batch.mask.view(batch_size, num_atoms)
        
        # Add noise to coordinates
        noised_coords, noise = diffusion_scheduler.add_noise(original_coords, timesteps)
        
        # Update batch with noised coordinates
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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        if rank == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Coord Loss': f'{coord_loss.item():.4f}',
                'Atom Loss': f'{atom_loss.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })
    
    return total_loss / num_batches


def validate(model: nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: nn.Module,
             diffusion_scheduler: DiffusionScheduler,
             device: torch.device,
             rank: int) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            batch_size = batch.num_graphs
            num_atoms = batch.pos.size(0) // batch_size
            
            # Create edge indices
            edge_index = create_edge_index(num_atoms, device)
            
            # Sample timesteps
            timesteps = torch.randint(0, diffusion_scheduler.num_timesteps, 
                                    (batch_size,), device=device)
            
            # Get data
            original_coords = batch.pos.view(batch_size, num_atoms, 3)
            atom_types = batch.x.view(batch_size, num_atoms).squeeze(-1)
            mask = batch.mask.view(batch_size, num_atoms)
            
            # Add noise
            noised_coords, noise = diffusion_scheduler.add_noise(original_coords, timesteps)
            batch.pos = noised_coords.view(-1, 3)
            
            # Forward pass
            predicted_noise, predicted_atoms = model(batch, timesteps)
            
            # Reshape predictions
            predicted_noise = predicted_noise.view(batch_size, num_atoms, 3)
            predicted_atoms = predicted_atoms.view(batch_size, num_atoms, -1)
            
            # Compute loss
            loss, _, _ = loss_fn(predicted_noise, noise, predicted_atoms, atom_types, mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def save_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   scheduler: optim.lr_scheduler._LRScheduler,
                   epoch: int,
                   loss: float,
                   save_path: str,
                   rank: int):
    """Save model checkpoint."""
    if rank == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, save_path)
        logging.info(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train 3D Molecular Diffusion Model')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = OmegaConf.load(args.config)
    else:
        # Default configuration
        config = OmegaConf.create({
            'model': {
                'node_dim': 1,
                'hidden_dim': 128,
                'num_layers': 6,
                'max_atoms': 29
            },
            'diffusion': {
                'num_timesteps': 1000,
                'beta_start': 1e-4,
                'beta_end': 0.02,
                'schedule_type': 'linear'
            },
            'training': {
                'batch_size': 16,
                'num_epochs': 100,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'warmup_steps': 1000,
                'save_every': 10,
                'val_every': 5
            },
            'data': {
                'max_atoms': 29,
                'num_workers': 4
            }
        })
    
    # Setup distributed training
    setup_distributed(args.local_rank, args.world_size)
    setup_logging(args.local_rank)
    
    # Set device
    device = torch.device(f'cuda:{args.local_rank}')
    
    # Set seed
    set_seed(42 + args.local_rank)
    
    # Initialize wandb (only for rank 0)
    if args.local_rank == 0:
        wandb.init(
            project="3d-molecular-diffusion",
            config=OmegaConf.to_container(config, resolve=True),
            name=f"qm9_diffusion_{args.world_size}gpu"
        )
    
    # Create dataset and dataloader
    dataset = QM9Dataset(
        data_dir=args.data_dir,
        max_atoms=config.data.max_atoms
    )
    
    dataloader = create_distributed_dataloader(
        dataset=dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        world_size=args.world_size,
        rank=args.local_rank
    )
    
    # Create validation dataset (use a subset)
    val_size = min(len(dataset) // 10, 1000)  # 10% or 1000 samples
    val_dataset = torch.utils.data.Subset(dataset, range(val_size))
    val_dataloader = create_distributed_dataloader(
        dataset=val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        world_size=args.world_size,
        rank=args.local_rank
    )
    
    # Create model
    model = DiffusionModel(
        node_dim=config.model.node_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        max_atoms=config.model.max_atoms
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # Create diffusion scheduler and loss function
    diffusion_scheduler = DiffusionScheduler(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule_type=config.diffusion.schedule_type
    ).to(device)
    
    loss_fn = DiffusionLoss(loss_type='mse', atom_loss_weight=0.1).to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Cosine learning rate scheduler with warmup
    def lr_lambda(step):
        if step < config.training.warmup_steps:
            return step / config.training.warmup_steps
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - config.training.warmup_steps) / 
                                     (len(dataloader) * config.training.num_epochs - config.training.warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.training.num_epochs):
        # Set epoch for distributed sampler
        dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            diffusion_scheduler=diffusion_scheduler,
            device=device,
            rank=args.local_rank,
            epoch=epoch
        )
        
        # Validate
        if epoch % config.training.val_every == 0:
            val_loss = validate(
                model=model,
                dataloader=val_dataloader,
                loss_fn=loss_fn,
                diffusion_scheduler=diffusion_scheduler,
                device=device,
                rank=args.local_rank
            )
            
            if args.local_rank == 0:
                logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': scheduler.get_last_lr()[0]
                })
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        loss=val_loss,
                        save_path='checkpoints/best_model.pth',
                        rank=args.local_rank
                    )
        else:
            if args.local_rank == 0:
                logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'learning_rate': scheduler.get_last_lr()[0]
                })
        
        # Save checkpoint periodically
        if epoch % config.training.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=train_loss,
                save_path=f'checkpoints/checkpoint_epoch_{epoch}.pth',
                rank=args.local_rank
            )
    
    # Cleanup
    cleanup_distributed()
    
    if args.local_rank == 0:
        wandb.finish()


if __name__ == "__main__":
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    main()