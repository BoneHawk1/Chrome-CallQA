# 3D Molecular Diffusion Model

A complete PyTorch implementation of a 3D molecular generation model using diffusion processes and E(3) Equivariant Graph Neural Networks. This project trains a model to generate novel, chemically valid molecules by learning the distribution of molecular geometries from the QM9 dataset.

## Features

- **E(3) Equivariant Architecture**: Respects rotational and translational symmetries of molecules
- **Diffusion Process**: Implements both forward (noising) and reverse (denoising) processes
- **Distributed Training**: Multi-GPU training support using PyTorch DDP
- **Chemical Validation**: RDKit integration for molecule validation
- **QM9 Dataset Support**: Custom dataset loader for molecular data
- **Comprehensive Logging**: WandB integration for experiment tracking

## Project Structure

```
├── dataset.py          # QM9 dataset loader and data processing
├── model.py            # E(3) Equivariant GNN architecture
├── diffusion.py        # Diffusion process and loss functions
├── train.py            # Distributed training script
├── inference.py        # Molecule generation and validation
├── config.yaml         # Configuration file
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd 3d-molecular-diffusion
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare the QM9 dataset**:
   - Download the QM9 dataset and place `.xyz` files in the `data/` directory
   - The dataset should contain molecular structures in XYZ format

## Usage

### Training

#### Single GPU Training
```bash
python train.py --data_dir data/ --config config.yaml
```

#### Multi-GPU Training (Tesla P40 + RTX 5070 Ti)
```bash
# Launch training on 2 GPUs
torchrun --nproc_per_node=2 train.py --data_dir data/ --config config.yaml
```

#### Resume Training
```bash
python train.py --data_dir data/ --config config.yaml --resume checkpoints/checkpoint_epoch_50.pth
```

### Inference

#### Generate Molecules
```bash
python inference.py \
    --model_path checkpoints/best_model.pth \
    --output_dir generated_molecules \
    --num_molecules 100 \
    --num_atoms 15 \
    --validate
```

#### Generate with Classifier-Free Guidance
```bash
python inference.py \
    --model_path checkpoints/best_model.pth \
    --output_dir generated_molecules \
    --num_molecules 50 \
    --num_atoms 10 \
    --guidance_scale 2.0 \
    --validate
```

## Model Architecture

### E(3) Equivariant Graph Neural Network

The model uses an E(3) equivariant architecture that respects the rotational and translational symmetries of molecules:

- **Input**: Molecular graph with atom features and 3D coordinates
- **Architecture**: Stack of equivariant message-passing layers
- **Output**: Predicted noise for coordinates and atom type probabilities

### Diffusion Process

1. **Forward Process**: Gradually adds noise to molecular coordinates
2. **Reverse Process**: Denoises coordinates to generate new molecules
3. **Training**: Model learns to predict the noise added during the forward process

## Configuration

The `config.yaml` file contains all hyperparameters:

```yaml
model:
  node_dim: 1              # Input node feature dimension
  hidden_dim: 128          # Hidden layer dimension
  num_layers: 6            # Number of GNN layers
  max_atoms: 29            # Maximum atoms per molecule

diffusion:
  num_timesteps: 1000      # Number of diffusion steps
  beta_start: 1e-4         # Initial noise level
  beta_end: 0.02           # Final noise level
  schedule_type: "linear"  # Noise schedule type

training:
  batch_size: 16           # Batch size per GPU
  num_epochs: 100          # Number of training epochs
  learning_rate: 1e-4      # Learning rate
  weight_decay: 1e-5       # Weight decay
  warmup_steps: 1000       # Learning rate warmup steps
```

## Dataset Format

The model expects QM9 data in XYZ format:

```
<number_of_atoms>
<molecule_name>
<atom_symbol> <x> <y> <z>
<atom_symbol> <x> <y> <z>
...
```

Example:
```
3
Methane
C 0.000000 0.000000 0.000000
H 1.089000 0.000000 0.000000
H -0.363000 1.033000 0.000000
H -0.363000 -0.516500 0.895000
```

## Output

### Training Outputs
- **Checkpoints**: Saved in `checkpoints/` directory
- **Logs**: Training logs in `logs/` directory
- **WandB**: Experiment tracking and metrics

### Inference Outputs
- **XYZ Files**: Generated molecules in XYZ format
- **Validation Results**: Chemical validity assessment using RDKit
- **JSON Summary**: Generation statistics and validation metrics

## Performance

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (Tesla P40, RTX 5070 Ti, etc.)
- **Memory**: 8GB+ GPU memory recommended
- **Storage**: 10GB+ for dataset and checkpoints

### Training Time
- **Single GPU**: ~24-48 hours for 100 epochs
- **Multi-GPU**: ~12-24 hours for 100 epochs (2x speedup)

## Validation

The model includes comprehensive chemical validation:

- **Bond Lengths**: Checks for reasonable interatomic distances
- **Molecular Connectivity**: Ensures molecules are properly connected
- **Chemical Validity**: RDKit-based validation
- **SMILES Generation**: Converts generated structures to SMILES notation

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in `config.yaml`
   - Use gradient accumulation
   - Reduce model size (hidden_dim, num_layers)

2. **Slow Training**:
   - Increase `num_workers` in data loading
   - Use mixed precision training
   - Optimize data preprocessing

3. **Poor Generation Quality**:
   - Increase training epochs
   - Adjust diffusion parameters
   - Use classifier-free guidance

### Debug Mode
```bash
# Run with debug logging
python train.py --data_dir data/ --config config.yaml --debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{3d-molecular-diffusion,
  title={3D Molecular Generation using E(3) Equivariant Diffusion Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

- QM9 dataset creators
- PyTorch Geometric team
- RDKit developers
- E(3) equivariant neural network research community