#!/bin/bash

# Launch script for distributed training of 3D Molecular Diffusion Model
# Usage: ./launch_training.sh [num_gpus] [data_dir] [config_file]

set -e

# Default values
NUM_GPUS=${1:-2}
DATA_DIR=${2:-"data/"}
CONFIG_FILE=${3:-"config.yaml"}

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not found. Please ensure CUDA is installed."
    exit 1
fi

# Check number of available GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS are available."
    echo "Using $AVAILABLE_GPUS GPUs instead."
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' not found."
    echo "Please ensure the QM9 dataset is placed in the data/ directory."
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file '$CONFIG_FILE' not found. Using default configuration."
fi

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p generated_molecules

echo "Starting distributed training with $NUM_GPUS GPUs..."
echo "Data directory: $DATA_DIR"
echo "Config file: $CONFIG_FILE"
echo ""

# Launch distributed training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=12355 \
    train.py \
    --data_dir "$DATA_DIR" \
    --config "$CONFIG_FILE" \
    --world_size $NUM_GPUS

echo ""
echo "Training completed!"
echo "Checkpoints saved in: checkpoints/"
echo "Logs saved in: logs/"