#!/bin/bash
#SBATCH -A csds312
#SBATCH --job-name=brats_25d_train
#SBATCH --output=segmentation/logs/brats_train_%j.out   # Standard output log (%j = job ID)
#SBATCH --error=segmentation/logs/brats_train_%j.err    # Standard error log
#SBATCH --time=2-00:00:00                    # Time limit (HH:MM:SS)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks (usually 1 for standard PyTorch)
#SBATCH --cpus-per-task=16                  # Increased from 4 to feed multiple GPUs
#SBATCH --mem=64G                          # Increased from 32G for data loading safety
#SBATCH --partition=markov_gpu             
#SBATCH --gres=gpu:4                       # Requesting 2 GPUs!
#SBATCH --constraint=gpu4090


# 1. Navigate to your specific working directory
cd /mnt/vstor/courses/csds312/sgl57/CSDS312-Project

# 2. Activate your virtual environment
source .venv/bin/activate

# 3. (Optional but recommended) Ensure W&B syncs correctly on cluster nodes
export WANDB_MODE=online
export WANDB_DIR="/mnt/vstor/courses/csds312/sgl57/CSDS312-Project/segmentation"

# 4. Run the training script
echo "Starting PyTorch training..."
python segmentation/train.py
echo "Training completed."