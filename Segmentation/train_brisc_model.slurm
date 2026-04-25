#!/bin/bash
#SBATCH -A csds312
#SBATCH --job-name=brisc25_train
#SBATCH --output=segmentation/logs/brisc_train_%j.out   # Standard output log (%j = job ID)
#SBATCH --error=segmentation/logs/brisc_train_%j.err    # Standard error log
#SBATCH --time=2-00:00:00                    # Time limit (HH:MM:SS)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks (usually 1 for standard PyTorch)
#SBATCH --cpus-per-task=8                  # Increased from 4 to feed multiple GPUs
#SBATCH --mem=64G                          # Increased from 32G for data loading safety
#SBATCH --partition=markov_gpu             
#SBATCH --gres=gpu:1                       
#SBATCH --constraint=gpul40s

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

module load CUDA/12.8.0
# 1. Navigate to your specific working directory
cd /mnt/vstor/courses/csds312/sgl57/CSDS312-Project

# 2. Activate your virtual environment
source .venv/bin/activate

# 3. (Optional but recommended) Ensure W&B syncs correctly on cluster nodes
export WANDB_MODE=online
export WANDB_DIR="/mnt/vstor/courses/csds312/sgl57/CSDS312-Project/segmentation/wandb/BRISC"

# 4. Run the training script
echo "Starting PyTorch training..."
python segmentation/train_brisc.py
echo "Training completed."