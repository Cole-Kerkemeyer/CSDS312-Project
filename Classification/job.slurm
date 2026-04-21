#!/bin/bash

#SBATCH --output=./train_%j.out
#SBATCH --nodes 1
#SBATCH --time=0-01:00:00
#SBATCH --job-name="classification"
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu2080
#SBATCH --mem=16gb

cd /home/cmk265/datasets/brisc2025

source /home/cmk265/csds312/p3rh8venv/bin/activate

python cnn.py

