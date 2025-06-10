#!/bin/bash

#SBATCH --job-name=setup_env
#SBATCH --partition=gpuq-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/setup_env_%j.log
#SBATCH --error=../logs/setup_env_%j.err

echo "=========================================="
echo "Setting up Medical Embedding Environment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Navigate to project directory
cd /home/pateln3/medical_embedding_thesis

# Load/create conda environment
source $HOME/miniconda3/bin/activate

# Remove old problematic environments
conda env remove -n medical_embedding_env -y 2>/dev/null || true

# Create fresh environment with compatible versions
echo "Creating fresh conda environment..."
conda create -y -n medical_embedding_env python=3.9

# Activate new environment
conda activate medical_embedding_env

# Install compatible PyTorch and dependencies
echo "Installing PyTorch 2.0.1 with CUDA 11.8..."
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install transformers and datasets (compatible versions)
echo "Installing Transformers and datasets..."
pip install transformers==4.30.2 datasets==2.14.5 tokenizers==0.13.3

# Install other ML dependencies
echo "Installing ML dependencies..."
pip install accelerate==0.20.3 evaluate==0.4.0 scikit-learn==1.3.0

# Install training utilities
echo "Installing training utilities..."
pip install tensorboard==2.13.0 wandb==0.15.5 tqdm==4.65.0

# Install data processing
echo "Installing data processing..."
pip install numpy==1.24.3 pandas==2.0.3 scipy==1.11.1

# Install visualization
echo "Installing visualization..."
pip install matplotlib==3.7.2 seaborn==0.12.2

# Verify installation
echo "Verifying installation..."
python -c "
import torch
import transformers
import datasets
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
print(f'Transformers version: {transformers.__version__}')
print(f'Datasets version: {datasets.__version__}')
"

# Save environment info
echo "Saving environment info..."
conda list > environment_info.txt
pip freeze > pip_requirements.txt

echo "Environment setup completed at $(date)"
echo "=========================================="