#!/bin/bash
#SBATCH --job-name=bert_simple
#SBATCH --output=/home/pateln3/medical_embedding_thesis/logs/bert_simple_%j.log
#SBATCH --error=/home/pateln3/medical_embedding_thesis/logs/bert_simple_%j.err
#SBATCH --partition=gpuq-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

echo "=========================================="
echo "Simple BERT Training - Debug Version"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Environment setup
source $HOME/miniconda3/bin/activate bert_env

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# GPU info
nvidia-smi

# Change to project directory
cd /home/pateln3/medical_embedding_thesis

# Create output directories
mkdir -p models/bert_raw_simple
mkdir -p models/bert_enhanced_simple
mkdir -p logs

echo "Starting BERT Raw training..."
python scripts/simple_bert_training.py \
    --model-type bert_raw \
    --output-dir models/bert_raw_simple

echo "Starting BERT Enhanced training..."  
python scripts/simple_bert_training.py \
    --model-type bert_enhanced \
    --output-dir models/bert_enhanced_simple

echo "Training completed at $(date)"
echo "=========================================="