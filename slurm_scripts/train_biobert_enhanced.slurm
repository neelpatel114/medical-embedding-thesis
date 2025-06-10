#!/bin/bash

#SBATCH --job-name=biobert_enh
#SBATCH --partition=gpuq-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=../logs/biobert_enhanced_%j.log
#SBATCH --error=../logs/biobert_enhanced_%j.err

echo "=========================================="
echo "BioBERT Enhanced Training - Optimized"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Navigate to project directory
cd /home/pateln3/medical_embedding_thesis

# Activate environment
source $HOME/miniconda3/bin/activate bert_env

# Verify environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Check GPU status
nvidia-smi

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# Create output directory
OUTPUT_DIR="models/biobert/biobert_enhanced_optimized"
mkdir -p $OUTPUT_DIR

# Copy medical textbook data to local storage for faster access
echo "Setting up data access..."
DATA_SOURCE="/home/pateln3/medical_bert_project/data"
LOCAL_DATA="/tmp/medical_data_$SLURM_JOB_ID"

if [ -d "$DATA_SOURCE" ]; then
    echo "Copying data to local storage for faster access..."
    mkdir -p $LOCAL_DATA
    # Copy enhanced textbook data
    if [ -d "$DATA_SOURCE/enhanced_textbooks/bert" ]; then
        cp -r "$DATA_SOURCE/enhanced_textbooks/bert" "$LOCAL_DATA/enhanced/"
        echo "Enhanced data copied: $(ls $LOCAL_DATA/enhanced/ | wc -l) files"
    else
        echo "Warning: Enhanced data not found at $DATA_SOURCE/enhanced_textbooks/bert"
        # Fallback to any available data
        find "$DATA_SOURCE" -name "enhanced_*.txt" -exec cp {} "$LOCAL_DATA/" \;
        echo "Found enhanced files: $(ls $LOCAL_DATA/enhanced_*.txt 2>/dev/null | wc -l)"
    fi
else
    echo "Warning: Data source not found at $DATA_SOURCE"
    echo "Available directories:"
    ls -la /home/pateln3/
fi

# Run optimized training
python scripts/optimized_training.py \
    --model-name "dmis-lab/biobert-v1.1" \
    --output-dir "$OUTPUT_DIR" \
    --data-dir "$LOCAL_DATA" \
    --use-enhanced-data \
    --batch-size 16 \
    --gradient-accumulation-steps 2 \
    --learning-rate 2e-5 \
    --num-epochs 10 \
    --max-length 512 \
    --warmup-ratio 0.1 \
    --weight-decay 0.01 \
    --logging-steps 50 \
    --eval-steps 500 \
    --save-steps 1000 \
    --fp16 \
    --dataloader-num-workers 4

# Cleanup temporary data
echo "Cleaning up temporary data..."
rm -rf $LOCAL_DATA

echo "Training completed at $(date)"
echo "=========================================="