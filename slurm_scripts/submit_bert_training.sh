#!/bin/bash

# Direct BERT training submission (environment already set up)
echo "=========================================="
echo "BERT Medical Embedding Training - Direct Launch"
echo "=========================================="

cd /home/pateln3/medical_embedding_thesis/slurm_scripts

echo "Environment check:"
source $HOME/miniconda3/bin/activate bert_env
python -c "
import torch
import transformers
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ Environment ready!')
"

echo "Submitting BERT training jobs..."

# Submit BERT baseline jobs  
echo "Submitting BERT baseline training jobs..."
BERT_RAW_JOB=$(sbatch --parsable train_bert_raw.slurm)
BERT_ENH_JOB=$(sbatch --parsable train_bert_enhanced.slurm)

echo "BERT Raw training job: $BERT_RAW_JOB"
echo "BERT Enhanced training job: $BERT_ENH_JOB"

# Submit medical BERT variants with slight delay
echo "Submitting medical BERT variant training jobs..."
sleep 30  # Brief delay

BIOBERT_JOB=$(sbatch --parsable train_biobert_enhanced.slurm)
CLINBERT_JOB=$(sbatch --parsable train_clinicalbert_enhanced.slurm)

echo "BioBERT Enhanced training job: $BIOBERT_JOB"
echo "ClinicalBERT Enhanced training job: $CLINBERT_JOB"

# Create job tracking file
cat > ../training_jobs.txt << EOF
# BERT Medical Embedding Training Jobs
# Started: $(date)

BERT_RAW_JOB=$BERT_RAW_JOB
BERT_ENH_JOB=$BERT_ENH_JOB
BIOBERT_JOB=$BIOBERT_JOB
CLINBERT_JOB=$CLINBERT_JOB

# Monitor with: squeue -u $USER
# Check logs in: logs/
EOF

echo "=========================================="
echo "All BERT training jobs submitted!"
echo "Job tracking saved to: training_jobs.txt"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo "  tail -f logs/bert_enhanced_*.log"
echo ""
echo "Expected completion times:"
echo "  All BERT models: ~24-36 hours each"
echo "=========================================="