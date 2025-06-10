#!/bin/bash

# Master script to submit all training jobs for medical embedding thesis
# This script manages the complete training pipeline with proper dependencies

echo "=========================================="
echo "Medical Embedding Thesis - Training Pipeline"
echo "=========================================="

# Navigate to project directory
cd /home/pateln3/medical_embedding_thesis/slurm_scripts

# Check if environment setup is needed
if [ ! -f "../environment_info.txt" ]; then
    echo "Setting up environment first..."
    SETUP_JOB=$(sbatch --parsable setup_environment.sh)
    echo "Environment setup job submitted: $SETUP_JOB"
    
    # Wait for environment setup to complete
    echo "Waiting for environment setup to complete..."
    while squeue -j $SETUP_JOB -h &>/dev/null; do
        sleep 30
    done
    
    # Check if setup was successful
    if [ ! -f "../environment_info.txt" ]; then
        echo "Environment setup failed! Check logs/setup_env_${SETUP_JOB}.log"
        exit 1
    fi
    echo "Environment setup completed successfully"
else
    echo "Environment already set up, proceeding with training..."
fi

# Submit training jobs with proper resource allocation
echo "Submitting training jobs..."

# Submit BERT baseline jobs (can run in parallel on A40s)
echo "Submitting BERT baseline training jobs..."
BERT_RAW_JOB=$(sbatch --parsable train_bert_raw.slurm)
BERT_ENH_JOB=$(sbatch --parsable train_bert_enhanced.slurm)

echo "BERT Raw training job: $BERT_RAW_JOB"
echo "BERT Enhanced training job: $BERT_ENH_JOB"

# Submit medical BERT variants after baseline jobs start
echo "Submitting medical BERT variant training jobs..."
sleep 60  # Brief delay to avoid resource conflicts

BIOBERT_JOB=$(sbatch --parsable train_biobert_enhanced.slurm)
CLINBERT_JOB=$(sbatch --parsable train_clinicalbert_enhanced.slurm)

echo "BioBERT Enhanced training job: $BIOBERT_JOB"
echo "ClinicalBERT Enhanced training job: $CLINBERT_JOB"

# Create job tracking file
cat > ../training_jobs.txt << EOF
# Medical Embedding Thesis Training Jobs
# Started: $(date)

BERT_RAW_JOB=$BERT_RAW_JOB
BERT_ENH_JOB=$BERT_ENH_JOB
BIOBERT_JOB=$BIOBERT_JOB
CLINBERT_JOB=$CLINBERT_JOB

# Monitor with: squeue -u $USER
# Check logs in: logs/
EOF

echo "=========================================="
echo "All training jobs submitted!"
echo "Job tracking saved to: training_jobs.txt"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo "  tail -f logs/*_\${JOB_ID}.log"
echo ""
echo "Expected completion times:"
echo "  BERT baseline models: ~24-36 hours"
echo "  Medical BERT variants: ~24-36 hours"
echo "=========================================="