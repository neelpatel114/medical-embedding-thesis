#!/bin/bash

echo "=== BERT Training Monitor ==="
echo "Job Status:"
squeue -u pateln3

echo -e "\nLatest Log Output:"
tail -10 /home/pateln3/medical_embedding_thesis/logs/bert_simple_41312.log

echo -e "\nModel Directories:"
ls -la /home/pateln3/medical_embedding_thesis/models/bert_*_simple/ | head -20

echo -e "\nTraining Logs:"
if [ -f "/home/pateln3/medical_embedding_thesis/models/bert_raw_simple/bert_raw_training.log" ]; then
    echo "BERT Raw Log:"
    tail -5 /home/pateln3/medical_embedding_thesis/models/bert_raw_simple/bert_raw_training.log
fi

if [ -f "/home/pateln3/medical_embedding_thesis/models/bert_enhanced_simple/bert_enhanced_training.log" ]; then
    echo "BERT Enhanced Log:"
    tail -5 /home/pateln3/medical_embedding_thesis/models/bert_enhanced_simple/bert_enhanced_training.log
fi

echo -e "\nGPU Usage:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1

echo "=========================="