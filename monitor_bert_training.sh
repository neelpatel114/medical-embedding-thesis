#\!/bin/bash
echo "=== BERT-Focused Medical Embedding Training Monitor ==="
echo "Started: $(date)"
echo

echo "Current Jobs:"
squeue -u $USER
echo

echo "Environment Setup Progress:"
if [ -f "logs/setup_env_*.log" ]; then
    echo "--- Latest Setup Log ---"
    tail -5 logs/setup_env_*.log 2>/dev/null
    echo
fi

echo "Training Progress:"
for log in logs/bert_*.log logs/biobert_*.log logs/clinical*.log; do
    if [ -f "$log" ]; then
        echo "=== $(basename $log) ==="
        tail -3 "$log"
        echo
    fi
done

echo "Expected Jobs:"
echo "1. BERT Raw (raw medical textbooks)"
echo "2. BERT Enhanced (LLM-enhanced textbooks)" 
echo "3. BioBERT Enhanced (biomedical pre-trained)"
echo "4. ClinicalBERT Enhanced (clinical pre-trained)"
echo
echo "Monitor: ./monitor_bert_training.sh"
