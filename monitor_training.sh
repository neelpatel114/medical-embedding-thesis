#\!/bin/bash
echo "=== Medical Embedding Training Monitor ==="
echo "Started: $(date)"
echo
echo "Job Status:"
squeue -u $USER
echo
echo "Recent Log Activity:"
echo "--- Environment Setup ---"
tail -5 logs/setup_env_*.log 2>/dev/null || echo "Setup logs not ready yet"
echo
echo "--- Training Logs ---"
for log in logs/bert_*.log logs/modernbert_*.log; do
    if [ -f "$log" ]; then
        echo "=== $(basename $log) ==="
        tail -3 "$log"
        echo
    fi
done
echo "Monitor script created. Run with: ./monitor_training.sh"
