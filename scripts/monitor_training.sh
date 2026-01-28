#!/bin/bash
# Monitor baseline training progress

echo "========================================"
echo "Baseline ViT Training Monitor"
echo "========================================"
echo ""

# Check if training is running
if pgrep -f "train_baseline.py" > /dev/null; then
    echo "✓ Training is RUNNING"

    # Show GPU usage
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1

    # Show latest training metrics
    echo ""
    echo "Latest Training Metrics:"
    tail -30 logs/baseline_training_250epochs.log | grep -E "(Epoch|train_loss|val_acc|lr-AdamW)" | tail -10

    # Count epochs completed
    echo ""
    COMPLETED=$(grep -c "Epoch " logs/baseline_training_250epochs.log || echo "0")
    echo "Epochs completed: $COMPLETED / 250"

    # Show checkpoints
    echo ""
    echo "Saved Checkpoints:"
    ls -lh checkpoints/baseline_small/*.ckpt 2>/dev/null || echo "No checkpoints yet"

else
    echo "✗ Training is NOT running"

    # Check if it completed or crashed
    if grep -q "Training completed" logs/baseline_training_250epochs.log 2>/dev/null; then
        echo ""
        echo "✓ Training completed successfully!"
        tail -20 logs/baseline_training_250epochs.log
    else
        echo ""
        echo "⚠ Training may have crashed. Last 50 lines:"
        tail -50 logs/baseline_training_250epochs.log
    fi
fi

echo ""
echo "========================================"
