#!/usr/bin/env python3
"""
Real-time training progress monitor

Usage:
    python scripts/training_status.py
"""
import re
import os
import time
from datetime import datetime, timedelta


def parse_log_file(log_path):
    """Parse training log to extract metrics"""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Find epoch lines
    epoch_pattern = r'Epoch (\d+):.*train_loss=([\d.]+).*train_acc=([\d.]+).*val_loss=([\d.]+).*val_acc=([\d.]+)'

    epochs = []
    for line in lines:
        match = re.search(epoch_pattern, line)
        if match:
            epoch_num = int(match.group(1))
            train_loss = float(match.group(2))
            train_acc = float(match.group(3))
            val_loss = float(match.group(4))
            val_acc = float(match.group(5))

            epochs.append({
                'epoch': epoch_num,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

    return epochs


def get_gpu_info():
    """Get GPU utilization info"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
            return {
                'gpu_util': int(util),
                'mem_used': int(mem_used),
                'mem_total': int(mem_total),
                'temperature': int(temp)
            }
    except:
        pass
    return None


def estimate_remaining_time(epochs_data, total_epochs):
    """Estimate remaining training time"""
    if len(epochs_data) < 2:
        return None

    # Get time between last two epochs
    # Assume ~5 seconds per epoch based on typical ViT training
    epochs_completed = len(epochs_data)
    epochs_remaining = total_epochs - epochs_completed

    # Estimate: ~5 seconds per epoch on RTX 4090
    seconds_per_epoch = 5
    remaining_seconds = epochs_remaining * seconds_per_epoch

    return timedelta(seconds=remaining_seconds)


def display_status():
    """Display training status"""
    log_path = 'logs/baseline_training_250epochs.log'

    print("=" * 80)
    print("🚀 Baseline ViT Training Status")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Parse log
    epochs = parse_log_file(log_path)

    if not epochs:
        print("⚠️  No training data found yet. Training may still be initializing...")
        return

    # Latest epoch
    latest = epochs[-1]
    total_epochs = 250

    print(f"📊 Progress: Epoch {latest['epoch']}/{total_epochs} ({latest['epoch']/total_epochs*100:.1f}%)")
    print()

    # Latest metrics
    print("📈 Latest Metrics:")
    print(f"  Train Loss: {latest['train_loss']:.4f}")
    print(f"  Train Acc:  {latest['train_acc']*100:.2f}%")
    print(f"  Val Loss:   {latest['val_loss']:.4f}")
    print(f"  Val Acc:    {latest['val_acc']*100:.2f}%")
    print()

    # Best validation accuracy
    best_val_acc = max(e['val_acc'] for e in epochs)
    best_epoch = [e for e in epochs if e['val_acc'] == best_val_acc][0]
    print(f"🏆 Best Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch['epoch']})")
    print()

    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info:
        print("💻 GPU Status:")
        print(f"  Utilization: {gpu_info['gpu_util']}%")
        print(f"  Memory:      {gpu_info['mem_used']} MB / {gpu_info['mem_total']} MB ({gpu_info['mem_used']/gpu_info['mem_total']*100:.1f}%)")
        print(f"  Temperature: {gpu_info['temperature']}°C")
        print()

    # Time estimate
    remaining = estimate_remaining_time(epochs, total_epochs)
    if remaining:
        print(f"⏱️  Estimated Time Remaining: {remaining}")
        finish_time = datetime.now() + remaining
        print(f"   Expected Completion: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    # Last 5 epochs
    if len(epochs) >= 5:
        print("📉 Recent Progress (Last 5 Epochs):")
        print("  Epoch | Train Loss | Train Acc | Val Loss | Val Acc")
        print("  ------|------------|-----------|----------|--------")
        for epoch in epochs[-5:]:
            print(f"  {epoch['epoch']:5d} | {epoch['train_loss']:10.4f} | {epoch['train_acc']*100:8.2f}% | {epoch['val_loss']:8.4f} | {epoch['val_acc']*100:6.2f}%")
        print()

    # Checkpoints
    checkpoint_dir = 'checkpoints/baseline_small'
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if checkpoints:
            print(f"💾 Saved Checkpoints: {len(checkpoints)}")
            # Show top 3 by modification time
            checkpoint_paths = [os.path.join(checkpoint_dir, c) for c in checkpoints]
            checkpoint_paths.sort(key=os.path.getmtime, reverse=True)
            for i, ckpt_path in enumerate(checkpoint_paths[:3]):
                ckpt_name = os.path.basename(ckpt_path)
                size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
                print(f"  {i+1}. {ckpt_name} ({size_mb:.1f} MB)")
            print()

    print("=" * 80)


if __name__ == '__main__':
    try:
        display_status()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
