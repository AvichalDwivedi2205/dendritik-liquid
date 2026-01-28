"""
Evaluation metrics for model comparison

Metrics:
- FLOPs counting
- Parameter counting
- Inference latency measurement
- Accuracy computation
"""
import torch
import time
from fvcore.nn import FlopCountAnalysis, parameter_count


def count_flops(model, input_shape=(1, 3, 32, 32), device='cuda'):
    """
    Count FLOPs using fvcore

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on

    Returns:
        flops: Total FLOPs (float)
    """
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    with torch.no_grad():
        flops_counter = FlopCountAnalysis(model, dummy_input)
        total_flops = flops_counter.total()

    return total_flops


def count_parameters(model):
    """
    Count trainable and total parameters

    Args:
        model: PyTorch model

    Returns:
        dict with 'total' and 'trainable' counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params
    }


def measure_latency(model, input_shape=(1, 3, 32, 32), device='cuda', num_runs=100, warmup=10):
    """
    Measure inference latency (GPU)

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on
        num_runs: Number of inference runs
        warmup: Number of warmup runs

    Returns:
        avg_latency: Average latency in milliseconds
    """
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Measure
    if device == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    avg_latency = (end_time - start_time) / num_runs * 1000  # Convert to ms
    return avg_latency


def compute_accuracy(model, dataloader, device='cuda', top_k=(1, 5)):
    """
    Compute top-k accuracy

    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to run on
        top_k: Tuple of k values for top-k accuracy

    Returns:
        dict with accuracy for each k
    """
    model.eval()
    model = model.to(device)

    correct = {k: 0 for k in top_k}
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            batch_size = labels.size(0)
            total += batch_size

            # Compute top-k accuracy
            _, pred = logits.topk(max(top_k), dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct_mask = pred.eq(labels.view(1, -1).expand_as(pred))

            for k in top_k:
                correct_k = correct_mask[:k].reshape(-1).float().sum(0)
                correct[k] += correct_k.item()

    # Compute percentages
    accuracy = {k: (correct[k] / total) * 100 for k in top_k}
    return accuracy


def model_summary(model, input_shape=(1, 3, 32, 32), device='cuda'):
    """
    Print comprehensive model summary

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on
    """
    model = model.to(device)

    print("=" * 70)
    print("Model Summary")
    print("=" * 70)

    # Parameters
    params = count_parameters(model)
    print(f"\n📊 Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")

    # FLOPs
    flops = count_flops(model, input_shape, device)
    print(f"\n⚡ FLOPs:")
    print(f"  Total: {flops:,}")
    print(f"  GFLOPs: {flops / 1e9:.2f}")

    # Latency
    latency = measure_latency(model, input_shape, device)
    print(f"\n⏱️  Latency (single sample):")
    print(f"  Average: {latency:.2f} ms")
    print(f"  Throughput: {1000 / latency:.1f} images/sec")

    # Model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"\n💾 Model Size:")
    print(f"  {model_size:.2f} MB")

    print("=" * 70)


if __name__ == '__main__':
    from src.models.vit_base import create_vit_small

    print("Testing Evaluation Metrics")
    print("=" * 70)

    # Create model
    model = create_vit_small(num_classes=10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test all metrics
    model_summary(model, device=device)

    print("\n✓ All metric tests passed!")
