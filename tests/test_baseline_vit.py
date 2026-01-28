"""
Unit tests for baseline Vision Transformer

Tests:
- Model forward pass
- Model output shapes
- Gradient flow
- Overfitting on small batch (sanity check)
"""
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.vit_base import create_vit_tiny, create_vit_small
from src.data.cifar_dataset import get_cifar10_dataloaders


def test_model_forward():
    """Test basic forward pass"""
    model = create_vit_tiny(num_classes=10)
    x = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (4, 10), f"Expected (4, 10), got {logits.shape}"
    print("✓ Forward pass test passed")


def test_model_gradients():
    """Test gradient flow"""
    model = create_vit_tiny(num_classes=10)
    x = torch.randn(4, 3, 32, 32, requires_grad=True)

    logits = model(x)
    loss = logits.sum()
    loss.backward()

    assert x.grad is not None, "No gradients flowing to input"
    assert not torch.isnan(x.grad).any(), "NaN gradients detected"

    # Check all parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    print("✓ Gradient flow test passed")


def test_attention_maps():
    """Test attention map extraction"""
    model = create_vit_tiny(num_classes=10)
    x = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        attn_maps = model.get_attention_maps(x)

    assert len(attn_maps) == 6, f"Expected 6 attention maps, got {len(attn_maps)}"

    # Check shape of each attention map
    for i, attn in enumerate(attn_maps):
        # [batch, num_heads, seq_len, seq_len]
        # seq_len = 64 patches + 1 CLS token = 65
        assert attn.shape == (4, 3, 65, 65), f"Layer {i}: unexpected shape {attn.shape}"

    print("✓ Attention map extraction test passed")


def test_overfitting_sanity_check():
    """
    Sanity check: Can the model overfit on a single batch?

    This ensures:
    - Model has capacity to learn
    - Gradients are flowing correctly
    - No major bugs in training loop
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_vit_tiny(num_classes=10).to(device)

    # Get one batch
    train_loader, _ = get_cifar10_dataloaders(batch_size=32, num_workers=0)
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    # Train for 50 iterations
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    initial_loss = None
    final_loss = None

    for i in range(50):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        if i == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if i == 49:
            final_loss = loss.item()

    # Model should significantly reduce loss on the same batch
    assert final_loss < initial_loss * 0.5, \
        f"Model failed to overfit. Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"

    # Check accuracy on this batch
    with torch.no_grad():
        logits = model(images)
        acc = (logits.argmax(dim=1) == labels).float().mean().item()

    assert acc > 0.8, f"Model should achieve >80% on overfitting test, got {acc*100:.2f}%"

    print(f"✓ Overfitting test passed (loss: {initial_loss:.4f} → {final_loss:.4f}, acc: {acc*100:.2f}%)")


def test_dataloader():
    """Test dataloader"""
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=128, num_workers=0)

    assert len(train_loader) > 0, "Empty training dataloader"
    assert len(test_loader) > 0, "Empty test dataloader"

    # Get one batch
    images, labels = next(iter(train_loader))
    assert images.shape == (128, 3, 32, 32), f"Unexpected batch shape: {images.shape}"
    assert labels.shape == (128,), f"Unexpected labels shape: {labels.shape}"

    print("✓ Dataloader test passed")


if __name__ == '__main__':
    print("=" * 70)
    print("Running Baseline ViT Tests")
    print("=" * 70)

    test_model_forward()
    test_model_gradients()
    test_attention_maps()
    test_dataloader()
    test_overfitting_sanity_check()

    print("=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
