"""
Unit tests for Dendritic Gating Module

Tests cover:
- Output shapes
- Gradient flow
- Positive bias initialization
- Branch specialization
- Max-pooling behavior
"""
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.dendritic_module import DendriticBranch, DendriticGating


def test_dendritic_branch_shape():
    """Test single branch output shapes"""
    batch_size = 4
    num_patches = 64
    dim = 256

    x = torch.randn(batch_size, num_patches, dim)
    branch = DendriticBranch(dim, hidden_dim=64)

    importance = branch(x)

    assert importance.shape == (batch_size, num_patches, 1), \
        f"Expected shape (4, 64, 1), got {importance.shape}"

    assert torch.all((importance >= 0) & (importance <= 1)), \
        "Importance scores must be in [0, 1]"

    print("✓ Branch shape test passed")


def test_dendritic_gating_shape():
    """Test multi-branch gating output shapes"""
    batch_size = 4
    num_patches = 64
    dim = 256
    num_branches = 4

    x = torch.randn(batch_size, num_patches, dim)
    dendritic = DendriticGating(dim, num_branches)

    importance = dendritic(x)

    assert importance.shape == (batch_size, num_patches), \
        f"Expected shape (4, 64), got {importance.shape}"

    assert torch.all((importance >= 0) & (importance <= 1)), \
        "Importance scores must be in [0, 1]"

    print("✓ Multi-branch shape test passed")


def test_gradient_flow():
    """Test gradient flow through dendritic module"""
    x = torch.randn(4, 64, 256, requires_grad=True)
    dendritic = DendriticGating(256, num_branches=4)

    importance = dendritic(x)
    loss = importance.mean()
    loss.backward()

    # Check input gradients
    assert x.grad is not None, "No gradients flowing to input"
    assert not torch.isnan(x.grad).any(), "NaN gradients in input"

    # Check all parameters have gradients
    for name, param in dendritic.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    print("✓ Gradient flow test passed")


def test_positive_bias_initialization():
    """
    Verify positive bias initialization

    Critical test: With positive bias, initial importance should be high (>0.6)
    This prevents the model from pruning all tokens at the start of training
    """
    x = torch.randn(4, 64, 256)
    dendritic = DendriticGating(256, num_branches=4)

    with torch.no_grad():
        importance = dendritic(x)

    mean_importance = importance.mean().item()

    assert mean_importance > 0.6, \
        f"Mean importance too low: {mean_importance:.3f}. Expected >0.6 (positive bias not working)"

    # Check distribution
    high_importance_count = (importance > 0.7).sum().item()
    total_tokens = importance.numel()

    assert high_importance_count / total_tokens > 0.5, \
        f"Not enough high-importance tokens. Expected >50%, got {high_importance_count/total_tokens*100:.1f}%"

    print(f"✓ Positive bias test passed (mean importance: {mean_importance:.3f})")


def test_max_pooling_behavior():
    """
    Test that max-pooling correctly aggregates branch scores

    If one branch gives high score and others low, max should be high
    """
    batch_size = 2
    num_patches = 10
    dim = 256

    x = torch.randn(batch_size, num_patches, dim)
    dendritic = DendriticGating(dim, num_branches=4)

    # Get individual branch scores
    with torch.no_grad():
        branch_scores = dendritic.get_branch_activations(x)  # [2, 10, 4]
        max_scores = dendritic(x)  # [2, 10]

        # Verify max-pooling
        expected_max, _ = torch.max(branch_scores, dim=-1)
        assert torch.allclose(max_scores, expected_max, atol=1e-5), \
            "Max-pooling not working correctly"

    print("✓ Max-pooling test passed")


def test_branch_diversity():
    """
    Test that different branches learn different patterns

    While initial weights are random, branches should have SOME variance
    """
    x = torch.randn(8, 64, 256)
    dendritic = DendriticGating(256, num_branches=4)

    with torch.no_grad():
        branch_scores = dendritic.get_branch_activations(x)  # [8, 64, 4]

    # Compute variance across branches for each token
    branch_variance = branch_scores.var(dim=-1).mean().item()

    # Should have some variance (not all branches identical)
    assert branch_variance > 0.001, \
        f"Branches too similar (variance: {branch_variance:.6f}). Expected diversity."

    print(f"✓ Branch diversity test passed (variance: {branch_variance:.4f})")


def test_parameter_count():
    """Test parameter count is reasonable"""
    dim = 256
    num_branches = 4
    hidden_dim = 64

    dendritic = DendriticGating(dim, num_branches, hidden_dim)

    total_params = sum(p.numel() for p in dendritic.parameters())

    # Expected: num_branches * (dim * hidden + hidden * 1 + biases)
    # = 4 * (256*64 + 64*1 + 64 + 1) = 4 * 16,449 = 65,796
    expected_params = num_branches * (dim * hidden_dim + hidden_dim * 1 + hidden_dim + 1)

    assert total_params == expected_params, \
        f"Parameter count mismatch. Expected {expected_params}, got {total_params}"

    # Check overhead on baseline ViT
    baseline_params = 4_766_474  # ViT-Small
    overhead_pct = (total_params / baseline_params) * 100

    assert overhead_pct < 2.0, \
        f"Dendritic overhead too high: {overhead_pct:.2f}%. Expected <2%"

    print(f"✓ Parameter count test passed ({total_params:,} params, {overhead_pct:.2f}% overhead)")


def test_forward_determinism():
    """Test that forward pass is deterministic (for reproducibility)"""
    x = torch.randn(4, 64, 256)
    dendritic = DendriticGating(256, num_branches=4)
    dendritic.eval()

    with torch.no_grad():
        out1 = dendritic(x)
        out2 = dendritic(x)

    assert torch.allclose(out1, out2), \
        "Forward pass not deterministic!"

    print("✓ Determinism test passed")


def test_batch_independence():
    """Test that batch elements are processed independently"""
    x = torch.randn(4, 64, 256)
    dendritic = DendriticGating(256, num_branches=4)

    with torch.no_grad():
        # Full batch
        full_out = dendritic(x)

        # Individual samples
        for i in range(x.size(0)):
            single_out = dendritic(x[i:i+1])
            assert torch.allclose(full_out[i], single_out[0], atol=1e-5), \
                f"Batch element {i} not independent"

    print("✓ Batch independence test passed")


if __name__ == '__main__':
    print("=" * 70)
    print("Running Dendritic Gating Tests")
    print("=" * 70)

    test_dendritic_branch_shape()
    test_dendritic_gating_shape()
    test_gradient_flow()
    test_positive_bias_initialization()
    test_max_pooling_behavior()
    test_branch_diversity()
    test_parameter_count()
    test_forward_determinism()
    test_batch_independence()

    print("=" * 70)
    print("✓ All dendritic tests passed!")
    print("=" * 70)
