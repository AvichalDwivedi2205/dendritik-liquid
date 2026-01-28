"""
Dendritic Gating Module

Implements multi-compartment dendritic branches that learn token importance scores.

Based on:
"Dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning"
Chavlis, S., Poirazi, P., et al., Nature 2025

Key Features:
- Multiple dendritic branches (K=3-5 typically)
- Each branch computes importance independently
- Max-pooling aggregation: token selected if ANY branch thinks it's important
- Lightweight: ~5% parameter overhead
- Biologically inspired by dendritic computation in neurons
"""
import torch
import torch.nn as nn


class DendriticBranch(nn.Module):
    """
    Single dendritic branch for computing token importance

    A dendritic branch is a small MLP that maps token features to an importance score.
    Each branch can specialize in detecting different types of important features.

    Architecture:
        input (dim) → Linear → GELU → Linear → Sigmoid → importance (0-1)
    """
    def __init__(self, dim, hidden_dim=64):
        """
        Args:
            dim: Input dimension (token embedding dimension)
            hidden_dim: Hidden layer dimension (default: 64)
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # CRITICAL: Initialize with positive bias
        # This ensures all tokens start with HIGH importance
        # Model then learns to prune unimportant ones
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with positive bias"""
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)

        nn.init.xavier_uniform_(self.net[2].weight)
        # Positive bias → sigmoid(2.0) ≈ 0.88 → high initial importance
        self.net[2].bias.data.fill_(2.0)

    def forward(self, x):
        """
        Compute importance score for each token

        Args:
            x: [batch, num_patches, dim] - token features

        Returns:
            importance: [batch, num_patches, 1] - importance scores in [0, 1]
        """
        return self.net(x)


class DendriticGating(nn.Module):
    """
    Multi-branch dendritic gating module

    Creates K dendritic branches that independently compute token importance.
    Final importance is max-pooled across branches: a token is important if
    ANY branch thinks it's important.

    This allows:
    - Different branches to specialize in different features
    - Redundancy: multiple paths to mark tokens as important
    - Biological plausibility: mimics how dendritic branches work in real neurons

    Example:
        Branch 1: detects edges
        Branch 2: detects textures
        Branch 3: detects colors
        → Token with strong edge gets high importance from Branch 1
        → Token with strong texture gets high importance from Branch 2
        → Max-pooling ensures both are kept
    """
    def __init__(self, dim, num_branches=4, hidden_dim=64):
        """
        Args:
            dim: Input dimension (token embedding dimension)
            num_branches: Number of dendritic branches (default: 4)
            hidden_dim: Hidden dimension for each branch (default: 64)
        """
        super().__init__()
        self.num_branches = num_branches
        self.dim = dim

        # Create multiple branches
        self.branches = nn.ModuleList([
            DendriticBranch(dim, hidden_dim) for _ in range(num_branches)
        ])

    def forward(self, x):
        """
        Compute token importance using max-pooling over branches

        Args:
            x: [batch, num_patches, dim] - token features

        Returns:
            importance: [batch, num_patches] - importance scores in [0, 1]
        """
        # Compute score from each branch
        branch_scores = [branch(x) for branch in self.branches]  # Each: [B, N, 1]
        branch_scores = torch.cat(branch_scores, dim=-1)  # [B, N, num_branches]

        # Max-pooling: token is important if ANY branch thinks so
        importance, _ = torch.max(branch_scores, dim=-1)  # [B, N]

        return importance

    def get_branch_activations(self, x):
        """
        Get individual branch scores for visualization

        This helps understand what each branch is learning:
        - Are branches specializing?
        - Do different branches respond to different features?
        - Is max-pooling working as expected?

        Args:
            x: [batch, num_patches, dim]

        Returns:
            branch_scores: [batch, num_patches, num_branches]
        """
        branch_scores = [branch(x) for branch in self.branches]
        return torch.cat(branch_scores, dim=-1)

    def extra_repr(self):
        """Extra information for print(model)"""
        return f'dim={self.dim}, num_branches={self.num_branches}'


class DendriticMLP(nn.Module):
    """
    Alternative: Dendritic MLP with multiplicative gating

    Instead of separate branches, uses multiplicative gates within a single MLP.
    This is computationally cheaper but less biologically plausible.

    NOT USED in main model, but provided for experimentation.
    """
    def __init__(self, dim, hidden_dim, num_gates=4, dropout=0.):
        super().__init__()
        self.num_gates = num_gates

        # Main MLP
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

        # Dendritic gates (one per hidden unit group)
        gate_size = hidden_dim // num_gates
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, gate_size),
                nn.Sigmoid()
            )
            for _ in range(num_gates)
        ])

    def forward(self, x):
        """
        Forward with multiplicative dendritic gating

        Args:
            x: [batch, num_patches, dim]

        Returns:
            out: [batch, num_patches, dim]
        """
        # MLP forward
        h = self.act(self.fc1(x))  # [B, N, hidden_dim]

        # Apply dendritic gates
        h_gated = []
        gate_size = h.size(-1) // self.num_gates

        for i, gate in enumerate(self.gates):
            start_idx = i * gate_size
            end_idx = (i + 1) * gate_size

            # Get slice and apply gate
            h_slice = h[:, :, start_idx:end_idx]
            gate_value = gate(x)
            h_gated.append(h_slice * gate_value)

        h = torch.cat(h_gated, dim=-1)
        h = self.dropout(h)
        out = self.fc2(h)

        return out


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Dendritic Gating Module")
    print("=" * 70)

    # Test configuration
    batch_size = 4
    num_patches = 64
    dim = 256
    num_branches = 4

    # Create input
    x = torch.randn(batch_size, num_patches, dim)
    print(f"\nInput shape: {x.shape}")

    # Test single branch
    print("\n1. Testing Single Dendritic Branch:")
    branch = DendriticBranch(dim, hidden_dim=64)
    branch_out = branch(x)
    print(f"   Output shape: {branch_out.shape}")
    print(f"   Output range: [{branch_out.min():.3f}, {branch_out.max():.3f}]")
    print(f"   Mean importance: {branch_out.mean():.3f}")

    # Test multi-branch gating
    print("\n2. Testing Multi-Branch Dendritic Gating:")
    dendritic = DendriticGating(dim, num_branches=num_branches)
    importance = dendritic(x)
    print(f"   Output shape: {importance.shape}")
    print(f"   Output range: [{importance.min():.3f}, {importance.max():.3f}]")
    print(f"   Mean importance: {importance.mean():.3f}")

    # Check that importance is high initially (due to positive bias)
    assert importance.mean() > 0.6, f"Initial importance should be >0.6, got {importance.mean():.3f}"
    print(f"   ✓ Positive bias working (mean > 0.6)")

    # Test gradient flow
    print("\n3. Testing Gradient Flow:")
    x_grad = torch.randn(batch_size, num_patches, dim, requires_grad=True)
    importance_grad = dendritic(x_grad)
    loss = importance_grad.mean()
    loss.backward()

    assert x_grad.grad is not None
    assert not torch.isnan(x_grad.grad).any()
    print(f"   ✓ Gradients flowing correctly")

    # Test branch activations
    print("\n4. Testing Branch Activation Extraction:")
    branch_acts = dendritic.get_branch_activations(x)
    print(f"   Branch activations shape: {branch_acts.shape}")
    print(f"   Per-branch means:")
    for i in range(num_branches):
        mean_act = branch_acts[:, :, i].mean().item()
        print(f"     Branch {i+1}: {mean_act:.3f}")

    # Count parameters
    total_params = sum(p.numel() for p in dendritic.parameters())
    print(f"\n5. Parameter Count:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Per branch: {total_params // num_branches:,}")

    # Test dendritic MLP (alternative)
    print("\n6. Testing Dendritic MLP (Alternative):")
    dendritic_mlp = DendriticMLP(dim, hidden_dim=512, num_gates=4)
    mlp_out = dendritic_mlp(x)
    print(f"   Output shape: {mlp_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in dendritic_mlp.parameters()):,}")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
