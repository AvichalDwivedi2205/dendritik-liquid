"""
Dendritic Vision Transformer (Dendritic-ViT)

Ablation model: ViT with dendritic gating (no sparsity, no CfC yet)

This model computes token importance scores using dendritic branches,
but still attends to ALL tokens. Importance scores are logged for analysis.

Purpose:
- Ablation study: isolate contribution of dendritic gating
- Verify dendritic module learns meaningful importance patterns
- Baseline for later sparse attention

Expected Performance:
- Accuracy: ~95.5% (slight improvement over vanilla ViT)
- FLOPs: ~98% of baseline (minimal overhead from dendritic branches)
- Interpretability: Can visualize which patches are important
"""
import torch
import torch.nn as nn
from einops import repeat

from src.models.vit_base import PatchEmbedding, MultiHeadSelfAttention, FeedForward
from src.models.dendritic_module import DendriticGating


class DendriticTransformerBlock(nn.Module):
    """
    Transformer block with dendritic gating

    Flow:
    x → DendriticGating → importance_scores
    x → LayerNorm → Attention → + → LayerNorm → MLP → + → out
    ↓____________________________|   ↓_______________|

    Importance scores are computed but NOT used for sparsity (yet)
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        dropout=0.,
        num_dendritic_branches=4
    ):
        super().__init__()

        # Dendritic gating (computes importance)
        self.dendritic = DendriticGating(dim, num_branches=num_dendritic_branches)

        # Standard transformer components
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, hidden_dim, dropout)

    def forward(self, x):
        """
        Forward pass with dendritic importance computation

        Args:
            x: [batch, num_patches+1, dim] (includes CLS token)

        Returns:
            x: [batch, num_patches+1, dim] - output features
            importance: [batch, num_patches+1] - token importance scores
        """
        # Compute token importance
        importance = self.dendritic(x)  # [B, N]

        # Standard attention (all tokens, no sparsity)
        x = x + self.attn(self.norm1(x))

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x, importance


class DendriticViT(nn.Module):
    """
    Vision Transformer with Dendritic Gating (Ablation Model)

    Differences from baseline ViT:
    - Each transformer block has dendritic gating
    - Computes importance scores for each token
    - Does NOT apply sparsity (all tokens still attend)
    - Can extract importance maps for visualization

    Args:
        Same as baseline ViT + num_dendritic_branches
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.,
        dropout=0.1,
        num_dendritic_branches=4
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks with dendritic gating
        self.blocks = nn.ModuleList([
            DendriticTransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                dropout,
                num_dendritic_branches
            )
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_importance=False):
        """
        Forward pass

        Args:
            x: [batch, channels, height, width] - input images
            return_importance: If True, return importance maps

        Returns:
            logits: [batch, num_classes] - classification logits
            importance_maps: (optional) [batch, depth, num_patches+1] - importance per layer
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        importance_maps = []
        for block in self.blocks:
            x, importance = block(x)
            importance_maps.append(importance)

        # Final layer norm
        x = self.norm(x)

        # Classification (use CLS token)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        if return_importance:
            importance_maps = torch.stack(importance_maps, dim=1)  # [B, depth, N]
            return logits, importance_maps

        return logits

    def get_importance_statistics(self, x):
        """
        Analyze importance patterns across layers

        Returns dict with:
        - mean_importance_per_layer: [depth]
        - std_importance_per_layer: [depth]
        - top_k_patches: indices of most important patches
        """
        logits, importance = self.forward(x, return_importance=True)

        stats = {
            'mean_per_layer': importance.mean(dim=(0, 2)).cpu(),  # [depth]
            'std_per_layer': importance.std(dim=(0, 2)).cpu(),    # [depth]
            'mean_overall': importance.mean().item(),
            'min_overall': importance.min().item(),
            'max_overall': importance.max().item()
        }

        # Top-k most important patches (averaged across layers)
        avg_importance = importance.mean(dim=1)  # [B, N]
        top_k = 10
        _, top_indices = torch.topk(avg_importance, k=top_k, dim=1)
        stats['top_k_indices'] = top_indices.cpu()

        return stats


def create_dendritic_vit_tiny(num_classes=10, num_dendritic_branches=4):
    """Create Dendritic-ViT-Tiny"""
    return DendriticViT(
        img_size=32,
        patch_size=4,
        num_classes=num_classes,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.,
        dropout=0.1,
        num_dendritic_branches=num_dendritic_branches
    )


def create_dendritic_vit_small(num_classes=10, num_dendritic_branches=4):
    """Create Dendritic-ViT-Small"""
    return DendriticViT(
        img_size=32,
        patch_size=4,
        num_classes=num_classes,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.,
        dropout=0.1,
        num_dendritic_branches=num_dendritic_branches
    )


def create_dendritic_vit_base(num_classes=10, num_dendritic_branches=4):
    """Create Dendritic-ViT-Base"""
    return DendriticViT(
        img_size=32,
        patch_size=4,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        dropout=0.1,
        num_dendritic_branches=num_dendritic_branches
    )


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Dendritic-ViT")
    print("=" * 70)

    # Create model
    model = create_dendritic_vit_small(num_classes=10)
    x = torch.randn(4, 3, 32, 32)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    logits = model(x)
    print(f"Output shape: {logits.shape}")

    # Forward with importance maps
    logits, importance = model(x, return_importance=True)
    print(f"\nImportance maps shape: {importance.shape}")
    print(f"Importance range: [{importance.min():.3f}, {importance.max():.3f}]")
    print(f"Mean importance: {importance.mean():.3f}")

    # Get statistics
    stats = model.get_importance_statistics(x)
    print(f"\nImportance Statistics:")
    print(f"  Overall mean: {stats['mean_overall']:.3f}")
    print(f"  Overall range: [{stats['min_overall']:.3f}, {stats['max_overall']:.3f}]")
    print(f"  Per-layer means: {stats['mean_per_layer'].tolist()}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    dendritic_params = sum(
        p.numel() for n, p in model.named_parameters()
        if 'dendritic' in n
    )
    baseline_params = 4_766_474  # ViT-Small

    print(f"\nParameter Comparison:")
    print(f"  Baseline ViT: {baseline_params:,}")
    print(f"  Dendritic-ViT: {total_params:,}")
    print(f"  Dendritic overhead: {dendritic_params:,} ({dendritic_params/total_params*100:.2f}%)")
    print(f"  Total overhead: {(total_params-baseline_params)/baseline_params*100:.2f}%")

    # Test gradient flow
    print(f"\nTesting gradient flow...")
    x_grad = torch.randn(2, 3, 32, 32, requires_grad=True)
    logits = model(x_grad)
    loss = logits.sum()
    loss.backward()

    assert x_grad.grad is not None
    assert not torch.isnan(x_grad.grad).any()
    print("  ✓ Gradients flowing correctly")

    print("\n" + "=" * 70)
    print("✓ Dendritic-ViT test passed!")
    print("=" * 70)
