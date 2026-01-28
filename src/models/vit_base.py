"""
Vision Transformer (ViT) - Baseline Implementation

Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
Dosovitskiy et al., ICLR 2021

This is the baseline model against which DendriticLiquid-ViT will be compared.
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """
    Convert image to sequence of patches and embed them.

    For CIFAR-10 (32x32 images):
    - patch_size=4 → 8x8 patches = 64 patches total
    - patch_size=8 → 4x4 patches = 16 patches total
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Use Conv2d for patch embedding (equivalent to linear projection of flattened patches)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )

    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            patches: [batch, num_patches, embed_dim]
        """
        return self.proj(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, dim, num_heads=8, dropout=0.):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, num_patches, dim]
        Returns:
            out: [batch, num_patches, dim]
        """
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        out = self.proj(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network (MLP)"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block

    Architecture:
    x → LayerNorm → MultiHeadAttention → + → LayerNorm → FeedForward → + → out
    ↓___________________________________|   ↓__________________________|
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, hidden_dim, dropout)

    def forward(self, x):
        # Attention block with residual
        x = x + self.attn(self.norm1(x))

        # MLP block with residual
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) - Baseline Model

    Args:
        img_size: Input image size (default: 32 for CIFAR-10)
        patch_size: Patch size (default: 4 → 64 patches for 32x32 image)
        in_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension = embed_dim * mlp_ratio
        dropout: Dropout rate
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
        pool='cls'
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token (prepended to sequence)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.pool = pool
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling"""
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            logits: [batch, num_classes]
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
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Classification (use CLS token)
        if self.pool == 'cls':
            cls_token_final = x[:, 0]
        elif self.pool == 'mean':
            cls_token_final = x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pool}")

        logits = self.head(cls_token_final)

        return logits

    def get_attention_maps(self, x):
        """
        Extract attention maps from all layers (for visualization)

        Returns:
            attention_maps: List of [batch, num_heads, num_patches+1, num_patches+1]
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attention_maps = []
        for block in self.blocks:
            # Extract attention before applying it
            qkv = block.attn.qkv(block.norm1(x))
            qkv = qkv.reshape(B, x.shape[1], 3, block.attn.num_heads, block.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = torch.softmax(attn, dim=-1)
            attention_maps.append(attn.detach())

            # Continue forward pass
            x = block(x)

        return attention_maps


def create_vit_tiny(num_classes=10):
    """Create ViT-Tiny model (for quick debugging)"""
    return VisionTransformer(
        img_size=32,
        patch_size=4,
        num_classes=num_classes,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.,
        dropout=0.1
    )


def create_vit_small(num_classes=10):
    """Create ViT-Small model (good balance for CIFAR)"""
    return VisionTransformer(
        img_size=32,
        patch_size=4,
        num_classes=num_classes,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.,
        dropout=0.1
    )


def create_vit_base(num_classes=10):
    """Create ViT-Base model (for best performance)"""
    return VisionTransformer(
        img_size=32,
        patch_size=4,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        dropout=0.1
    )


if __name__ == '__main__':
    # Test the model
    model = create_vit_small(num_classes=10)
    x = torch.randn(4, 3, 32, 32)

    print("=" * 60)
    print("Vision Transformer - Baseline Model")
    print("=" * 60)

    # Forward pass
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test attention maps extraction
    with torch.no_grad():
        attn_maps = model.get_attention_maps(x)
    print(f"\nAttention maps: {len(attn_maps)} layers")
    print(f"Each map shape: {attn_maps[0].shape}")

    print("=" * 60)
    print("✓ Model test passed!")
    print("=" * 60)
