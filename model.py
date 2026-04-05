"""
Vision Transformer (ViT) — built from scratch.

Architecture overview:
    Image → Patch Embedding → [CLS token + Positional Embedding]
          → N × Transformer Encoder Blocks
          → CLS token output → MLP Classification Head → 4 classes
"""

import torch
import torch.nn as nn
import config


# ── 1. Patch Embedding ────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits the image into fixed-size patches and linearly projects each
    patch to a vector of size EMBED_DIM.

    Uses a Conv2d with kernel_size = stride = PATCH_SIZE, which is
    mathematically equivalent to splitting + linear projection but faster.
    """

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=config.EMBED_DIM,
            kernel_size=config.PATCH_SIZE,
            stride=config.PATCH_SIZE,
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.projection(x)          # (B, EMBED_DIM, H/P, W/P)
        x = x.flatten(2)                # (B, EMBED_DIM, num_patches)
        x = x.transpose(1, 2)          # (B, num_patches, EMBED_DIM)
        return x


# ── 2. Multi-Head Self-Attention ──────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads  = config.NUM_HEADS
        self.head_dim   = config.EMBED_DIM // config.NUM_HEADS
        self.scale      = self.head_dim ** -0.5

        self.qkv        = nn.Linear(config.EMBED_DIM, config.EMBED_DIM * 3)
        self.proj       = nn.Linear(config.EMBED_DIM, config.EMBED_DIM)
        self.attn_drop  = nn.Dropout(config.DROPOUT)
        self.proj_drop  = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        B, N, C = x.shape

        # compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ── 3. MLP / Feed-Forward Block ───────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.EMBED_DIM, config.MLP_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.MLP_DIM, config.EMBED_DIM),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


# ── 4. Transformer Encoder Block ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-LN variant: LayerNorm is applied before each sub-layer (more stable
    when training from scratch).
    """

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.EMBED_DIM)
        self.attn  = MultiHeadSelfAttention()
        self.norm2 = nn.LayerNorm(config.EMBED_DIM)
        self.mlp   = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # residual connection
        x = x + self.mlp(self.norm2(x))    # residual connection
        return x


# ── 5. Vision Transformer ─────────────────────────────────────────────────────

class VisionTransformer(nn.Module):
    """
    Full ViT pipeline:
        patch_embed → prepend CLS token → add positional embedding
        → transformer blocks → norm → CLS token → classifier head
    """

    def __init__(self):
        super().__init__()

        self.patch_embed = PatchEmbedding()

        # learnable [CLS] token — one vector prepended to the sequence
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, config.EMBED_DIM))

        # learnable positional embeddings (num_patches + 1 for CLS)
        self.pos_embed   = nn.Parameter(
            torch.zeros(1, config.NUM_PATCHES + 1, config.EMBED_DIM)
        )
        self.pos_drop    = nn.Dropout(config.DROPOUT)

        # stack of transformer blocks
        self.blocks      = nn.Sequential(
            *[TransformerBlock() for _ in range(config.DEPTH)]
        )

        self.norm        = nn.LayerNorm(config.EMBED_DIM)

        # classification head
        self.head        = nn.Linear(config.EMBED_DIM, config.NUM_CLASSES)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)                                    # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                          # (B, N+1, D)
        x   = self.pos_drop(x + self.pos_embed)

        x   = self.blocks(x)                                       # (B, N+1, D)
        x   = self.norm(x)

        cls_out = x[:, 0]                                          # (B, D)
        return self.head(cls_out)                                  # (B, num_classes)


def build_model():
    model = VisionTransformer()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ViT built from scratch | Trainable params: {total_params:,}")
    return model
