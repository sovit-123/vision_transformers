import torch
import torch.nn as nn

from .layers.patches import CreatePatches
from .layers.transformer import Transformer
from ..utils.params import params

# Default values for ViT_B_16 from the paper
# https://arxiv.org/pdf/2010.11929v2.pdf
"""
IMG_SIZE = 224
NUM_CLASSES = 1000
EMBED_DIM = 768
MLP_RATIO = 4 # Muliplies with `EMBED_DIM` to give hidden_dim (3072 for ViT_B_16).
CHANNELS = 3
PATCH_SIZE = 16
NUM_HEADS = 12
MLP_IN_FEATURES = 768 # Equals to `EMBED_DIM`.
MLP_OUT_FEATURES = 768 # # Equals to `EMBED_DIM`.
TRANSFORMER_DEPTH = 12
DROPOUT = 0.0
EMB_DROPOUT = 0.0
DIM_HEAD = 64
"""

class ViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        mlp_in=768,
        mlp_ratio=4,
        mlp_out=768,
        depth=12,
        num_heads=12,
        dim_head=64,
        drop_rate=0.0,
        emb_drop_rate=0.0
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size//patch_size) ** 2

        # Image patches.
        self.patches = CreatePatches(in_channels, embed_dim, self.patch_size)

        # Postional encoding.
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.dropout = nn.Dropout(emb_drop_rate)
        self.mlp_hidden = mlp_in * mlp_ratio

        self.transformer = Transformer(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=drop_rate,
            mlp_in=mlp_in,
            mlp_hidden=self.mlp_hidden,
            mlp_out=mlp_out
        )

        self.latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patches(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x = self.dropout(x)

        x = self.transformer(x)

        x = x[:, 0]

        x = self.latent(x)
        return self.mlp_head(x)

def vit_b_16(
    num_classes=1000
):
    return ViT(
        img_size=224, 
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=768,
        mlp_in=768,
        mlp_ratio=4,
        mlp_out=768,
        depth=12,
        num_heads=12,
        drop_rate=0.0,
        emb_drop_rate=0.0
    )

def vit_b_32(
    num_classes=1000
):
    return ViT(
        img_size=224, 
        patch_size=32,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=768,
        mlp_in=768,
        mlp_ratio=4,
        mlp_out=768,
        depth=12,
        num_heads=12,
        drop_rate=0.0,
        emb_drop_rate=0.0
    )