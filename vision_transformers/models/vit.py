import torch
import torch.nn as nn

from .layers.patches import CreatePatches
from .layers.transformer import Transformer
from ..utils import load_weights

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

        nn.init.trunc_normal_(self.pos_embedding, std=0.2)
        nn.init.trunc_normal_(self.cls_token, std=0.2)

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

        self.ln = nn.LayerNorm(embed_dim, eps=1e-06)
        self.latent = nn.Identity()

        self.mlp_head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patches(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x = self.dropout(x)

        x = self.transformer(x)
        x = self.ln(x)
        x = self.latent(x)
        x = x[:, 0]
        return self.mlp_head(x)

def vit_b_p16_224(
    image_size=224,
    num_classes=1000,
    pretrained=False
):
    """
    ViT-B/16 from https://arxiv.org/pdf/2010.11929v2.pdf.
    Patch size = 16.
    """
    name = 'vit_b_p16_224'
    model = ViT(
        img_size=224, 
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        mlp_in=768,
        mlp_ratio=4,
        mlp_out=768,
        depth=12,
        num_heads=12,
        drop_rate=0.0,
        emb_drop_rate=0.0
    )
    if pretrained:
        print(f"Loading {name} pretrained weights")
        model = load_weights.load_pretrained_state_dict(model, name)
    
    model.mlp_head = nn.Linear(
        in_features=model.mlp_head.in_features, 
        out_features=num_classes, 
        bias=True
    )
    return model

def vit_b_p32_224(
    image_size=224,
    num_classes=1000,
    pretrained=False
):
    """
    ViT-B/32 from https://arxiv.org/pdf/2010.11929v2.pdf.
    Patch size  = 32.
    """
    name = 'vit_b_p32_224'
    model =  ViT(
        img_size=224, 
        patch_size=32,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        mlp_in=768,
        mlp_ratio=4,
        mlp_out=768,
        depth=12,
        num_heads=12,
        drop_rate=0.0,
        emb_drop_rate=0.0
    )
    if pretrained:
        print(f"Loading {name} pretrained weights")
        model = load_weights.load_pretrained_state_dict(model, name)

    model.mlp_head = nn.Linear(
        in_features=model.mlp_head.in_features, 
        out_features=num_classes, 
        bias=True
    )
    return model

def vit_ti_p16_224(
    image_size=224,
    num_classes=1000,
    pretrained=False
):
    name = 'vit_ti_p16_224'
    model = ViT(
        img_size=224, 
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=192,
        mlp_in=192,
        mlp_ratio=4,
        mlp_out=192,
        depth=12,
        num_heads=3,
        drop_rate=0.0,
        emb_drop_rate=0.0
    )
    if pretrained:
        print(f"Loading {name} pretrained weights")
        model = load_weights.load_pretrained_state_dict(model, name)
    model.mlp_head = nn.Linear(
        in_features=model.mlp_head.in_features, 
        out_features=num_classes, 
        bias=True
    )
    return model

def vit_ti_p16_384(
    image_size=384,
    num_classes=1000,
    pretrained=False
):
    name = 'vit_ti_p16_384'
    model = ViT(
        img_size=384, 
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=192,
        mlp_in=192,
        mlp_ratio=4,
        mlp_out=192,
        depth=12,
        num_heads=3,
        drop_rate=0.0,
        emb_drop_rate=0.0
    )
    print(model)
    if pretrained:
        print(f"Loading {name} pretrained weights")
        model = load_weights.load_pretrained_state_dict(model, name)

    model.mlp_head = nn.Linear(
        in_features=model.mlp_head.in_features, 
        out_features=num_classes, 
        bias=True
    )
    return model