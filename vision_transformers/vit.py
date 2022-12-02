import torch
import torch.nn as nn

# Default values to initialize the Vision Transformer.
# Pass these values while creating the model.
"""
IMG_SIZE = 256
NUM_CLASSES = 1000
EMBED_DIM = 1024
MLP_RATIO = 2
CHANNELS = 3
PATCH_SIZE = 32
NUM_HEADS = 16
MLP_IN_FEATURES = EMBED_DIM
MLP_OUT_FEATURES = EMBED_DIM
TRANSFORMER_DEPTH = 6
DROPOUT = 0.1
EMB_DROPOUT = 0.1
DIM_HEAD = 64
"""

# Create image patches.
class CreatePatches(nn.Module):
    def __init__(
        self, channels=3, embed_dim=1024, path_size=32
    ):
        super().__init__()
        self.patch = nn.Conv2d(
            in_channels=channels, 
            out_channels=embed_dim, 
            kernel_size=path_size,
            stride=path_size
        )
    def forward(self, x):
        """
        :param x: Image batches [b, c, h, w]
        """
        # Flatten along dim = 2 to main channel dimension.
        patches = self.patch(x).flatten(2).transpose(1, 2)
        return patches

class Attention(nn.Module):
    # No attention dropout by default to keep things simple.
    # PyTorch MultiheadAttention does not use dropout by default either.
    # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    def __init__(
        self, 
        embed_dim=1024, 
        num_heads=16,
        dim_head=64,
        dropout=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        hidden_dim = dim_head * num_heads

        # Scale = 1 / sqrt(d_k).
        # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#Scaled-Dot-Product-Attention
        self.scale = 1. / (embed_dim ** 0.5)
        self.qkv = nn.Linear(
            in_features=embed_dim, out_features=hidden_dim*3, bias=True
        )
        self.out = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        b, n, c = x.shape # b: batch size, n: dimensions, c: channels.
        qkv = self.qkv(x).reshape(
            b, n, 3, self.num_heads, c//self.num_heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        dot = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        return out

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        dropout
    ):
        super().__init__()
        self.mlp_net = nn.Sequential(
            nn.Linear(
                in_features=in_features, out_features=hidden_features
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                in_features=hidden_features, out_features=out_features
            ),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp_net(x)

class Normalization(nn.Module):
    """ 
    The Normalization layer that is to be there before every
    Attention block and MLP block.
    Figure 1:
    https://arxiv.org/pdf/2010.11929v2.pdf
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Transformer(nn.Module):
    def __init__(
        self,
        depth,
        embed_dim,
        num_heads,
        dim_head,
        mlp_in,
        mlp_hidden,
        mlp_out,
        dropout=0.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Normalization(
                    embed_dim, 
                    Attention(
                        embed_dim=embed_dim, 
                        num_heads=num_heads,
                        dim_head=dim_head,
                        dropout=dropout
                    )
                ),
                Normalization(
                    embed_dim,
                    MLP(
                        in_features=mlp_in, 
                        hidden_features=mlp_hidden,
                        out_features=mlp_out,
                        dropout=dropout
                    )
                )
            ]))

    def forward(self, x):
        for attn, mlp in self.layers:
            x = attn(x) + x # `+ x` for Residual.
            x = mlp(x) + x # `+ x` for Residual.
        return x

class Vit(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        num_classes,
        embed_dim,
        mlp_in,
        mlp_ratio,
        mlp_out,
        depth,
        num_heads,
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
        
if __name__ == '__main__':
    img_size = 224
    in_channels = 3
    model = Vit(
        img_size=img_size, 
        patch_size=16,
        in_channels=in_channels,
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
     # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    tensor = torch.randn(32, in_channels, img_size, img_size)
    out = model(tensor)