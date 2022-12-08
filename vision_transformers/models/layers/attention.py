import torch.nn as nn
import torch

class Attention(nn.Module):
    # No attention dropout by default to keep things simple.
    # PyTorch MultiheadAttention does not use dropout by default either.
    # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    def __init__(
        self, 
        embed_dim=768, 
        num_heads=16,
        dim_head=64,
        dropout=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        hidden_dim = dim_head * num_heads

        # Scale = 1 / sqrt(d_k).
        # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#Scaled-Dot-Product-Attention
        self.scale = 1 / (dim_head ** 0.5)
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
        out = self.out(out)
        return out