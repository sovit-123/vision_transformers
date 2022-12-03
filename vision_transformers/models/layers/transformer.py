from .norm import Normalization
from .attention import Attention
from .mlp import MLP

import torch.nn as nn

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