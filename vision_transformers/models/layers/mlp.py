import torch.nn as nn

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