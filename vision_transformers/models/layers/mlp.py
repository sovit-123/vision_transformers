import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        dropout=0.
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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x