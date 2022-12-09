import torch.nn as nn

class Normalization(nn.Module):
    """ 
    The Normalization layer that is to be there before every
    Attention block and MLP block.
    Figure 1:
    https://arxiv.org/pdf/2010.11929v2.pdf
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-06)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)