import torch
import torch.nn as nn

pool_types = ['mean', 'rms', 'abs']

class GlobalPool(nn.Module):
    """
        Global pooling 
    """
    def __init__(self, pool_type='mean', keep_dim=False):
        """
            :param pool_type: Global pool operation type (mean, rms, abs)
            :param keep_dim: Keep dimensions the same as the input or not
        """
        super(GlobalPool, self).__init__()
        if pool_type not in pool_types:
            raise ValueError('Supported pool types are: {}. Got {}'.format(pool_types, pool_type))
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    def _global_pool(self, x):
        assert x.dim() == 4, "Got: {}".format(x.shape)
        if self.pool_type == 'rms':
            x = x ** 2
            x = torch.mean(x, dim=[-2, -1], keepdim=self.keep_dim)
            x = x ** -0.5
        elif self.pool_type == 'abs':
            x = torch.mean(torch.abs(x), dim=[-2, -1], keepdim=self.keep_dim)
        else:
            # default is mean
            # same as AdaptiveAvgPool
            x = torch.mean(x, dim=[-2, -1], keepdim=self.keep_dim)
        return x

    def forward(self, x):
        return self._global_pool(x)