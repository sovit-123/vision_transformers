import torch.nn as nn
import math

SUPPORTED_NORM_FNS = [
    'batch_norm_2d', 
    'batch_norm_1d', 
    'sync_batch_norm', 
    'group_norm',
    'instance_norm_2d', 
    'instance_norm_1d',
    'layer_norm',
    'identity'
    ]

norm_layers_tuple = (
    nn.BatchNorm1d, 
    nn.BatchNorm2d, 
    nn.SyncBatchNorm, 
    nn.LayerNorm, 
    nn.InstanceNorm1d, 
    nn.InstanceNorm2d, 
    nn.GroupNorm
    )

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
    
class Identity(nn.Module):
    def __init__(self):
        """
            Identity operator
        """
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
def get_normalization_layer(opts, num_features, norm_type = None, num_groups = None):

    norm_type = getattr(opts, "model.normalization.name", "batch_norm_2d") if norm_type is None else norm_type
    num_groups = getattr(opts, "model.normalization.groups", 1) if num_groups is None else num_groups
    momentum = getattr(opts, "model.normalization.momentum", 0.1)

    norm_layer = None
    norm_type = norm_type.lower() if norm_type is not None else None
    if norm_type == 'batch_norm_2d':
        norm_layer = nn.BatchNorm2d(num_features=num_features, momentum=momentum)
    elif norm_type == 'batch_norm_1d':
        norm_layer = nn.BatchNorm1d(num_features=num_features, momentum=momentum)
    elif norm_type == 'sync_batch_norm':
        norm_layer = nn.SyncBatchNorm(num_features=num_features, momentum=momentum)
    elif norm_type == 'group_norm':
        num_groups = math.gcd(num_features, num_groups)
        norm_layer = nn.GroupNorm(num_channels=num_features, num_groups=num_groups)
    elif norm_type == 'instance_norm_2d':
        norm_layer = nn.InstanceNorm2d(num_features=num_features, momentum=momentum)
    elif norm_type == 'instance_norm_1d':
        norm_layer = nn.InstanceNorm1d(num_features=num_features, momentum=momentum)
    elif norm_type == 'layer_norm':
        norm_layer = nn.LayerNorm(num_features)
    elif norm_type == 'identity':
        norm_layer = Identity()
    else:
        raise ValueError(
            'Supported normalization layer arguments are: {}. Got: {}'.format(SUPPORTED_NORM_FNS, norm_type))
    return norm_layer