"""
by lyuwenyu
"""

import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 

from ..core import register
from .hybrid_encoder import HybridEncoder
from .rtdetr_decoder import RTDETRTransformer
from ..nn.backbone.presnet import PResNet
from .model_config import model_configs
from torch.hub import load_state_dict_from_url

__all__ = ['RTDETR', ]

@register
class RTDETR(nn.Module):
    # __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(
              self, 
              backbone_configs, 
              decoder_configs,
              encoder_configs,
              multi_scale=None
    ):
        super().__init__()

        self.backbone = PResNet(**backbone_configs)
        self.decoder = RTDETRTransformer(**decoder_configs)
        self.encoder = HybridEncoder(**encoder_configs)
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
    
def load_model(model_name='rtdetr_resnet50', multi_scale=None):
        config = model_configs[model_name]
        backbone_configs = config['backbone']
        encoder_configs = config['encoder']
        decoder_configs = config['decoder']
        weights = config['weights']

        # Initialize model with pretrained backbone
        model = RTDETR(
            backbone_configs=backbone_configs, 
            decoder_configs=decoder_configs, 
            encoder_configs=encoder_configs, 
            multi_scale=multi_scale
        )
        
        # Load weights.
        ckpt = load_state_dict_from_url(weights)
        model.load_state_dict(ckpt['ema']['module'])
        return model