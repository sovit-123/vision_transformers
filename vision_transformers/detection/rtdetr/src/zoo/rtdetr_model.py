"""by lyuwenyu
"""

import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 

from ..core import register
from .hybrid_encoder import HybridEncoder
from .rtdetr_decoder import RTDETRTransformer
from ..nn.backbone.presnet import PResNet

__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = PResNet(
            depth=50,
            variant='d', 
            num_stages=4, 
            return_idx=[1, 2, 3], 
            act='relu',
            freeze_at=-1, 
            freeze_norm=True, 
            pretrained=True
        )
        self.decoder = RTDETRTransformer(
            feat_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            num_levels=3,

            num_queries=300,

            num_decoder_layers=6,
            num_denoising=100,
            
            eval_idx=-1,
            eval_spatial_size=[640, 640]
        )
        self.encoder = HybridEncoder(
            in_channels=[512, 1024, 2048],
            feat_strides=[8, 16, 32],
            # intra
            hidden_dim=256,
            use_encoder_idx=[2],
            num_encoder_layers=1,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.,
            enc_act='gelu',
            pe_temperature=10000,
            # cross
            expansion=1.0,
            depth_mult=1,
            act='silu',
            # eval
            eval_spatial_size=[640, 640]
        )
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