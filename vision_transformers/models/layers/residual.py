import torch.nn as nn

from .conv_bn_act import ConvBlock
from .helpers import make_divisible

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class InvertedResidual(nn.Module):
    """
        Inverted residual block (MobileNetv2)
    """
    def __init__(self,
                 opts,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 dilation = 1
                 ):
        """
            :param opts: arguments
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: move the kernel by this amount during convolution operation
            :param expand_ratio: expand ratio for hidden dimension
            :param dilation: add zeros between kernel elements to increase the effective receptive field of the kernel.
        """
        assert stride in [1, 2]
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(name="exp_1x1",
                             module=ConvBlock(opts, in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                                              use_act=True, use_norm=True))

        block.add_module(
            name="conv_3x3",
            module=ConvBlock(opts, in_channels=hidden_dim, out_channels=hidden_dim, stride=stride, kernel_size=3,
                             groups=hidden_dim, use_act=True, use_norm=True, dilation=dilation)
        )

        block.add_module(name="red_1x1",
                         module=ConvBlock(opts, in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                                          use_act=False, use_norm=True))

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)