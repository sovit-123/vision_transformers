import torch.nn as nn

from .norm import get_normalization_layer
from .activation import get_activation_fn

class ConvBlock(nn.Module):
    """
        2D Convolution block with normalization and activation layer
    """
    def __init__(self, 
                 opts, 
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 stride = 1,
                 dilation = 1, 
                 groups = 1,
                 bias = False, 
                 padding_mode = 'zeros',
                 use_norm = True, 
                 use_act = True
                 ):
        """
            :param opts: arguments
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param kernel_size: kernel size
            :param stride: move the kernel by this amount during convolution operation
            :param dilation: add zeros between kernel elements to increase the effective receptive field of the kernel.
            :param groups: number of groups. If groups=in_channels=out_channels, then it is a depth-wise convolution
            :param bias: use bias or not
            :param padding_mode: padding mode
            :param use_norm: use normalization layer after convolution layer or not
            :param use_act: Use activation layer or not
        """
        super(ConvBlock, self).__init__()

        if use_norm:
            assert not bias, 'Do not use bias when using normalization layers.'

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, (tuple, list))

        padding = (int((kernel_size[0] - 1) / 2) * dilation[0], int((kernel_size[1] - 1) / 2) * dilation[1])

        if in_channels % groups != 0:
            raise ValueError('Input channels are not divisible by groups. {}%{} != 0 '.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('Output channels are not divisible by groups. {}%{} != 0 '.format(out_channels, groups))

        block = nn.Sequential()

        conv_layer = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=kernel_size,
                               stride=stride, 
                               padding=padding, 
                               dilation=dilation, 
                               groups=groups, bias=bias,
                               padding_mode=padding_mode)

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            norm_layer = get_normalization_layer(opts=opts, num_features=out_channels)
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        act_type = getattr(opts, "model.activation.name", "prelu")

        if act_type is not None and use_act:
            neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
            inplace = getattr(opts, "model.activation.inplace", False)
            act_layer = get_activation_fn(act_type=act_type,
                                          inplace=inplace,
                                          negative_slope=neg_slope,
                                          num_parameters=out_channels)
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

    def forward(self, x):
        return self.block(x)