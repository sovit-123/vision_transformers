from torch import nn


SUPPORTED_ACT_FNS = [
    'relu', 
    'prelu', 
    'relu6', 
    'leaky_relu', 
    'gelu',
    'sigmoid', 
    'hard_sigmoid', 
    'swish', 
    'hard_swish'
]


def get_activation_fn(act_type = 'swish', 
                      num_parameters = -1, 
                      inplace = True,
                      negative_slope = 0.1
                      ):
    if act_type == 'relu':
        return nn.ReLU(inplace=False)
    elif act_type == 'prelu':
        assert num_parameters >= 1
        return nn.PReLU(num_parameters=num_parameters)
    elif act_type == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    elif act_type == 'hard_sigmoid':
        return nn.Hardsigmoid(inplace=inplace)
    elif act_type == 'swish':
        return nn.SiLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'sigmoid':
        return nn.Sigmoid()
    elif act_type == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_type == 'hard_swish':
        return nn.Hardswish(inplace=inplace)
    else:
        raise ValueError(
            'Supported activation layers are: {}. Supplied argument is: {}'.format(SUPPORTED_ACT_FNS, act_type))
