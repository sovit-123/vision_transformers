"""
Modules adapted/borrowed from https://github.com/kennethzhao24/mobilevit_pytorch
for easier loading of official weights.
Official code and weights: https://github.com/apple/ml-cvnets

Put together by Sovit Ranjan Rath.
"""

import torch.nn as nn
import math
import torch
import torch.nn.functional as F

from .layers.residual import InvertedResidual
from .layers.conv_bn_act import ConvBlock
from .layers.norm import get_normalization_layer
from .layers.weight_init import initialize_weights
from .layers.global_pool import GlobalPool
from .layers.activation import get_activation_fn

from argparse import Namespace

class MHSA(nn.Module):
    """
        Multi-head self attention: https://arxiv.org/pdf/1706.03762
    """
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 attn_dropout = 0.0, 
                 bias = True
                 ):
        """
            :param embed_dim: embedding dimension
            :param num_heads: number of attention heads
            :param attn_dropout: attention dropout
            :param bias: use bias or not
        """
        super(MHSA, self).__init__()
        assert embed_dim % num_heads == 0, "Got: embed_dim={} and num_heads={}".format(embed_dim, num_heads)

        self.qkv_proj = nn.Linear(in_features=embed_dim, out_features=3*embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # [B x N x C]
        b_sz, n_patches, _ = x.shape

        # linear projection to qkv
        # [B x N x C] --> [B x N x 3 x h x C]
        qkv = (self.qkv_proj(x).reshape(b_sz, n_patches, 3, self.num_heads, -1))
        # [B x N x 3 x h x C] --> [B x h x 3 x N x C]
        qkv = qkv.transpose(1, 3)
        # [B x h x 3 x N x C] --> [B x h x N x C] x 3
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q * self.scaling
        # [B x h x N x C] --> [B x h x c x N]
        k = k.transpose(2, 3)

        # compute attention score
        # [B x h x N x c] x [B x h x c x N] --> [B x h x N x N]
        attn = torch.matmul(q, k)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [B x h x N x N] x [B x h x N x c] --> [B x h x N x c]
        out = torch.matmul(attn, v)
        # [B x h x N x c] --> [B x N x h x c] --> [B x N x C=ch]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out

class TransformerEncoder(nn.Module):
    """
        Transfomer Encoder
    """
    def __init__(self, 
                 opts, 
                 embed_dim, 
                 ffn_latent_dim, 
                 num_heads = 8, 
                 attn_dropout = 0.0,
                 dropout = 0.1, 
                 ffn_dropout = 0.0,
                 transformer_norm_layer = "layer_norm",
                 ):
        """
            :param opts: arguments
            :param embed_dim: embedding dimension
            :param ffn_latent_dim: latent dimension of feedforward layer
            :param num_heads: Number of attention heads
            :param attn_dropout: attention dropout rate
            :param dropout: dropout rate
            :param ffn_dropout: feedforward dropout rate
            :param transformer_norm_layer: transformer norm layer
        """
        super(TransformerEncoder, self).__init__()

        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim),
            MHSA(
                embed_dim, 
                num_heads, 
                # dim_head=embed_dim//num_heads,
                attn_dropout=attn_dropout
            ),
            nn.Dropout(p=dropout)
        )

        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            self.build_act_layer(opts=opts),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout)
        )

    @staticmethod
    def build_act_layer(opts):
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)
        act_layer = get_activation_fn(act_type=act_type, inplace=inplace, negative_slope=neg_slope,
                                      num_parameters=1)
        return act_layer

    def forward(self, x):
        # Multi-head attention
        x = x + self.pre_norm_mha(x)
        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTBlock(nn.Module):
    """
        MobileViT block: https://arxiv.org/pdf/2110.02178
    """
    def __init__(self, 
                 opts, 
                 in_channels, 
                 transformer_dim, 
                 ffn_dim,
                 n_transformer_blocks = 2,
                 head_dim = 32, 
                 attn_dropout = 0.1,
                 dropout = 0.1, 
                 ffn_dropout = 0.1, 
                 patch_h = 8,
                 patch_w = 8, 
                 transformer_norm_layer = "layer_norm",
                 conv_ksize = 3,
                 dilation = 1, 
                 ):
        """
            :param opts: arguments
            :param in_channels: number of input channels
            :param transformer_dim: dimension of transformer encoder
            :param ffn_dim: dimension of feedforward layer
            :param n_transformer_block: number of transformer blocks
            :param head_dim: transformer head dimension     
            :param attn_dropout: Attention dropout     
            :param dropout: dropout
            :param ffn_dropout: feedforward dropout
            :param patch_h: split patch height size      
            :param patch_w: split patch width size
            :param transformer_norm_layer: transformer norm layer    
            :param conv_ksize: kernel size for convolutional block    
            :param dilation: add zeros between kernel elements to increase the effective receptive field of the kernel.    
        """

        conv_3x3_in = ConvBlock(
            opts=opts, in_channels=in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True, dilation=dilation
        )
        conv_1x1_in = ConvBlock(
            opts=opts, in_channels=in_channels, out_channels=transformer_dim,
            kernel_size=1, stride=1, use_norm=False, use_act=False
        )

        conv_1x1_out = ConvBlock(
            opts=opts, in_channels=transformer_dim, out_channels=in_channels,
            kernel_size=1, stride=1, use_norm=True, use_act=True
        )
        conv_3x3_out = ConvBlock(
            opts=opts, in_channels=2 * in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True
        )
        
        super(MobileViTBlock, self).__init__()
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        ffn_dims = [ffn_dim] * n_transformer_blocks

        global_rep = [
            TransformerEncoder(
                opts=opts, 
                embed_dim=transformer_dim, 
                ffn_latent_dim=ffn_dims[block_idx], 
                num_heads=num_heads,
                attn_dropout=attn_dropout, 
                dropout=dropout, 
                ffn_dropout=ffn_dropout,
                transformer_norm_layer=transformer_norm_layer
            )
            for block_idx in range(n_transformer_blocks)
        ]
        global_rep.append(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=transformer_dim)
        )
        self.global_rep = nn.Sequential(*global_rep)
        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

    def unfolding(self, feature_map):
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w # n_w
        num_patch_h = new_h // patch_h # n_h
        num_patches = num_patch_h * num_patch_w # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches, info_dict):
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, _, _, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map

    def forward(self, x):
        res = x
        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)
        # learn global representations
        patches = self.global_rep(patches)
        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)
        fm = self.fusion(torch.cat((res, fm), dim=1))

        return fm

class MobileViT(nn.Module):
    """
        MobileViT: https://arxiv.org/pdf/2110.02178
    """
    def __init__(self, opts):
        image_channels, input_channels = 3, 16
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout", 0.1)

        # original mobilevit uses swish activation function
        setattr(opts, "model.activation.name", "swish")

        mobilevit_config = get_config(opts=opts)

        super(MobileViT, self).__init__()

        self.dilation = 1
        self.conv_1 = ConvBlock(
                opts=opts, in_channels=image_channels, out_channels=input_channels,
                kernel_size=3, stride=2, use_norm=True, use_act=True
            )

        self.layer_1, self.layer_1_channels = self._make_layer(
            opts=opts, input_channel=input_channels, cfg=mobilevit_config["layer1"]
        )

        self.layer_2, self.layer_2_channels = self._make_layer(
            opts=opts, input_channel=self.layer_1_channels, cfg=mobilevit_config["layer2"]
        )

        self.layer_3, self.layer_3_channels = self._make_layer(
            opts=opts, input_channel=self.layer_2_channels, cfg=mobilevit_config["layer3"]
        )

        self.layer_4, self.layer_4_channels = self._make_layer(
            opts=opts, input_channel=self.layer_3_channels, cfg=mobilevit_config["layer4"],
        )

        self.layer_5, self.layer_5_channels = self._make_layer(
            opts=opts, input_channel=self.layer_4_channels, cfg=mobilevit_config["layer5"], 
        )

        exp_channels = min(mobilevit_config["last_layer_exp_factor"] * self.layer_5_channels, 960)
        self.conv_1x1_exp = ConvBlock(
                opts=opts, in_channels=self.layer_5_channels, out_channels=exp_channels,
                kernel_size=1, stride=1, use_act=True, use_norm=True
            )

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool())
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=classifier_dropout, inplace=True))
        self.classifier.add_module(
            name="fc",
            module=nn.Linear(in_features=exp_channels, out_features=num_classes, bias=True)
        )

        # weight initialization
        self.reset_parameters(opts=opts)

    def reset_parameters(self, opts):
        initialize_weights(opts=opts, modules=self.modules())

    def extract_features(self, x):
        out_dict = {} # consider input size of 224
        x = self.conv_1(x) # 112 x112
        x = self.layer_1(x) # 112 x112
        out_dict["out_l1"] = x  # level-1 feature

        x = self.layer_2(x) # 56 x 56
        out_dict["out_l2"] = x

        x = self.layer_3(x) # 28 x 28
        out_dict["out_l3"] = x

        x = self.layer_4(x) # 14 x 14
        out_dict["out_l4"] = x

        x = self.layer_5(x) # 7 x 7
        out_dict["out_l5"] = x

        if self.conv_1x1_exp is not None:
            x = self.conv_1x1_exp(x) # 7 x 7
            out_dict["out_l5_exp"] = x

        return out_dict, x

    def forward(self, x):
        _, x = self.extract_features(x)
        x = self.classifier(x)
        return x

    def _make_layer(self, opts, input_channel, cfg, dilate = False):
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg,
                dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(opts, input_channel, cfg):
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            input_channel = output_channels
            block.append(layer)
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(self, opts, input_channel, cfg, dilate = False):
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        block.append(
            MobileViTBlock(
                opts=opts,
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=getattr(opts, "model.classification.mit.dropout", 0.1),
                ffn_dropout=getattr(opts, "model.classification.mit.ffn_dropout", 0.0),
                attn_dropout=getattr(opts, "model.classification.mit.attn_dropout", 0.0),
                head_dim=head_dim,
                conv_ksize=getattr(opts, "model.classification.mit.conv_kernel_size", 3)
            )
        )

        return nn.Sequential(*block), input_channel
    
def get_config(opts):
    model_name = getattr(opts, "model_name", 'mobilevit_s')
    return model_cfg[model_name]

mobilevit_s_cfg = {
    "layer1": {
        "out_channels": 32,
        "expand_ratio": 4,
        "num_blocks": 1,
        "stride": 1,
        "block_type": "mv2"
    },
    "layer2": {
        "out_channels": 64,
        "expand_ratio": 4,
        "num_blocks": 3,
        "stride": 2,
        "block_type": "mv2"
    },
    "layer3": {  # 28x28
        "out_channels": 96,
        "transformer_channels": 144,
        "ffn_dim": 288,
        "transformer_blocks": 2,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer4": {  # 14x14
        "out_channels": 128,
        "transformer_channels": 192,
        "ffn_dim": 384,
        "transformer_blocks": 4,
        "patch_h": 2, 
        "patch_w": 2, 
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer5": {  # 7x7
        "out_channels": 160,
        "transformer_channels": 240,
        "ffn_dim": 480,
        "transformer_blocks": 3,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "last_layer_exp_factor": 4
}

mobilevit_xs_cfg = {
    "layer1": {
        "out_channels": 32,
        "expand_ratio": 4,
        "num_blocks": 1,
        "stride": 1,
        "block_type": "mv2"
    },
    "layer2": {
        "out_channels": 48,
        "expand_ratio": 4,
        "num_blocks": 3,
        "stride": 2,
        "block_type": "mv2"
    },
    "layer3": {  # 28x28
        "out_channels": 64,
        "transformer_channels": 96,
        "ffn_dim": 192,
        "transformer_blocks": 2,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer4": {  # 14x14
        "out_channels": 80,
        "transformer_channels": 120,
        "ffn_dim": 240,
        "transformer_blocks": 4,
        "patch_h": 2, 
        "patch_w": 2, 
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer5": {  # 7x7
        "out_channels": 96,
        "transformer_channels": 144,
        "ffn_dim": 288,
        "transformer_blocks": 3,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "last_layer_exp_factor": 4
}

mobilevit_mini_cfg = {
    "layer1": {
        "out_channels": 32,
        "expand_ratio": 4,
        "num_blocks": 1,
        "stride": 1,
        "block_type": "mv2"
    },
    "layer2": {
        "out_channels": 48,
        "expand_ratio": 4,
        "num_blocks": 3,
        "stride": 2,
        "block_type": "mv2"
    },
    "layer3": {  # 28x28
        "out_channels": 64,
        "transformer_channels": 80,
        "ffn_dim": 160,
        "transformer_blocks": 2,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer4": {  # 14x14
        "out_channels": 80,
        "transformer_channels": 96,
        "ffn_dim": 192,
        "transformer_blocks": 4,
        "patch_h": 2, 
        "patch_w": 2, 
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer5": {  # 7x7
        "out_channels": 96,
        "transformer_channels": 128,
        "ffn_dim": 256,
        "transformer_blocks": 3,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 4,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "last_layer_exp_factor": 4
}

mobilevit_xxs_cfg = {
    "layer1": {
        "out_channels": 16,
        "expand_ratio": 2,
        "num_blocks": 1,
        "stride": 1,
        "block_type": "mv2"
    },
    "layer2": {
        "out_channels": 24,
        "expand_ratio": 2,
        "num_blocks": 3,
        "stride": 2,
        "block_type": "mv2"
    },
    "layer3": {  # 28x28
        "out_channels": 48,
        "transformer_channels": 64,
        "ffn_dim": 128,
        "transformer_blocks": 2,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 2,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer4": {  # 14x14
        "out_channels": 64,
        "transformer_channels": 80,
        "ffn_dim": 160,
        "transformer_blocks": 4,
        "patch_h": 2, 
        "patch_w": 2, 
        "stride": 2,
        "mv_expand_ratio": 2,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer5": {  # 7x7
        "out_channels": 80,
        "transformer_channels": 96,
        "ffn_dim": 192,
        "transformer_blocks": 3,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 2,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "last_layer_exp_factor": 4
}

model_cfg = {
    'mobilevit_xxs': mobilevit_xxs_cfg, 
    'mobilevit_mini': mobilevit_mini_cfg,
    'mobilevit_xs': mobilevit_xs_cfg, 
    'mobilevit_s': mobilevit_s_cfg, 
    }

def mobilevit_s(num_classes=1000, pretrained=False):
    model_name = 'mobilevit_s'
    opts = Namespace(model_name = model_name)
    model = MobileViT(opts=opts)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            'https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt'
        )
        model.load_state_dict(ckpt)

    in_features = model.classifier.fc.in_features
    model.classifier.fc = nn.Linear(
        in_features=in_features, out_features=num_classes, bias=True
    )
    return model

def mobilevit_xs(num_classes=1000, pretrained=False):
    model_name = 'mobilevit_xs'
    opts = Namespace(model_name = model_name)
    model = MobileViT(opts=opts)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            'https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt'
        )
        model.load_state_dict(ckpt)
    in_features = model.classifier.fc.in_features
    model.classifier.fc = nn.Linear(
        in_features=in_features, out_features=num_classes, bias=True
    )
    return model

def mobilevit_mini(num_classes=1000, pretrained=False):
    model_name = 'mobilevit_mini'
    opts = Namespace(model_name = model_name)
    model = MobileViT(opts=opts)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            'https://github.com/kennethzhao24/mobilevit_pytorch/blob/main/weights/mobilevit_mini.pt'
        )
        model.load_state_dict(ckpt)
    in_features = model.classifier.fc.in_features
    model.classifier.fc = nn.Linear(
        in_features=in_features, out_features=num_classes, bias=True
    )
    return model

def mobilevit_xxs(num_classes=1000, pretrained=False):
    model_name = 'mobilevit_xxs'
    opts = Namespace(model_name = model_name)
    model = MobileViT(opts=opts)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            'https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt'
        )
        model.load_state_dict(ckpt)
    in_features = model.classifier.fc.in_features
    model.classifier.fc = nn.Linear(
        in_features=in_features, out_features=num_classes, bias=True
    )
    return model