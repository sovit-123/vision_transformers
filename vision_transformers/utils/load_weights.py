from torch.hub import load_state_dict_from_url

urls = {
    'vit_b_p16_224': 'https://download.pytorch.org/models/vit_b_16-c867db91.pth',
    # 'vit_b_p16_224': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
    'vit_b_p32_224': 'https://download.pytorch.org/models/vit_b_32-d86f8d99.pth',
    'vit_ti_p16_224': 'https://www.dropbox.com/s/rtdzmmnod5vzc3o/vit_tiny_patch16_224.pth?dl=1',
    'vit_ti_p16_384': 'https://www.dropbox.com/s/0848g7suf3v2h0w/vit_tiny_patch16_384.pth?dl=1'
}

def load_pretrained_state_dict(model, model_name='vit_b_p16_224'):
    weights = load_state_dict_from_url(urls[model_name])
    # Model's current state dictionary.
    state_dict = model.state_dict()

    # NOTE: This loads torchvision weights.
    if model_name == 'vit_b_p16_224' or model_name == 'vit_b_p32_224':
        print('Loading Torchvision pretrained weights')
        state_dict['cls_token'] = weights['class_token']
        state_dict['pos_embedding'] = weights['encoder.pos_embedding']
        state_dict['patches.patch.weight'] = weights['conv_proj.weight']
        state_dict['patches.patch.bias'] = weights['conv_proj.bias']
        
        for i in range(12):
            state_dict[f"transformer.layers.{i}.0.norm.weight"] = weights[f"encoder.layers.encoder_layer_{i}.ln_1.weight"]
            state_dict[f"transformer.layers.{i}.0.norm.bias"] = weights[f"encoder.layers.encoder_layer_{i}.ln_1.bias"]
            state_dict[f"transformer.layers.{i}.0.fn.qkv.weight"] = weights[f"encoder.layers.encoder_layer_{i}.self_attention.in_proj_weight"]
            state_dict[f"transformer.layers.{i}.0.fn.qkv.bias"] = weights[f"encoder.layers.encoder_layer_{i}.self_attention.in_proj_bias"]
            state_dict[f"transformer.layers.{i}.0.fn.out.0.weight"] = weights[f"encoder.layers.encoder_layer_{i}.self_attention.out_proj.weight"]
            state_dict[f"transformer.layers.{i}.0.fn.out.0.bias"] = weights[f"encoder.layers.encoder_layer_{i}.self_attention.out_proj.bias"]
            state_dict[f"transformer.layers.{i}.1.norm.weight"] = weights[f"encoder.layers.encoder_layer_{i}.ln_2.weight"]
            state_dict[f"transformer.layers.{i}.1.norm.bias"] = weights[f"encoder.layers.encoder_layer_{i}.ln_2.bias"]
            state_dict[f"transformer.layers.{i}.1.fn.mlp_net.0.weight"] = weights[f"encoder.layers.encoder_layer_{i}.mlp.linear_1.weight"]
            state_dict[f"transformer.layers.{i}.1.fn.mlp_net.0.bias"] = weights[f"encoder.layers.encoder_layer_{i}.mlp.linear_1.bias"]
            state_dict[f"transformer.layers.{i}.1.fn.mlp_net.3.weight"] = weights[f"encoder.layers.encoder_layer_{i}.mlp.linear_2.weight"]
            state_dict[f"transformer.layers.{i}.1.fn.mlp_net.3.bias"] = weights[f"encoder.layers.encoder_layer_{i}.mlp.linear_2.bias"]
            
        state_dict['ln.weight'] = weights['encoder.ln.weight']
        state_dict['ln.bias'] = weights['encoder.ln.bias']
        state_dict['mlp_head.weight'] = weights['heads.head.weight']
        state_dict['mlp_head.bias'] = weights['heads.head.bias']
        model.load_state_dict(state_dict)
        return model

    # NOTE: This loads timm weights.
    weights = load_state_dict_from_url(urls[model_name])
    # Model's current state dictionary.
    state_dict = model.state_dict()

    if model_name == 'vit_ti_p16_224' or model_name == 'vit_ti_p16_384':
        print('Loading timm weights')
        state_dict['cls_token'] = weights['cls_token']
        state_dict['pos_embedding'] = weights['pos_embed']
        state_dict['patches.patch.weight'] = weights['patch_embed.proj.weight']
        state_dict['patches.patch.bias'] = weights['patch_embed.proj.bias']
        
        for i in range(12):
            state_dict[f"transformer.layers.{i}.0.norm.weight"] = weights[f"blocks.{i}.norm1.weight"]
            state_dict[f"transformer.layers.{i}.0.norm.bias"] = weights[f"blocks.{i}.norm1.bias"]
            state_dict[f"transformer.layers.{i}.0.fn.qkv.weight"] = weights[f"blocks.{i}.attn.qkv.weight"]
            state_dict[f"transformer.layers.{i}.0.fn.qkv.bias"] = weights[f"blocks.{i}.attn.qkv.bias"]
            state_dict[f"transformer.layers.{i}.0.fn.out.0.weight"] = weights[f"blocks.{i}.attn.proj.weight"]
            state_dict[f"transformer.layers.{i}.0.fn.out.0.bias"] = weights[f"blocks.{i}.attn.proj.bias"]
            state_dict[f"transformer.layers.{i}.1.norm.weight"] = weights[f"blocks.{i}.norm2.weight"]
            state_dict[f"transformer.layers.{i}.1.norm.bias"] = weights[f"blocks.{i}.norm2.bias"]
            state_dict[f"transformer.layers.{i}.1.fn.mlp_net.0.weight"] = weights[f"blocks.{i}.mlp.fc1.weight"]
            state_dict[f"transformer.layers.{i}.1.fn.mlp_net.0.bias"] = weights[f"blocks.{i}.mlp.fc1.bias"]
            state_dict[f"transformer.layers.{i}.1.fn.mlp_net.3.weight"] = weights[f"blocks.{i}.mlp.fc2.weight"]
            state_dict[f"transformer.layers.{i}.1.fn.mlp_net.3.bias"] = weights[f"blocks.{i}.mlp.fc2.bias"]
            
        state_dict['ln.weight'] = weights['norm.weight']
        state_dict['ln.bias'] = weights['norm.bias']
        # MAYBE no need to load head weights.
        state_dict['mlp_head.weight'] = weights['head.weight']
        state_dict['mlp_head.bias'] = weights['head.bias']
        model.load_state_dict(state_dict)
        return model