from torch.hub import load_state_dict_from_url

urls = {
    'vit_b_16': 'https://download.pytorch.org/models/vit_b_16-c867db91.pth',
    'vit_b_32': 'https://download.pytorch.org/models/vit_b_32-d86f8d99.pth'
}

def load_pretrained_state_dict(model, model_name='vit_b_16'):
    weights = load_state_dict_from_url(urls[model_name])
    # Model's current state dictionary.
    state_dict = model.state_dict()

    if model_name == 'vit_b_16' or model_name == 'vit_b_32':
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
        state_dict['mlp_head.0.weight'] = weights['heads.head.weight']
        state_dict['mlp_head.0.bias'] = weights['heads.head.bias']
    model.load_state_dict(state_dict)
    return model