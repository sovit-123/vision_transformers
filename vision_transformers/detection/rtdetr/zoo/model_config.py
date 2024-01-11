model_configs = {
    'rtdetr_resnet18': {
        # PResNet configs.
        'backbone': {
            'depth': 18,
            'variant': 'd',
            'num_stages': 4,
            'return_idx': [1, 2, 3],
            'act': 'relu',
            'freeze_at': -1,
            'freeze_norm': True,
            'pretrained': True,
        },
        'encoder': {
            # HybridEncoder configs.
            'in_channels': [128, 256, 512],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'use_encoder_idx': [2],
            'num_encoder_layers': 1,
            'nhead': 8,
            'dim_feedforward': 1024,
            'dropout': 0.,
            'enc_act': 'gelu',
            'pe_temperature': 10000,
            'expansion': 0.5,
            'depth_mult': 1,
            'act': 'silu',
            'eval_spatial_size': [640, 640],
        },
        'decoder': {
            # RTDETRTransformer configs.
            'feat_channels': [256, 256, 256],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'num_levels': 3,
            'num_queries': 300,
            'num_decoder_layers': 3,
            'num_denoising': 100,
            'eval_idx': -1,
            'eval_spatial_size': [640, 640]
        },
        'weights': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth'
    },

    'rtdetr_resnet34': {
        # PResNet configs.
        'backbone': {
            'depth': 34,
            'variant': 'd',
            'num_stages': 4,
            'return_idx': [1, 2, 3],
            'act': 'relu',
            'freeze_at': -1,
            'freeze_norm': False,
            'pretrained': True,
        },
        'encoder': {
            # HybridEncoder configs.
            'in_channels': [128, 256, 512],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'use_encoder_idx': [2],
            'num_encoder_layers': 1,
            'nhead': 8,
            'dim_feedforward': 1024,
            'dropout': 0.,
            'enc_act': 'gelu',
            'pe_temperature': 10000,
            'expansion': 0.5,
            'depth_mult': 1,
            'act': 'silu',
            'eval_spatial_size': [640, 640],
        },
        'decoder': {
            # RTDETRTransformer configs.
            'feat_channels': [256, 256, 256],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'num_levels': 3,
            'num_queries': 300,
            'num_decoder_layers': 4,
            'num_denoising': 100,
            'eval_idx': -1,
            'eval_spatial_size': [640, 640]
        },
        'weights': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth'
    },

    'rtdetr_resnet50': {
        # PResNet configs.
        'backbone': {
            'depth': 50,
            'variant': 'd',
            'num_stages': 4,
            'return_idx': [1, 2, 3],
            'act': 'relu',
            'freeze_at': -1,
            'freeze_norm': True,
            'pretrained': True,
        },
        'encoder': {
            # HybridEncoder configs.
            'in_channels': [512, 1024, 2048],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'use_encoder_idx': [2],
            'num_encoder_layers': 1,
            'nhead': 8,
            'dim_feedforward': 1024,
            'dropout': 0.,
            'enc_act': 'gelu',
            'pe_temperature': 10000,
            'expansion': 1.0,
            'depth_mult': 1,
            'act': 'silu',
            'eval_spatial_size': [640, 640],
        },
        'decoder': {
            # RTDETRTransformer configs.
            'feat_channels': [256, 256, 256],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'num_levels': 3,
            'num_queries': 300,
            'num_decoder_layers': 6,
            'num_denoising': 100,
            'eval_idx': -1,
            'eval_spatial_size': [640, 640]
        },
        'weights': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth'
    },

    'rtdetr_resnet101': {
        # PResNet configs.
        'backbone': {
            'depth': 101,
            'variant': 'd',
            'num_stages': 4,
            'return_idx': [1, 2, 3],
            'act': 'relu',
            'freeze_at': -1,
            'freeze_norm': True,
            'pretrained': True,
        },
        'encoder': {
            # HybridEncoder configs.
            'in_channels': [512, 1024, 2048],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 384,
            'use_encoder_idx': [2],
            'num_encoder_layers': 1,
            'nhead': 8,
            'dim_feedforward': 2048,
            'dropout': 0.,
            'enc_act': 'gelu',
            'pe_temperature': 10000,
            'expansion': 1.0,
            'depth_mult': 1,
            'act': 'silu',
            'eval_spatial_size': [640, 640],
        },
        'decoder': {
            # RTDETRTransformer configs.
            'feat_channels': [384, 384, 384],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'num_levels': 3,
            'num_queries': 300,
            'num_decoder_layers': 6,
            'num_denoising': 100,
            'eval_idx': -1,
            'eval_spatial_size': [640, 640]
        },
        'weights': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth'
    },
}