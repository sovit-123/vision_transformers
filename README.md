# vision_transformers

***A repository for everything Vision Transformers.***

## Currently Supported Models

- Image Classification

  :

  - ViT Base Patch 16 | 224x224: Torchvision pretrained weights
  - ViT Base Patch 32 | 224x224: Torchvision pretrained weights
  - ViT Tiny Patch 16 | 224x224: Timm pretrained weights

## Quick Setup

### Stable PyPi Package

```
pip install vision-transformers==0.0.2
```

### Latest Git Updates

```
git clone https://github.com/sovit-123/vision_transformers.git
cd vision_transformers
```

Installation in the environment of your choice:

```
pip install .
```

## Importing Models

```
from vision_transformers.models import vit

model = vit.vit_b_p16_224(pretrained=True)
# model = vit.vit_b_p32_224(pretrained=True)
# model = vit.vit_ti_p16_224(pretrained=True)
```

## [Examples](https://github.com/sovit-123/vision_transformers/tree/main/examples)

- [ViT Base 16 | 224x224 pretrained fine-tuning on CIFAR10](https://github.com/sovit-123/vision_transformers/blob/main/examples/cifar10_vit_pretrained.ipynb)
- [ViT Tiny 16 | 224x224 pretrained fine-tuning on CIFAR10](https://github.com/sovit-123/vision_transformers/blob/main/examples/cifar10_vit_tiny_p16_224.ipynb)
- [DETR image inference notebook](https://github.com/sovit-123/vision_transformers/blob/main/examples/detr_image_inference.ipynb)
- [DETR video inference script](https://github.com/sovit-123/vision_transformers/blob/main/examples/detr_video_inference.py) (**Fine Tuning Coming Soon**) --- [Check commands here](#DETR-Video-Inference-Commands)

## DETR Video Inference Commands

***All commands to be executed from the root project directory (`vision_transformers`)***

* Using default video:

```
python examples/detr_video_inference.py
```

* Using CPU only:

```
python examples/detr_video_inference.py --device cpu
```

* Using another video file:

```
python examples/detr_video_inference.py --input /path/to/video/file
```

