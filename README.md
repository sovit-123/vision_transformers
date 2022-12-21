# vision_transformers



***A repository for everything Vision Transformers.***

## Currently Supported Models

* **Image Classification**:
  * ViT Base Patch 16 | 224x224: Torchvision pretrained weights
  * ViT Base Patch 32 | 224x224: Torchvision pretrained weights
  * ViT Tiny Patch 16 | 224x224: Timm pretrained weights

## Quick Setup

```
git clone https://github.com/sovit-123/vision_transformers.git
```

```
cd vision_transformers
```

Installation in the environment of your choice (***PyPi package coming soon***):

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

## [Example Notebooks](https://github.com/sovit-123/vision_transformers/tree/main/example_notebooks)

* [ViT Base 16 | 224x224 pretrained fine-tuning on CIFAR10](https://github.com/sovit-123/vision_transformers/blob/main/example_notebooks/cifar10_vit_pretrained.ipynb)