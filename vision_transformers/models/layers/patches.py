import torch.nn as nn

# Create image patches.
class CreatePatches(nn.Module):
    def __init__(
        self, channels=3, embed_dim=768, patch_size=16
    ):
        super().__init__()
        self.patch = nn.Conv2d(
            in_channels=channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self, x):
        """
        :param x: Image batches [b, c, h, w]
        """
        # Flatten along dim = 2 to maintain channel dimension.
        patches = self.patch(x).flatten(2).transpose(1, 2)
        return patches