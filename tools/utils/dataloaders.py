from torchvision import datasets
from torch.utils.data import DataLoader
from utils.transforms import get_train_transform, get_valid_transform

def get_dataloaders(
    train_dir, 
    valid_dir, 
    image_size,
    batch_size=32,
    num_workers=4
):
    """
    Paths provided should be in PyTorch `ImageFolder` format.
    See: https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html

    :param image_size: Image size to resize to. Passing integer (224), 
        will make the final image
    """
    dataset_train = datasets.ImageFolder(
        train_dir,
        transform=(get_train_transform(image_size))
    )    
    dataset_valid = datasets.ImageFolder(
        valid_dir, 
        transform=(get_valid_transform(image_size))
    )

    train_dataloader = DataLoader(
        dataset_train, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    return (
        dataset_train, 
        dataset_valid, 
        train_dataloader, 
        valid_dataloader,
        dataset_train.classes
    ) 