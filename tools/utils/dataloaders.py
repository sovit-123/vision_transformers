import torch

from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from utils.transforms import get_train_transform, get_valid_transform

def get_dataloaders(
    train_dir=None, 
    valid_dir=None, 
    image_size=224,
    batch_size=32,
    num_workers=4,
    data_dir=None,
    valid_split=None
):
    """
    Paths provided should be in PyTorch `ImageFolder` format.
    See: https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html

    :param image_size: Image size to resize to. Passing integer (224), 
        will make the final image 224x224.
    :param valid_dir: Path to training directory,
    :param valid_dir: Path to validation directory,
    :param batch_size: Batch size for data loader.
    :param num_workers: Parallel workers for data processing.
    :param data_dir: Path to dataset when train/valid split not present.
    :param valid_split: Validation split percentange when proving `data_dir`.

    Returns:
        dataset_train: The training dataset.
        dataset_valid: The validation dataset.
        train_dataloader: Training data loader.
        valid_dataloader: Validation data loader.
        dataset_classes: The class names.
    """
    if data_dir == None: # If train/valid path provided.
        dataset_train = datasets.ImageFolder(
            train_dir,
            transform=(get_train_transform(image_size))
        )    
        dataset_valid = datasets.ImageFolder(
            valid_dir, 
            transform=(get_valid_transform(image_size))
        )
        dataset_classes = dataset_train.classes
    else: # If single data dir provided as train/valid split not present.
        dataset = datasets.ImageFolder(
            data_dir, 
            transform=(get_train_transform(image_size))
        )
        dataset_temp = datasets.ImageFolder(
            data_dir,
            transform=(get_train_transform(image_size))
        )
        dataset_size = len(dataset)
        # Calculate validation dataset size.
        valid_size = int(valid_split*dataset_size)
        # Randomize data indices.
        indices = torch.randperm(len(dataset)).tolist()
        # Final training and validation sets.
        dataset_train = Subset(dataset, indices[:-valid_size])
        dataset_valid = Subset(dataset_temp, indices[-valid_size:])
        dataset_classes = dataset.classes

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
        dataset_classes
    ) 