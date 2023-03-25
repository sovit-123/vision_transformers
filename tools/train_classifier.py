import torch
import argparse
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils.dataloaders import get_dataloaders
from utils.general import (
    SaveBestModel, 
    save_model,
    set_training_dir,
    save_loss_plot,
    save_accuracy_plot
)
from utils.load_model import create_model
from utils.logging import set_log, log
from vision_transformers.models import vit

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    dest='model',
    default='vit_b_p16_224'
)
parser.add_argument(
    '--train-dir', 
    dest='train_dir',
    help='path to training data directory'
)
parser.add_argument(
    '--valid-dir', 
    dest='valid_dir',
    help='path to validation data directory'
)
parser.add_argument(
    '--data-dir', 
    dest='data_dir',
    nargs='+',
    help='path to data directory if training and validation split not present \
          need to pass the validation percentage as well \
          e.g. --data-dir path/to/data 0.15'
)
parser.add_argument(
    '-e', '--epochs', 
    type=int, 
    default=10,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-lr', '--learning-rate', 
    type=float,
    dest='learning_rate', 
    default=0.001,
    help='Learning rate for training the model'
)
parser.add_argument(
    '-b', '--batch',
    default=32,
    type=int,
    help='batch size for data loader'
)
parser.add_argument(
    '-n', '--name',
    default=None,
    type=str,
    help='set result dir name in runs/training/, (default res#)'
)
args = parser.parse_args()

# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    log('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, testloader, criterion, class_names):
    model.eval()
    log('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    OUT_DIR = set_training_dir(args.name)
    set_log(OUT_DIR)

    if args.data_dir == None:
        # Load the training and validation datasets.
        dataset_train, \
            dataset_valid, \
            train_loader, \
            valid_loader, dataset_classes = get_dataloaders(
                train_dir=args.train_dir,
                valid_dir=args.valid_dir,
                batch_size=args.batch,
                image_size=224
            )
    else:
        dataset_train, \
            dataset_valid, \
            train_loader, \
            valid_loader, dataset_classes = get_dataloaders(
                data_dir=args.data_dir[0],
                valid_split=float(args.data_dir[1]),
                batch_size=args.batch,
                image_size=224
            )
    log(f"[INFO]: Number of training images: {len(dataset_train)}")
    log(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    log(f"[INFO]: Classes: {dataset_classes}")
    # Load the training and validation data loaders.

    # Learning_parameters. 
    lr = args.learning_rate
    epochs = args.epochs
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Computation device: {device}")
    log(f"Learning rate: {lr}")
    log(f"Epochs to train for: {epochs}\n")

    # Load the model.
    build_model = create_model[args.model]
    model = build_model(
        image_size=224, num_classes=len(dataset_classes), pretrained=True
    )
    _ = model.to(device)
    log(model)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    log(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    log(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Initialize SaveBestModel class
    save_best_model = SaveBestModel()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # if torch.__version__ >= '2.0.0':
    #     model = torch.compile(model)

    # Start the training.
    for epoch in range(epochs):
        log(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                    criterion, dataset_classes)
        # Save the best model till now.
        save_best_model(
            model, valid_epoch_loss, epoch, OUT_DIR
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        # Save loss and accuracy plots.
        save_loss_plot(OUT_DIR, train_loss, valid_loss)
        save_accuracy_plot(OUT_DIR, train_acc, valid_acc)
        save_model(OUT_DIR, epoch, model, optimizer, criterion)
        log(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        log(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        log('-'*50)
        
    log('TRAINING COMPLETE')