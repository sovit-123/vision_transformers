import torch
import argparse
import numpy as np
import os
import torchinfo
import yaml

from utils.detection.datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
from utils.detection.utils.engine import train, evaluate
from vision_transformers.detection.detr.model import DETRModel
from utils.detection.utils.matcher import HungarianMatcher
from utils.detection.utils.detr import SetCriterion, PostProcess
from torch.utils.data import (
    distributed, RandomSampler, SequentialSampler
)
from utils.detection.utils.general import (
    SaveBestModel,
    init_seeds,
    set_training_dir,
    save_model_state,
    save_mAP,
    show_tranformed_image
)
from utils.detection.utils.logging import set_log, coco_log

RANK = int(os.getenv('RANK', -1))

np.random.seed(42)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', 
        '--epochs',
        default=10,
        type=int
    )
    parser.add_argument(
        '--model', 
        default='detr_resnet50',
        help='name of the model',
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='path to the data config file'
    )
    parser.add_argument(
        '-d', 
        '--device', 
        default='cuda',
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '--name', 
        default=None, 
        type=str, 
        help='training result dir name in runs/training/, (default res_#)'
    )
    parser.add_argument(
        '-ims', '--img-size',
        dest='img_size', 
        default=640, 
        type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '--batch', 
        default=4, 
        type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '-j', '--workers', 
        default=4,
        type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-st', '--square-training',
        dest='square_training',
        action='store_true',
        help='Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.'
    )
    parser.add_argument(
        '-uta', '--use-train-aug', 
        dest='use_train_aug', 
        action='store_true',
        help='whether to use train augmentation, uses some advanced \
            augmentation that may make training difficult when used \
            with mosaic'
    )
    parser.add_argument(
        '-nm', '--no-mosaic', 
        dest='no_mosaic', 
        action='store_true',
        help='pass this to not to use mosaic augmentation'
    )
    parser.add_argument(
        '-vt', '--vis-transformed', 
        dest='vis_transformed', 
        action='store_true',
        help='visualize transformed images fed to the network'
    )
    parser.add_argument(
        '-lr', 
        '--learning-rate',
        dest='learning_rate',
        type=float,
        default=5e-5
    )
    parser.add_argument(
        '-lrb',
        '--lr-backbone',
        dest='lr_backbone',
        type=float,
        default=1e-6
    )
    parser.add_argument(
        '--weight-decay',
        dest='weight_decay',
        default=1e-4,
        type=float
    )
    parser.add_argument(
        '--eos_coef',
        default=0.1,
        type=float,
        help='relative classification weight of the no-object class'
    )
    parser.add_argument(
        '--no-lrs', 
        dest='no_lrs',
        action='store_true',
        help='pass this to not use learning rate scheduler'
    )
    parser.add_argument(
        '--seed',
        default=0,
        type=int ,
        help='global seed for training'
    )
    args = parser.parse_args()
    return args

def main(args):
    # Load the data configurations
    with open(args.data) as file:
        data_configs = yaml.safe_load(file)

    init_seeds(args.seed + 1 + RANK, deterministic=True)
    
    # Settings/parameters/constants.
    TRAIN_DIR_IMAGES = os.path.normpath(data_configs['TRAIN_DIR_IMAGES'])
    TRAIN_DIR_LABELS = os.path.normpath(data_configs['TRAIN_DIR_LABELS'])
    VALID_DIR_IMAGES = os.path.normpath(data_configs['VALID_DIR_IMAGES'])
    VALID_DIR_LABELS = os.path.normpath(data_configs['VALID_DIR_LABELS'])
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']
    LR = args.learning_rate
    EPOCHS = args.epochs
    DEVICE = args.device
    NUM_CLASSES = len(CLASSES)
    IMAGE_SIZE = args.img_size
    BATCH_SIZE = args.batch
    IS_DISTRIBUTED = False
    NUM_WORKERS = args.workers
    VISUALIZE_TRANSFORMED_IMAGES = args.vis_transformed
    OUT_DIR = set_training_dir(args.name)
    COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))
    set_log(OUT_DIR)
    train_dataset = create_train_dataset(
        TRAIN_DIR_IMAGES, 
        TRAIN_DIR_LABELS,
        IMAGE_SIZE, 
        CLASSES,
        use_train_aug=args.use_train_aug,
        no_mosaic=True,
        square_training=True
    )

    valid_dataset = create_valid_dataset(
        VALID_DIR_IMAGES, 
        VALID_DIR_LABELS, 
        IMAGE_SIZE, 
        CLASSES,
        square_training=True
    )

    if IS_DISTRIBUTED:
        train_sampler = distributed.DistributedSampler(
            train_dataset
        )
        valid_sampler = distributed.DistributedSampler(
            valid_dataset, shuffle=False
        )
    else:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)

    # train_batch_sampler = BatchSampler(train_sampler, BATCH_SIZE, drop_last=False)
    # valid_batch_sampler = BatchSampler(valid_sampler, BATCH_SIZE, drop_last=False)
    train_loader = create_train_loader(
        train_dataset, BATCH_SIZE, NUM_WORKERS, batch_sampler=train_sampler
    )
    valid_loader = create_valid_loader(
        valid_dataset, BATCH_SIZE, NUM_WORKERS, batch_sampler=valid_sampler
    )

    if VISUALIZE_TRANSFORMED_IMAGES:
        show_tranformed_image(train_loader, DEVICE, CLASSES, COLORS)

    matcher = HungarianMatcher(cost_giou=2,cost_class=1,cost_bbox=5)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes', 'cardinality']
    model = DETRModel(num_classes=NUM_CLASSES, model=args.model)
    model = model.to(DEVICE)
    try:
        torchinfo.summary(
            model, device=DEVICE, input_size=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        )
    except:
        print(model)
        # Total parameters and trainable parameters.
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")

    criterion = SetCriterion(
        NUM_CLASSES-1, 
        matcher, 
        weight_dict, 
        eos_coef=args.eos_coef, 
        losses=losses
    )
    criterion = criterion.to(DEVICE)

    # TODO Check how this works when with model params differently in model.py
    lr_dict = {
        'backbone': 0.1,
        'transformer': 1,
        'embed': 1,
        'final': 5
    }
    optimizer = torch.optim.AdamW([{
        'params': v,
        'lr': lr_dict.get(k,1)*LR
    } for k,v in model.parameter_groups().items()], 
        weight_decay=args.weight_decay
    )
    
    save_best_model = SaveBestModel()

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [EPOCHS // 2, EPOCHS // 1.333], gamma=0.5
    )

    val_map_05 = []
    val_map = []

    # if torch.__version__ >= '2.0.0':
    #     model = torch.compile(model)

    for epoch in range(EPOCHS):
        train_loss = train(
            train_loader, 
            model, 
            criterion, 
            optimizer, 
            DEVICE, 
            epoch=epoch
        )
        if not args.no_lrs:
            lr_scheduler.step()
        stats, coco_evaluator = evaluate(
            model=model,
            criterion=criterion,
            postprocessors={'bbox': PostProcess()},
            data_loader=valid_loader,
            device=DEVICE, 
            output_dir='outputs'
        )
        
        # COCO log to train log file.
        coco_log(OUT_DIR, stats)

        val_map_05.append(stats['coco_eval_bbox'][1])
        val_map.append(stats['coco_eval_bbox'][0])

        # Save mAP plots.
        save_mAP(OUT_DIR, val_map_05, val_map)
        # Save the model dictionary only for the current epoch.
        save_model_state(model, OUT_DIR, data_configs, args.model)
        save_best_model(
            model, 
            val_map[-1], 
            epoch, 
            OUT_DIR,
            data_configs,
            args.model
        )

if __name__ == '__main__':
    args = parse_opt()
    main(args)