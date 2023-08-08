"""
Run evaluation on a trained model to get mAP and class wise AP.

USAGE:
python tools/eval.py --model detr_resnet50 --weights runs/training/eval_trial/best_model.pth --data data/aquarium.yaml --imgsz 640
"""
from utils.detection.datasets import (
    create_valid_dataset, create_valid_loader
)
from vision_transformers.detection.detr.model import DETRModel
from utils.detection.metrics.coco_eval import CocoEvaluator
from utils.detection.metrics.panoptic_eval import PanopticEvaluator
from utils.detection.metrics.coco_utils import get_coco_api_from_dataset
from utils.detection.detr.matcher import HungarianMatcher
from utils.detection.detr.detr import SetCriterion, PostProcess
from torch.utils.data import SequentialSampler

import torch
import argparse
import yaml
import os
import utils.detection.detr.misc as utils
import torchvision

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', 
        default=None,
        required=True,
        help='path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', 
        default='detr_resnet50',
        help='name of the model'
    )
    parser.add_argument(
        '--weights', 
        default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '--imgsz', 
        default=640, 
        type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-j', '--workers', 
        default=4, 
        type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '--batch', 
        default=8, 
        type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='show class-wise mAP'
    )
    parser.add_argument(
        '-st', 
        '--square-training',
        dest='square_training',
        action='store_true',
        help='Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.'
    )
    parser.add_argument(
        '--eos_coef',
        default=0.1,
        type=float,
        help='relative classification weight of the no-object class'
    )
    args = parser.parse_args()

    # Load the data configurations
    with open(args.data) as file:
        data_configs = yaml.safe_load(file)

    # Validation settings and constants.
    try: # Use test images if present.
        VALID_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    except: # Else use the validation images.
        VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    NUM_WORKERS = args.workers
    DEVICE = args.device
    BATCH_SIZE = args.batch

    # Model configurations
    IMAGE_SIZE = args.imgsz

    # Load weights.
    model = DETRModel(num_classes=NUM_CLASSES, model=args.model)
    checkpoint = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    valid_dataset = create_valid_dataset(
        VALID_DIR_IMAGES, 
        VALID_DIR_LABELS, 
        IMAGE_SIZE, 
        CLASSES,
        square_training=args.square_training
    )
    valid_sampler = SequentialSampler(valid_dataset)
    valid_loader = create_valid_loader(
        valid_dataset, BATCH_SIZE, NUM_WORKERS, batch_sampler=valid_sampler
    )

    def _get_iou_types(model):
        model_without_ddp = model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = model.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types

    @torch.no_grad()
    def evaluate(
        model, 
        criterion, 
        postprocessors, 
        data_loader, 
        device, 
        output_dir
    ):
        model.eval()

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'

        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = _get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        panoptic_evaluator = None
        if 'panoptic' in postprocessors.keys():
            panoptic_evaluator = PanopticEvaluator(
                data_loader.dataset.ann_file,
                data_loader.dataset.ann_folder,
                output_dir=os.path.join(output_dir, "panoptic_eval"),
            )
        
        print_freq = 100
        counter = 0
        for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
            counter += 1
            samples = list(image.to(device) for image in samples)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
    
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            if 'segm' in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(res)

            if panoptic_evaluator is not None:
                res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
                for i, target in enumerate(targets):
                    image_id = target["image_id"].item()
                    file_name = f"{image_id:012d}.png"
                    res_pano[i]["image_id"] = image_id
                    res_pano[i]["file_name"] = file_name

                panoptic_evaluator.update(res_pano)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        if coco_evaluator is not None:
            coco_evaluator.synchronize_between_processes()
        if panoptic_evaluator is not None:
            panoptic_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        if coco_evaluator is not None:
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
        panoptic_res = None
        if panoptic_evaluator is not None:
            panoptic_res = panoptic_evaluator.summarize()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if coco_evaluator is not None:
            if 'bbox' in postprocessors.keys():
                stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            if 'segm' in postprocessors.keys():
                stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        if panoptic_res is not None:
            stats['PQ_all'] = panoptic_res["All"]
            stats['PQ_th'] = panoptic_res["Things"]
            stats['PQ_st'] = panoptic_res["Stuff"]
        return stats, coco_evaluator
    
    matcher = HungarianMatcher(cost_giou=2,cost_class=1,cost_bbox=5)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(
        NUM_CLASSES-1, 
        matcher, 
        weight_dict, 
        eos_coef=args.eos_coef, 
        losses=losses
    )
    criterion = criterion.to(DEVICE)
    stats, coco_evaluator = evaluate(
        model=model,
        criterion=criterion,
        postprocessors={'bbox': PostProcess()},
        data_loader=valid_loader,
        device=DEVICE, 
        output_dir='outputs'
    )