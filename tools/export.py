"""
Export to ONNX.

Requirements:
pip install onnx onnxruntime

USAGE:
python tools/export.py --weights runs/training/onnx_trial/best_model.pth --out detr_resnet50.onnx
"""

import torch
import argparse
import yaml
import os

from vision_transformers.detection.detr.model import DETRModel
from utils.detection.detr.general import load_weights

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', '--weights', 
        default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '--out',
        help='output model name, e.g. model.onnx',
        required=True, 
        type=str
    )
    parser.add_argument(
        '--width',
        default=640,
        type=int,
        help='onnx model input width'
    )
    parser.add_argument(
        '--height',
        default=640,
        type=int,
        help='onnx model input height'
    )
    args = parser.parse_args()
    return args

def main(args):
    NUM_CLASSES = None
    CLASSES = None
    OUT_DIR = 'weights'
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    # Load the data configurations.
    data_configs = None
    if args.data is not None:
        with open(args.data) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']
    DEVICE = args.device
    model, CLASSES, data_path = load_weights(
        args, 
        DEVICE, 
        DETRModel, 
        data_configs, 
        NUM_CLASSES, 
        CLASSES, 
    )
    model.eval()

    # Input to the model
    x = torch.randn(1, 3, args.width, args.height, requires_grad=True)

    # Export the model
    torch.onnx.export(
        model,
        x,
        os.path.join(OUT_DIR, args.out),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input' : {0 : 'batch_size'},
            'output' : {0 : 'batch_size'}
        }
    )
    print(f"Model saved to {os.path.join(OUT_DIR, args.out)}")

if __name__ == '__main__':
    args = parse_opt()
    main(args)