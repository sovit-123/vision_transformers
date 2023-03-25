import torch
import cv2
import numpy as np
import argparse
import yaml
import glob
import os
import time

from vision_transformers.detection.detr.model import DETRModel
from utils.detection.utils.general import (
    set_infer_dir,
    load_weights
)
from utils.detection.utils.transforms import infer_transforms, resize
from utils.detection.utils.annotations import inference_annotations

np.random.seed(2023)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', 
        '--weights',
    )
    parser.add_argument(
        '-i', '--input', 
        help='folder path to input input image (one image or a folder path)',
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '--model', 
        default='detr_resnet50',
        help='name of the model'
    )
    parser.add_argument(
        '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-ims', 
        '--img-size', 
        default=640,
        dest='img_size',
        type=int,
        help='resize image to, by default use the original frame/image size'
    )
    parser.add_argument(
        '-t', 
        '--threshold',
        type=float,
        default=0.5,
        help='confidence threshold for visualization'
    )
    parser.add_argument(
        '--name', 
        default=None, 
        type=str, 
        help='training result dir name in outputs/training/, (default res_#)'
    )
    parser.add_argument(
        '--hide-labels',
        dest='hide_labels',
        action='store_true',
        help='do not show labels during on top of bounding boxes'
    )
    parser.add_argument(
        '--show', 
        dest='show', 
        action='store_true',
        help='visualize output only if this argument is passed'
    )
    args = parser.parse_args()
    return args

def collect_all_images(dir_test):
    """
    Function to return a list of image paths.
    :param dir_test: Directory containing images or single image path.
    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images   

def main(args):
    NUM_CLASSES = None
    CLASSES = None
    data_configs = None
    if args.data is not None:
        with open(args.data) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']
    
    DEVICE = args.device
    OUT_DIR = set_infer_dir(args.name)

    model, CLASSES = load_weights(
        args, DEVICE, DETRModel, data_configs, NUM_CLASSES, CLASSES
    )
    _ = model.to(DEVICE).eval()

    # Colors for visualization.
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    DIR_TEST = args.input
    test_images = collect_all_images(DIR_TEST)
    print(f"Test instances: {len(test_images)}")

    # To count the total number of frames iterated through.
    frame_count = 0
    # To keep adding the frames' FPS.
    total_fps = 0
    for image_num in range(len(test_images)):
        image_name = test_images[image_num].split(os.path.sep)[-1].split('.')[0]
        orig_image = cv2.imread(test_images[image_num])
        frame_height, frame_width, _ = orig_image.shape
        if args.img_size != None:
            RESIZE_TO = args.img_size
        else:
            RESIZE_TO = frame_width
        
        image_resized = resize(orig_image, RESIZE_TO, square=True)
        image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = infer_transforms(image)
        input_tensor = torch.tensor(image, dtype=torch.float32)
        input_tensor = torch.permute(input_tensor, (2, 0, 1))
        input_tensor = input_tensor.unsqueeze(0)
        h, w, _ = orig_image.shape

        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor.to(DEVICE))
        end_time = time.time()
        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1

        if len(outputs['pred_boxes'][0]) != 0:
            orig_image = inference_annotations(
                outputs,
                args.threshold,
                CLASSES,
                COLORS,
                orig_image,
                args
            )
            if args.show:
                cv2.imshow('Prediction', orig_image)
                cv2.waitKey(1)
            
        cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
        print(f"Image {image_num+1} done...")
        print('-'*50)

    print('TEST PREDICTIONS COMPLETE')
    if args.show:
        cv2.destroyAllWindows()
        # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

if __name__ == '__main__':
    args = parse_opt()
    main(args)