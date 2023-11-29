"""
USAGE:

python tools/inference_video_detect.py --model detr_resnet50 --show --input example_test_data/video_1.mp4

# Tracking.
python tools/inference_video_detect.py --model detr_resnet50 --show --input example_test_data/video_1.mp4 --track

# Selecting classes (person).
python tools/inference_video_detect.py --model detr_resnet50 --show --input example_test_data/video_1.mp4 --classes 2

# Selecting classes (person and bicycle).
python tools/inference_video_detect.py --model detr_resnet50 --show --input example_test_data/video_1.mp4 --classes 2 3
"""

import torch
import cv2
import numpy as np
import argparse
import yaml
import os
import time
import torchinfo

from vision_transformers.detection.detr.model import DETRModel
from utils.detection.detr.general import (
    set_infer_dir,
    load_weights
)
from utils.detection.detr.transforms import infer_transforms, resize
from utils.detection.detr.annotations import (
    convert_detections,
    inference_annotations, 
    annotate_fps,
    convert_pre_track,
    convert_post_track
)
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.detection.detr.viz_attention import visualize_attention

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
        '--imgsz', 
        '--img-size', 
        default=640,
        dest='imgsz',
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
    parser.add_argument(
        '--track',
        action='store_true'
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        default=None,
        help='filter classes by visualization, --classes 1 2 3'
    )
    parser.add_argument(
        '--viz-atten',
        dest='vis_atten',
        action='store_true',
        help='visualize attention map of detected boxes'
    )
    args = parser.parse_args()
    return args

def read_return_video_data(video_path):
    cap = cv2.VideoCapture(video_path)
    # Get the video's frame width and height.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    assert (frame_width != 0 and frame_height !=0), 'Please check video path...'
    return cap, frame_width, frame_height, fps

def main(args):
    if args.track: # Initialize Deep SORT tracker if tracker is selected.
        tracker = DeepSort(max_age=30)

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

    model, CLASSES, data_path = load_weights(
        args, 
        DEVICE, 
        DETRModel, 
        data_configs, 
        NUM_CLASSES, 
        CLASSES, 
        video=True
    )
    _ = model.to(DEVICE).eval()
    try:
        torchinfo.summary(
            model, 
            device=DEVICE, 
            input_size=(1, 3, args.imgsz, args.imgsz), 
            row_settings=["var_names"]
        )
    except:
        print(model)
        # Total parameters and trainable parameters.
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")

    # Colors for visualization.
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    VIDEO_PATH = args.input
    if VIDEO_PATH == None:
        VIDEO_PATH = data_path

    cap, frame_width, frame_height, fps = read_return_video_data(VIDEO_PATH)

    save_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0]
    # Define codec and create VideoWriter object.
    out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4", 
                        cv2.VideoWriter_fourcc(*'mp4v'), fps, 
                        (frame_width, frame_height))
    if args.imgsz != None:
        RESIZE_TO = args.imgsz
    else:
        RESIZE_TO = frame_width

    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            orig_frame = frame.copy()
            frame = resize(frame, RESIZE_TO, square=True)
            image = frame.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = infer_transforms(image)
            image = torch.tensor(image, dtype=torch.float32)
            image = torch.permute(image, (2, 0, 1))
            image = image.unsqueeze(0)
            
            # Get the start time.
            start_time = time.time()
            with torch.no_grad():
                outputs = model(image.to(args.device))
            forward_end_time = time.time()

            forward_pass_time = forward_end_time - start_time
            
            # Get the current fps.
            fps = 1 / (forward_pass_time)
            # Add `fps` to `total_fps`.
            total_fps += fps
            # Increment frame count.
            frame_count += 1

            if args.vis_atten:
                visualize_attention(
                    model,
                    image, 
                    args.threshold, 
                    orig_frame,
                    f"{OUT_DIR}/frame_{str(frame_count)}.png",
                    DEVICE
                )

            if len(outputs['pred_boxes'][0]) != 0:
                draw_boxes, pred_classes, scores = convert_detections(
                    outputs, 
                    args.threshold,
                    CLASSES,
                    orig_frame,
                    args 
                )
                if args.track:
                    tracker_inputs = convert_pre_track(
                        draw_boxes, pred_classes, scores
                    )
                    # Update tracker with detections.
                    tracks = tracker.update_tracks(
                        tracker_inputs, frame=frame
                    )
                    draw_boxes, pred_classes, scores = convert_post_track(tracks) 
                orig_frame = inference_annotations(
                    draw_boxes,
                    pred_classes,
                    scores,
                    CLASSES,
                    COLORS,
                    orig_frame,
                    args
                )
            orig_frame = annotate_fps(orig_frame, fps)
            out.write(orig_frame)
            if args.show:
                cv2.imshow('Prediction', orig_frame)
                # Press `q` to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        else:
            break
    if args.show:
        # Release VideoCapture().
        cap.release()
        # Close all frames and video windows.
        cv2.destroyAllWindows()

    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

if __name__ == '__main__':
    args = parse_opt()
    main(args)