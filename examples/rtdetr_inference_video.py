# from vision_transformers.detection.rtdetr.src.zoo.rtdetr_model import load_model
from vision_transformers.detection.rtdetr import load_model
from torchvision import transforms

import cv2
import time
import torch
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    help='model name',
    default='rtdetr_resnet50'
)
parser.add_argument(
    '--input',
    help='path to the input video',
    default='../example_test_data/video_1.mp4'
)
parser.add_argument(
    '--device',
    help='cpu or cuda',
    default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
)
args = parser.parse_args()

np.random.seed(42)

OUT_DIR = os.path.join('results', 'rt_detr')
os.makedirs(OUT_DIR, exist_ok=True)

device = args.device

mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}

COLORS = np.random.uniform(0, 255, size=(len(mscoco_category2name), 3))

# Load model.
model = load_model(args.model)

print(model)

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

model.eval().to(device)

# Load video and read data.
cap = cv2.VideoCapture(args.input)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
vid_fps = int(cap.get(5))
file_name = args.input.split(os.path.sep)[-1].split('.')[0]
out = cv2.VideoWriter(
    f"{OUT_DIR}/{file_name}.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), vid_fps, 
    (frame_width, frame_height)
)

# Inference transforms.
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

frame_count = 0
total_fps = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model(image)
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        total_fps += fps

        for i in range(len(outputs['pred_boxes'][0])):
            logits = outputs['pred_logits'][0][i]
            soft_logits = torch.softmax(logits, dim=-1)
            max_index = torch.argmax(soft_logits).cpu()
            class_name = mscoco_category2name[mscoco_label2category[int(max_index.numpy())]]
            
            if soft_logits[max_index] > 0.50:
                box = outputs['pred_boxes'][0][i].cpu().numpy()
                cx, cy, w, h = box
                cx = cx * frame.shape[1]
                cy = cy * frame.shape[0]
                w = w * frame.shape[1]
                h = h * frame.shape[0]
                
                x1 = int(cx - (w//2))
                y1 = int(cy - (h//2))
                x2 = int(x1 + w)
                y2 = int(y1 + h)

                color = COLORS[max_index]
                
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    thickness=2,
                    color=color,
                    lineType=cv2.LINE_AA
                )
                
                cv2.putText(
                    frame,
                    text=class_name,
                    org=(x1, y1-10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    thickness=2,
                    color=color,
                    lineType=cv2.LINE_AA
                )
        cv2.putText(
            frame,
            text=f"FPS: {fps:.1f}",
            org=(15, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            thickness=2,
            color=(0, 255, 0),
            lineType=cv2.LINE_AA
        )
        out.write(frame)
        cv2.imshow('Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()

# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")