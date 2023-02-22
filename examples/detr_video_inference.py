import torch
import cv2
import torchvision.transforms as T
import numpy as np
import time
import argparse
import os

from torch import nn
from torchvision.models import resnet50

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', 
    default='example_test_data/video_1.mp4',
    help='path to input video',
)
parser.add_argument(
    '-d', '--device', 
    default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    help='computation/training device, default is GPU if GPU present'
)
args = parser.parse_args()

np.random.seed(2000)

class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}

detr = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval().to(args.device)

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(640),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

if __name__ == '__main__':
    out_dir = os.path.join('examples', 'results', 'detr_infer') 
    os.makedirs(out_dir, exist_ok=True)  

    cap = cv2.VideoCapture(args.input)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    save_name = args.input.split(os.path.sep)[-1].split('.')[0]
    # Define codec and create VideoWriter object.
    out = cv2.VideoWriter(f"{out_dir}/{save_name}.mp4", 
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (frame_width, frame_height))

    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = transform(rgb_frame).unsqueeze(0)
            
            # Get the start time.
            start_time = time.time()
            with torch.no_grad():
                outputs = detr(rgb_frame.to(args.device))
            forward_end_time = time.time()

            forward_pass_time = forward_end_time - start_time
            
            # Get the current fps.
            fps = 1 / (forward_pass_time)
            # Add `fps` to `total_fps`.
            total_fps += fps
            # Increment frame count.
            frame_count += 1

            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.8
        
            # convert boxes from [0; 1] to image scales
            bboxes_scaled = rescale_bboxes(
                outputs['pred_boxes'][0, keep], 
                (frame.shape[1], frame.shape[0])
            )

            for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], bboxes_scaled.tolist(), COLORS):
                cl = p.argmax()
                class_name = CLASSES[cl]
                color = COLORS[CLASSES.index(class_name)]
                cv2.rectangle(
                    frame,
                    (int(xmin), int(ymin)),
                    (int(xmax), int(ymax)),
                    color=color[::-1],
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    frame, 
                    class_name, 
                    (int(xmin), int(ymin)),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.75, 
                    color=color[::-1], 
                    thickness=2, 
                    lineType=cv2.LINE_AA
                )
            cv2.putText(
                frame, 
                f"FPS: {str(int(fps))}", 
                (25, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.75, 
                color=(0, 0, 255), 
                thickness=1, 
                lineType=cv2.LINE_AA
            )
            out.write(frame)
            cv2.imshow('Image', frame)
            # Press `q` to exit
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
