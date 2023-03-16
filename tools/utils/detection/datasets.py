import torch
import cv2
import numpy as np
import os
import glob as glob
import random
import albumentations as A

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
from utils.detection.utils.transforms import (
    get_train_transform, 
    get_valid_transform,
    get_train_aug,
    transform_mosaic,
)

# the dataset class
class DETRDataset(Dataset):
    def __init__(
        self, 
        images_path, 
        labels_path, 
        img_size, 
        classes, 
        transforms=None, 
        use_train_aug=False,
        train=False, 
        no_mosaic=False,
        square_training=False
    ):
        self.transforms = transforms
        self.use_train_aug = use_train_aug
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.classes = classes
        self.train = train
        self.no_mosaic = no_mosaic
        self.square_training = square_training
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []
        
        # get all the image paths in sorted order
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
        self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)
        # Remove all annotations and images when no object is present.
        self.read_and_clean()

    def read_and_clean(self):
        # Discard any images and labels when the XML 
        # file does not contain any object.
        for annot_path in self.all_annot_paths:
            tree = et.parse(annot_path)
            root = tree.getroot()
            object_present = False
            for member in root.findall('object'):
                if member.find('bndbox'):
                    object_present = True
            if object_present == False:
                image_name = annot_path.split(os.path.sep)[-1].split('.xml')[0]
                image_root = self.all_image_paths[0].split(os.path.sep)[:-1]

        # Discard any image file when no annotation file 
        # is not found for the image. 
        for image_name in self.all_images:
            possible_xml_name = os.path.join(self.labels_path, os.path.splitext(image_name)[0]+'.xml')
            if possible_xml_name not in self.all_annot_paths:
                print(f"{possible_xml_name} not found...")
                print(f"Removing {image_name} image")
                # items = [item for item in items if item != element]
                self.all_images = [image_instance for image_instance in self.all_images if image_instance != image_name]

    def resize(self, im, square=False):
        if square:
            im = cv2.resize(im, (self.img_size, self.img_size))
        else:
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
        return im

    def load_image_and_labels(self, index):
        image_name = self.all_images[index]
        image_path = os.path.join(self.images_path, image_name)

        # Read the image.
        image = cv2.imread(image_path)
        # Convert BGR to RGB color format.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = self.resize(image, square=self.square_training)
        image_resized /= 255.0
        
        # Capture the corresponding XML file for getting the annotations.
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.labels_path, annot_filename)

        boxes = []
        orig_boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # Get the height and width of the image.
        image_width = image.shape[1]
        image_height = image.shape[0]
                
        # Box coordinates for xml files are extracted and corrected for image size given.
        for member in root.findall('object'):
            # Map the current object name to `classes` list to get
            # the label index and append to `labels` list.
            labels.append(self.classes.index(member.find('name').text))
            
            # xmin = left corner x-coordinates
            xmin = int(float(member.find('bndbox').find('xmin').text))
            # xmax = right corner x-coordinates
            xmax = int(float(member.find('bndbox').find('xmax').text))
            # ymin = left corner y-coordinates
            ymin = int(float(member.find('bndbox').find('ymin').text))
            # ymax = right corner y-coordinates
            ymax = int(float(member.find('bndbox').find('ymax').text))

            xmin, ymin, xmax, ymax = self.check_image_and_annotation(
                xmin, 
                ymin, 
                xmax, 
                ymax, 
                image_width, 
                image_height, 
                orig_data=True
            )

            orig_boxes.append([xmin, ymin, xmax, ymax])
            
            # Resize the bounding boxes according to the
            # desired `width`, `height`.
            xmin_final = (xmin/image_width)*image_resized.shape[1]
            xmax_final = (xmax/image_width)*image_resized.shape[1]
            ymin_final = (ymin/image_height)*image_resized.shape[0]
            ymax_final = (ymax/image_height)*image_resized.shape[0]

            xmin_final, ymin_final, xmax_final, ymax_final = self.check_image_and_annotation(
                xmin_final, 
                ymin_final, 
                xmax_final, 
                ymax_final, 
                image_resized.shape[1], 
                image_resized.shape[0],
                orig_data=False
            )

            bw = xmax_final - xmin_final
            bh = ymax_final - ymin_final
            # Normalize.
            h, w, _ = image_resized.shape
            final_coords = [xmin_final, ymin_final, bw, bh]
                
            boxes.append(final_coords)
        
        # Bounding box to tensor.
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Area of the bounding boxes.

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return image, image_resized, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height)

    def check_image_and_annotation(
        self, 
        xmin, 
        ymin, 
        xmax, 
        ymax, 
        width, 
        height, 
        orig_data=False
    ):
        """
        Check that all x_max and y_max are not more than the image
        width or height.
        """
        if ymax > height:
            ymax = height
        if xmax > width:
            xmax = width
        if xmax - xmin <= 1.0:
            if orig_data:
                # print(
                    # '\n',
                    # '!!! xmax is equal to xmin in data annotations !!!'
                    # 'Please check data'
                # )
                # print(
                    # 'Increasing xmax by 1 pixel to continue training for now...',
                    # 'THIS WILL ONLY BE LOGGED ONCE',
                    # '\n'
                # )
                self.log_annot_issue_x = False
            xmin = xmin - 1
        if ymax - ymin <= 1.0:
            if orig_data:
                # print(
                #     '\n',
                #     '!!! ymax is equal to ymin in data annotations !!!',
                #     'Please check data'
                # )
                # print(
                #     'Increasing ymax by 1 pixel to continue training for now...',
                #     'THIS WILL ONLY BE LOGGED ONCE',
                #     '\n'
                # )
                self.log_annot_issue_y = False
            ymin = ymin - 1
        return xmin, ymin, xmax, ymax


    def load_cutmix_image_and_boxes(self, index, resize_factor=512):
        """ 
        Adapted from: https://www.kaggle.com/shonenkov/oof-evaluation-mixup-efficientdet
        """
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.all_images) - 1) for _ in range(3)]

        # Create empty image with the above resized image.
        # result_image = np.full((h, w, 3), 1, dtype=np.float32)
        result_boxes = []
        result_classes = []

        for i, index in enumerate(indices):
            _, image_resized, orig_boxes, boxes, \
            labels, area, iscrowd, dims = self.load_image_and_labels(
                index=index
            )

            h, w = image_resized.shape[:2]

            if i == 0:
                # Create empty image with the above resized image.
                result_image = np.full((s * 2, s * 2, image_resized.shape[2]), 114, dtype=np.float32)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image_resized[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            if len(orig_boxes) > 0:
                boxes[:, 0] += padw
                boxes[:, 1] += padh
                boxes[:, 2] += padw
                boxes[:, 3] += padh

                result_boxes.append(boxes)
                result_classes += labels

        final_classes = []
        if len(result_boxes) > 0:
            result_boxes = np.concatenate(result_boxes, 0)
            np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
            result_boxes = result_boxes.astype(np.int32)
            for idx in range(len(result_boxes)):
                if ((result_boxes[idx, 2] - result_boxes[idx, 0]) * (result_boxes[idx, 3] - result_boxes[idx, 1])) > 0:
                    final_classes.append(result_classes[idx])
            result_boxes = result_boxes[
                np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)
            ]
        # Resize the mosaic image to the desired shape and transform boxes.
        result_image, result_boxes = transform_mosaic(
            result_image, result_boxes, self.img_size
        )
        return result_image, torch.tensor(result_boxes), \
            torch.tensor(np.array(final_classes)), area, iscrowd, dims

    def __getitem__(self, idx):
        # Capture the image name and the full image path.
        if self.no_mosaic:
            image, image_resized, orig_boxes, boxes, \
                labels, area, iscrowd, dims = self.load_image_and_labels(
                index=idx
            )

        if self.train and not self.no_mosaic:
            #while True:
            image_resized, boxes, labels, \
                area, iscrowd, dims = self.load_cutmix_image_and_boxes(
                idx, resize_factor=(self.img_size, self.img_size)
            )

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.use_train_aug: # Use train augmentation if argument is passed.
            train_aug = get_train_aug()
            sample = train_aug(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            # target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.float)
        else:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            # target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.float)

        _, h, w = image_resized.shape
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        boxes = A.core.bbox_utils.normalize_bboxes(sample['bboxes'], rows=h, cols=w)
        boxes = np.array(boxes)
        # Try-except block in case an image does not contains target boxes.
        try:
            boxes[:,:2] += boxes[:,2:] / 2
        except:
            pass

        target['boxes'] = torch.Tensor(boxes).to(torch.float)
        # Fix to enable training without target bounding boxes,
        # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float)
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

# Prepare the final datasets and data loaders.
def create_train_dataset(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    use_train_aug=False,
    no_mosaic=False,
    square_training=False
):
    train_dataset = DETRDataset(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        get_train_transform(),
        use_train_aug=use_train_aug,
        train=True, 
        no_mosaic=no_mosaic,
        square_training=square_training
    )
    return train_dataset
def create_valid_dataset(
    valid_dir_images, 
    valid_dir_labels, 
    img_size, 
    classes,
    square_training=False
):
    valid_dataset = DETRDataset(
        valid_dir_images, 
        valid_dir_labels, 
        img_size, 
        classes, 
        get_valid_transform(),
        train=False, 
        no_mosaic=True,
        square_training=square_training
    )
    return valid_dataset

def create_train_loader(
    train_dataset, batch_size, num_workers=0, batch_sampler=None
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler
    )
    return train_loader

def create_valid_loader(
    valid_dataset, batch_size, num_workers=0, batch_sampler=None
):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler
    )
    return valid_loader