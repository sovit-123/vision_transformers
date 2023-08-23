import cv2
import numpy as np

from utils.detection.detr.general import rescale_bboxes

def convert_detections(
    outputs, 
    detection_threshold, 
    classes,
    orig_image,
    args
):
    height, width, _ = orig_image.shape
    probas   = outputs['pred_logits'].softmax(-1).detach().cpu()[0, :, :-1]
    keep = probas.max(-1).values > detection_threshold
    draw_boxes = rescale_bboxes(
        outputs['pred_boxes'][0, keep].detach().cpu(), 
        (width, height)
    )
    probas = probas[keep]
    if args.classes is not None: # Filter by classes.
        labels = np.array([int(probas[j].argmax()) + 1 for j in range(len(draw_boxes))])
        lbl_mask = np.isin(labels, args.classes)
        scores = np.array([float(probas[j].max()) for j in range(len(draw_boxes))])
        scores = scores[lbl_mask]
        draw_boxes = draw_boxes[lbl_mask]
        labels = labels[lbl_mask]
        pred_classes = [classes[label] for label in labels]
    else: # Keep all classes.
        scores = [float(probas[j].max()) for j in range(len(draw_boxes))]
        labels = [int(probas[j].argmax()) + 1 for j in range(len(draw_boxes))]
        pred_classes = [classes[label] for label in labels]

    return draw_boxes, pred_classes, scores

def convert_pre_track(
    draw_boxes, pred_classes, scores
):
    final_preds = []
    for i, box in enumerate(draw_boxes):
        # Append ([x, y, w, h], score, label_string). For deep sort real-time.
        final_preds.append(
            (
                [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                scores[i],
                str(pred_classes[i])
            )
        )
    return final_preds

def convert_post_track(
    tracks
):
    draw_boxes, pred_classes, scores, track_id = [], [], [], []
    for track in tracks:
        if not track.is_confirmed():
            continue
        score = track.det_conf
        if score is None:
            continue
        track_id = track.track_id
        pred_class = track.det_class
        pred_classes.append(f"{track_id} {pred_class}")
        scores.append(score)
        draw_boxes.append(track.to_ltrb())
    return draw_boxes, pred_classes, scores

def inference_annotations(
    draw_boxes,
    pred_classes,
    scores, 
    classes,
    colors, 
    orig_image, 
    args
):
    lw = max(round(sum(orig_image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.
    
    # Draw the bounding boxes and write the class name on top of it.
    for j, box in enumerate(draw_boxes):
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))
        class_name = pred_classes[j]
        if args.track:
            color = colors[classes.index(' '.join(class_name.split(' ')[1:]))]
        else:
            color = colors[classes.index(class_name)]
        cv2.rectangle(
            orig_image,
            p1, p2,
            color=color, 
            thickness=lw,
            lineType=cv2.LINE_AA
        )
        if not args.hide_labels:
            # For filled rectangle.
            final_label = class_name + ' ' + str(round(scores[j], 2))
            w, h = cv2.getTextSize(
                final_label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=lw / 3, 
                thickness=tf
            )[0]  # text width, height
            w = int(w - (0.20 * w))
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(
                orig_image, 
                p1, 
                p2, 
                color=color, 
                thickness=-1, 
                lineType=cv2.LINE_AA
            )  
            cv2.putText(
                orig_image, 
                final_label, 
                (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=lw / 3.8, 
                color=(255, 255, 255), 
                thickness=tf, 
                lineType=cv2.LINE_AA
            )
    return orig_image

def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
    ):
        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, int(y + text_h + font_scale - 1)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )
        return img

def annotate_fps(orig_image, fps_text):
    draw_text(
        orig_image,
        f"FPS: {fps_text:0.1f}",
        pos=(20, 20),
        font_scale=1.0,
        text_color=(204, 85, 17),
        text_color_bg=(255, 255, 255),
        font_thickness=2,
    )
    return orig_image