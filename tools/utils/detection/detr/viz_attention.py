import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

from .general import rescale_bboxes

def visualize_attention(
    model, 
    inputs, 
    detection_threshold, 
    orig_image,
    out_dir,
    device
):
    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    # For COCO pretrained model.
    try:
        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]
    # For custom model which is enclosed within `model` block.
    except:
        hooks = [
            model.model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            model.model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]
    
    # propagate through the model
    outputs = model(inputs.to(device))

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0].cpu().detach()

    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]

    height, width, _ = orig_image.shape
    probas   = outputs['pred_logits'].softmax(-1).detach().cpu()[0, :, :-1]
    keep = probas.max(-1).values > detection_threshold
    bboxes_scaled = rescale_bboxes(
        outputs['pred_boxes'][0, keep].detach().cpu(), 
        (width, height)
    ).cpu().detach()

    final_image = dec_attn_weights[0, 0].view(h, w)
    final_image = final_image

    # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(22, 7))
    for idx, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), bboxes_scaled):
        ax = axs
        if idx > 0:
            final_image += dec_attn_weights[0, idx].view(h, w)
        ax.imshow(final_image)
        ax.axis('off')
    fig.tight_layout()

    final_image = cv2.resize(np.array(final_image), (width, height)) * 255.
    # cv2.imwrite(
    #     os.path.join(out_dir, 'attention_map.png'), final_image
    # )
    plt.savefig(out_dir)