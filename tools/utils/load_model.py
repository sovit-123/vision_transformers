from vision_transformers.models import *

def return_vit_b_p32_224(image_size=224, num_classes=1000, pretrained=False):
    model = vit.vit_b_p32_224(
        image_size=image_size, num_classes=num_classes, pretrained=pretrained
    )
    return model

def return_vit_ti_p16_224(image_size=224, num_classes=1000, pretrained=False):
    model = vit.vit_ti_p16_224(
        image_size=image_size, num_classes=num_classes, pretrained=pretrained
    )
    return model

def return_vit_ti_p16_384(image_size=384, num_classes=1000, pretrained=False):
    model = vit.vit_ti_p16_384(
        image_size=image_size, num_classes=num_classes, pretrained=pretrained
    )
    return model

def return_vit_b_p16_224(image_size=224, num_classes=1000, pretrained=False):
    model = vit.vit_b_p16_224(
        image_size=image_size, num_classes=num_classes, pretrained=pretrained
    )
    return model

def return_swin_b_p4_w7_224(
    image_size=224, num_classes=1000, pretrained=False
):
    model = swin_transformer.swin_b_p4_w7_224(
        image_size=image_size, num_classes=num_classes, pretrained=pretrained
    )
    return model

def return_swin_t_p4_w7_224(
    image_size=224, num_classes=1000, pretrained=False
):
    model = swin_transformer.swin_t_p4_w7_224(
        image_size=image_size, num_classes=num_classes, pretrained=pretrained
    )
    return model

def return_swin_s_p4_w7_224(
    image_size=224, num_classes=1000, pretrained=False
):
    model = swin_transformer.swin_s_p4_w7_224(
        image_size=image_size, num_classes=num_classes, pretrained=pretrained
    )
    return model

def return_swin_l_p4_w7_224(
    image_size=224, num_classes=1000, pretrained=False
):
    model = swin_transformer.swin_l_p4_w7_224(
        image_size=image_size, num_classes=num_classes, pretrained=pretrained
    )
    return model

def return_mobilevit_s(
    image_size=224, num_classes=1000, pretrained=False
):
    model = mobile_vit.mobilevit_s(
        num_classes=num_classes, pretrained=pretrained
    )
    return model

def return_mobilevit_xs(
    image_size=224, num_classes=1000, pretrained=False
):
    model = mobile_vit.mobilevit_xs(
        num_classes=num_classes, pretrained=pretrained
    )
    return model

def return_mobilevit_xxs(
    image_size=224, num_classes=1000, pretrained=False
):
    model = mobile_vit.mobilevit_xxs(
        num_classes=num_classes, pretrained=pretrained
    )
    return model

create_model = {
    'vit_b_p32_224': return_vit_b_p32_224,
    'vit_ti_p16_224': return_vit_ti_p16_224,
    'vit_ti_p16_384': return_vit_ti_p16_384,
    'vit_b_p16_224': return_vit_b_p16_224,
    'swin_b_p4_w7_224': return_swin_b_p4_w7_224,
    'swin_t_p4_w7_224': return_swin_t_p4_w7_224,
    'swin_s_p4_w7_224': return_swin_s_p4_w7_224,
    'swin_l_p4_w7_224': return_swin_l_p4_w7_224,
    'mobilevit_s': return_mobilevit_s,
    'mobilevit_xs': return_mobilevit_xs,
    'mobilevit_xxs': return_mobilevit_xxs
}