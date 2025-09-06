import timm


def make_backbone(name: str, pretrained=True, out_indices=(1, 2, 3), img_size=640):
    """
    Create a timm backbone that returns feature maps (features_only=True).
    Some models (e.g., Swin) accept img_size; others (e.g., ConvNeXt) don't.
    We try with img_size first, then fall back without it.
    """
    kwargs = dict(features_only=True, pretrained=pretrained, out_indices=out_indices)
    try:
        # Works for Swin / ViT-like models
        return timm.create_model(name, **kwargs, img_size=img_size)
    except TypeError:
        # For ConvNeXt and others that don't take img_size
        return timm.create_model(name, **kwargs)


def backbone_channels(name: str):
    # Channels for tiny/base variants used at out_indices (1,2,3)
    if "swin" in name:
        # swin_* -> [96, 192, 384, 768] -> use last 3
        return [192, 384, 768]
    if "convnext" in name:
        # convnext_* -> [96, 192, 384, 768] -> use last 3
        return [192, 384, 768]
    raise ValueError(f"Unknown backbone: {name}")
