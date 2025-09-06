import torch
import torch.nn as nn
from .backbones import make_backbone, backbone_channels
from .fpn import SimpleFPN
from .yolo_head import YOLOHead


class TIMMYolo(nn.Module):
    def __init__(
        self, backbone_name: str, num_classes: int, pretrained=True, img_size=640
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.expected_ch = backbone_channels(backbone_name)
        self.backbone = make_backbone(
            backbone_name,
            pretrained=pretrained,
            out_indices=(1, 2, 3),
            img_size=img_size,
        )
        self.fpn = SimpleFPN(self.expected_ch, out_channels=256)
        self.head = YOLOHead(num_classes=num_classes, ch=256)

    def _ensure_nchw(self, f: torch.Tensor) -> torch.Tensor:
        # If last dim matches expected channels (e.g., 192/384/768), but dim=1 does not,
        # the feature is NHWC -> permute to NCHW.
        if f.dim() == 4:
            if (f.shape[-1] in self.expected_ch) and (
                f.shape[1] not in self.expected_ch
            ):
                return f.permute(0, 3, 1, 2).contiguous()
        return f

    def forward(self, x):
        feats = self.backbone(x)  # list of 3 feature maps
        feats = [self._ensure_nchw(f) for f in feats]
        feats = self.fpn(feats)  # p3, p4, p5 in NCHW
        return self.head(feats)
