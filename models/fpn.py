import torch.nn as nn


class SimpleFPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        c3, c4, c5 = in_channels
        self.l3 = nn.Conv2d(c3, out_channels, 1)
        self.l4 = nn.Conv2d(c4, out_channels, 1)
        self.l5 = nn.Conv2d(c5, out_channels, 1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, feats):
        c3, c4, c5 = feats  # low→high
        p5 = self.l5(c5)
        p4 = self.l4(c4) + nn.functional.interpolate(
            p5, size=c4.shape[-2:], mode="nearest"
        )
        p3 = self.l3(c3) + nn.functional.interpolate(
            p4, size=c3.shape[-2:], mode="nearest"
        )
        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)
        return [p3, p4, p5]  # strides approx 8/16/32 if backbones downsample similarly
