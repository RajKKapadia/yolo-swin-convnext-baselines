import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.cv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.cv(x)))


class YOLOHead(nn.Module):
    def __init__(
        self, num_classes, ch=256, reg_max=0
    ):  # reg_max=0 -> direct box deltas
        super().__init__()
        self.num_classes = num_classes
        self.stems = nn.ModuleList([ConvBNAct(ch, ch, 3, 1, 1) for _ in range(3)])
        self.cls_convs = nn.ModuleList(
            [nn.Sequential(ConvBNAct(ch, ch), ConvBNAct(ch, ch)) for _ in range(3)]
        )
        self.reg_convs = nn.ModuleList(
            [nn.Sequential(ConvBNAct(ch, ch), ConvBNAct(ch, ch)) for _ in range(3)]
        )
        self.cls_preds = nn.ModuleList(
            [nn.Conv2d(ch, num_classes, 1) for _ in range(3)]
        )
        self.reg_preds = nn.ModuleList([nn.Conv2d(ch, 4, 1) for _ in range(3)])

    def forward(self, feats):
        cls_outs, reg_outs = [], []
        for i, x in enumerate(feats):
            x = self.stems[i](x)
            cls_feat = self.cls_convs[i](x)
            reg_feat = self.reg_convs[i](x)
            cls_outs.append(self.cls_preds[i](cls_feat))  # [B,C,H,W]
            reg_outs.append(self.reg_preds[i](reg_feat))  # [B,4,H,W]
        return cls_outs, reg_outs
