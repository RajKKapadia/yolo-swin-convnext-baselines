import torch
import torch.nn as nn
from .box_ops import box_iou


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.red = alpha, gamma, reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):  # logits [B,C,K], targets [B,C,K] in {0,1}
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        loss = (
            (self.alpha * targets + (1 - self.alpha) * (1 - targets))
            * (1 - pt).pow(self.gamma)
            * bce
        )
        return loss.mean() if self.red == "mean" else loss.sum()


def giou_loss(pred, target):  # pred,target [N,4] xyxy
    # minimal, fast approximation: 1 - IoU
    iou = box_iou(pred, target).diag()  # assumes matched
    return (1 - iou).mean()
