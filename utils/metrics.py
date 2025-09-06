# utils/metrics.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Dict
import torch
import torchvision
import numpy as np


# ------- small helper: NMS-based postprocess (same shape contract as your train.py) -------
@torch.no_grad()
def _postprocess_logits(
    cls_outs: List[torch.Tensor],
    reg_outs: List[torch.Tensor],
    conf_thres: float = 0.0,  # keep everything; ranking by score will handle thresholds
    iou_nms: float = 0.6,
    max_det: int = 300,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    preds = []
    B = cls_outs[0].shape[0]
    for b in range(B):
        all_boxes, all_scores, all_labels = [], [], []
        for cl, rg in zip(cls_outs, reg_outs):
            _, C, H, W = cl.shape
            cl_b = torch.sigmoid(cl[b].reshape(C, -1))  # [C, HW]
            rg_b = rg[b].permute(1, 2, 0).reshape(-1, 4)  # [HW, 4] (xyxy)
            scores, labels = cl_b.max(dim=0)  # [HW], [HW]
            keep = scores > conf_thres
            if keep.any():
                all_boxes.append(rg_b[keep])
                all_scores.append(scores[keep])
                all_labels.append(labels[keep])
        if len(all_boxes) == 0:
            preds.append(
                (
                    torch.zeros((0, 4)),
                    torch.zeros((0,)),
                    torch.zeros((0,), dtype=torch.long),
                )
            )
            continue
        boxes = torch.cat(all_boxes, 0)
        scores = torch.cat(all_scores, 0)
        labels = torch.cat(all_labels, 0)
        keep = torchvision.ops.nms(boxes, scores, iou_nms)[:max_det]
        preds.append((boxes[keep], scores[keep], labels[keep]))
    return preds


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [Na,4], b: [Nb,4] in xyxy
    inter = (
        torch.min(a[:, None, 2:], b[None, :, 2:])
        - torch.max(a[:, None, :2], b[None, :, :2])
    ).clamp(min=0)
    inter = inter[..., 0] * inter[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union


def _per_class_ap50(
    dets: List[Tuple[int, float, torch.Tensor]],  # [(img_idx, score, box_xyxy)]
    gts: List[Tuple[int, torch.Tensor]],  # [(img_idx, box_xyxy)]
    iou_thr: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AP@0.5 for ONE class by sorting detections by score, greedy-match to GT (1 per GT).
    Returns (ap, precision[], recall[]).
    """
    if len(gts) == 0:
        # no ground-truth -> define AP as 0 to avoid NaN
        return 0.0, np.array([0.0]), np.array([0.0])

    # sort detections by score desc
    dets_sorted = sorted(dets, key=lambda x: x[1], reverse=True)
    scores = []
    tp_flags = []  # 1 if matched a new GT at iou>=thr else 0
    # mark which GT in each image is used
    gt_used: Dict[
        int, torch.Tensor
    ] = {}  # image_idx -> bool tensor (len = num gt in that image)

    # build quick GT index by image
    gt_by_img: Dict[int, List[torch.Tensor]] = {}
    for img_idx, box in gts:
        gt_by_img.setdefault(img_idx, []).append(box)
    for img_idx, lst in gt_by_img.items():
        gt_by_img[img_idx] = (
            torch.stack(lst, dim=0) if len(lst) else torch.zeros((0, 4))
        )
        gt_used[img_idx] = torch.zeros((gt_by_img[img_idx].shape[0],), dtype=torch.bool)

    for img_idx, score, box in dets_sorted:
        scores.append(score)
        gboxes = gt_by_img.get(img_idx, None)
        if gboxes is None or gboxes.numel() == 0:
            tp_flags.append(0)
            continue
        ious = _box_iou(box.unsqueeze(0), gboxes).squeeze(0)  # [num_gt]
        ious[gt_used[img_idx]] = -1.0  # do not match used GTs
        best_iou, best_gt = (
            (ious.max().item(), int(ious.argmax().item()))
            if ious.numel()
            else (0.0, -1)
        )
        if best_iou >= iou_thr and best_gt >= 0:
            if not gt_used[img_idx][best_gt]:
                gt_used[img_idx][best_gt] = True
                tp_flags.append(1)
            else:
                tp_flags.append(0)
        else:
            tp_flags.append(0)

    tp = np.cumsum(tp_flags)
    fp = np.cumsum([1 - t for t in tp_flags])
    total_gt = len(gts)
    recall = tp / (total_gt + 1e-12)
    precision = tp / np.maximum(tp + fp, 1e-12)

    # precision envelope
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # integrate PR curve (all-points interpolation)
    ap = 0.0
    prev_r = 0.0
    for r, p in zip(recall, precision):
        if r > prev_r:
            ap += p * (r - prev_r)
            prev_r = r
    return float(ap), precision, recall


def _auc_roc_from_scores(
    scores: np.ndarray, is_pos: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Binary ROC AUC using detection scores:
    positives = detections with IoU>=thr to ANY GT (not 1-to-1),
    negatives = detections with IoU<thr to all GT.
    """
    if len(scores) == 0 or is_pos.sum() == 0 or is_pos.sum() == len(is_pos):
        # edge cases: no detections, or all pos / all neg -> undefined; return 0.5 neutral
        return 0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(-scores)  # desc
    labels = is_pos[order]
    # cumulative counts as we lower threshold
    cum_tp = np.cumsum(labels == 1)
    cum_fp = np.cumsum(labels == 0)
    P = (is_pos == 1).sum()
    N = (is_pos == 0).sum()
    # avoid division by zero
    tpr = cum_tp / (P + 1e-12)
    fpr = cum_fp / (N + 1e-12)
    # trapezoidal AUC over FPR (x-axis)
    auc = float(np.trapz(tpr, fpr))
    return auc, fpr, tpr


@torch.no_grad()
def evaluate_map50_aucroc(
    model: torch.nn.Module,
    val_dl,
    device: str,
    save_dir: Path,
    iou_thr: float = 0.5,
    nms_iou: float = 0.6,
) -> Dict[str, float]:
    """
    Run model on val dataloader and save:
      - metrics.json with {mAP50, AUC_ROC, num_gt, num_det}
      - pr_curve.csv   with "recall,precision" (macro over classes)
      - roc_curve.csv  with "fpr,tpr"        (global)
    Works for 1+ classes. For mAP50 we macro-average AP over classes that appear in GT.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # storage per class
    dets_by_cls: Dict[int, List[Tuple[int, float, torch.Tensor]]] = {}
    gts_by_cls: Dict[int, List[Tuple[int, torch.Tensor]]] = {}

    # also store for ROC
    all_scores: List[float] = []
    all_is_pos: List[int] = []

    img_idx = 0
    for imgs, targets in val_dl:
        imgs = imgs.to(device)
        cls_outs, reg_outs = model(imgs)
        batch_preds = _postprocess_logits(
            cls_outs, reg_outs, conf_thres=0.0, iou_nms=nms_iou
        )

        for b, (boxes, scores, labels) in enumerate(batch_preds):
            # GT for this image
            gt_boxes = targets[b]["boxes"]  # tensor [Ng,4] on CPU
            gt_labels = targets[b]["labels"]  # tensor [Ng]
            if gt_boxes.numel() > 0:
                for lb, bx in zip(gt_labels.tolist(), gt_boxes):
                    gts_by_cls.setdefault(int(lb), []).append((img_idx, bx))

            # AP bookkeeping (per class)
            if boxes.numel() > 0:
                for box, score, lb in zip(boxes, scores, labels):
                    c = int(lb.item())
                    dets_by_cls.setdefault(c, []).append(
                        (img_idx, float(score.item()), box.cpu())
                    )

            # ROC bookkeeping (global): label detection as positive if IoU>=thr to ANY GT (any class)
            if boxes.numel() > 0:
                if gt_boxes.numel() > 0:
                    ious = _box_iou(boxes.cpu(), gt_boxes)  # [Nd, Ng]
                    best_iou, _ = ious.max(dim=1)
                    is_pos = (best_iou.numpy() >= iou_thr).astype(np.int32)
                else:
                    is_pos = np.zeros((boxes.shape[0],), dtype=np.int32)
                all_scores.extend([float(s) for s in scores.cpu().numpy().tolist()])
                all_is_pos.extend(is_pos.tolist())

            img_idx += 1

    # compute AP per class (only for classes that have GT)
    aps: List[float] = []
    # store a macro PR curve by averaging precisions at same recall grid (optional)
    pr_curves = []

    # pick all classes present in GT
    gt_classes = sorted(gts_by_cls.keys())
    for c in gt_classes:
        ap, precision, recall = _per_class_ap50(
            dets=dets_by_cls.get(c, []),
            gts=gts_by_cls.get(c, []),
            iou_thr=iou_thr,
        )
        aps.append(ap)
        # resample PR to a fixed recall grid for macro plotting
        grid = np.linspace(0, 1, 101)
        # for each grid r, take max precision where recall >= r (step-wise)
        p_at_r = []
        for r in grid:
            mask = recall >= r
            p_at_r.append(float(precision[mask].max()) if mask.any() else 0.0)
        pr_curves.append(np.array(p_at_r))

    if aps:
        map50 = float(np.mean(aps))
        pr_macro = np.mean(np.stack(pr_curves, axis=0), axis=0)
        recall_grid = np.linspace(0, 1, 101)
    else:
        map50 = 0.0
        pr_macro = np.zeros((101,), dtype=float)
        recall_grid = np.linspace(0, 1, 101)

    # ROC AUC (global)
    if len(all_scores) == 0:
        auc, fpr, tpr = 0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0])
    else:
        auc, fpr, tpr = _auc_roc_from_scores(
            np.array(all_scores, dtype=float), np.array(all_is_pos, dtype=int)
        )

    # save metrics
    metrics = {
        "iou_thr": iou_thr,
        "mAP50": map50,
        "AUC_ROC": float(auc),
        "num_gt": int(sum(len(v) for v in gts_by_cls.values())),
        "num_det": int(len(all_scores)),
        "num_classes_with_gt": int(len(gt_classes)),
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # save curves
    pr_csv = save_dir / "pr_curve.csv"
    with open(pr_csv, "w") as f:
        f.write("recall,precision\n")
        for r, p in zip(recall_grid, pr_macro):
            f.write(f"{r:.6f},{p:.6f}\n")

    roc_csv = save_dir / "roc_curve.csv"
    with open(roc_csv, "w") as f:
        f.write("fpr,tpr\n")
        for x, y in zip(fpr, tpr):
            f.write(f"{float(x):.6f},{float(y):.6f}\n")

    print(
        f"✅ Saved metrics to {save_dir / 'metrics.json'} (mAP50={map50:.4f}, AUC_ROC={float(auc):.4f})"
    )
    return metrics
