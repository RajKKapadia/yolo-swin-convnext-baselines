import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CocoLikeDataset, collate_fn
from data.transforms import get_train_transforms, get_val_transforms
from models import TIMMYolo
from utils.losses import FocalLoss, giou_loss
from utils.metrics import evaluate_map50_aucroc


# ---------- CONFIG (adjust if needed) ----------
IMG_ROOT = Path(
    "dataset"
)  # because file_name in COCO is like "EuroCityPersons/prague/xxx.png"
ANN_DIR = IMG_ROOT / "COCO" / "annotations"
TRAIN_JSON = ANN_DIR / "train.json"
VAL_JSON = ANN_DIR / "val.json"  # will be created if missing
RUNS_DIR = Path("runs")

EPOCHS = 1
BATCH_SIZE = 8
IMG_SIZE = 640
LR = 1e-3
NUM_CLASSES = 1  # set to 4 if you encoded occlusion as classes
NUM_WORKERS = min(8, os.cpu_count() or 4)
VAL_SPLIT = 0.1
RANDOM_SEED = 42
# ----------------------------------------------


def generate_grid(h, w, stride, device):
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    grid = torch.stack([(xs + 0.5) * stride, (ys + 0.5) * stride], dim=-1)  # [H,W,2]
    return grid


def assign_targets(cls_outs, reg_outs, targets, strides=(8, 16, 32), num_classes=1):
    # simple center-in-box assigner
    B = len(targets)
    device = cls_outs[0].device
    cls_tgts, box_tgts, mask_pos = [], [], []
    for cls_l, reg_l, s in zip(cls_outs, reg_outs, strides):
        B, C, H, W = cls_l.shape
        grid = generate_grid(H, W, s, device)  # [H,W,2]
        cls_t = torch.zeros((B, num_classes, H, W), device=device)
        box_t = torch.zeros((B, 4, H, W), device=device)
        pos_m = torch.zeros((B, 1, H, W), device=device)

        for b in range(B):
            boxes = targets[b]["boxes"].to(device)  # [N,4] xyxy
            labels = targets[b]["labels"].to(device)
            if boxes.numel() == 0:
                continue

            gx = grid[..., 0][None, None, :, :]
            gy = grid[..., 1][None, None, :, :]
            x1 = boxes[:, 0][:, None, None, None]
            y1 = boxes[:, 1][:, None, None, None]
            x2 = boxes[:, 2][:, None, None, None]
            y2 = boxes[:, 3][:, None, None, None]
            inside = (gx > x1) & (gx < x2) & (gy > y1) & (gy < y2)  # [N,1,H,W]
            if inside.any():
                areas = (x2 - x1) * (y2 - y1)
                best_idx = torch.argmin(
                    torch.where(inside, areas, torch.full_like(areas, 1e18)), dim=0
                )  # [1,H,W]
                pos = inside.any(dim=0)  # [1,H,W]
                pos_m[b, 0] = pos[0]

                best_labels = labels[best_idx.squeeze(0)]
                cls_hot = torch.zeros((num_classes, H, W), device=device)
                cls_hot.scatter_(0, best_labels[None, :, :], 1.0)
                cls_t[b] = cls_hot

                xx1 = x1.squeeze(0)[best_idx.squeeze(0)]
                yy1 = y1.squeeze(0)[best_idx.squeeze(0)]
                xx2 = x2.squeeze(0)[best_idx.squeeze(0)]
                yy2 = y2.squeeze(0)[best_idx.squeeze(0)]
                box_t[b, 0] = xx1
                box_t[b, 1] = yy1
                box_t[b, 2] = xx2
                box_t[b, 3] = yy2

        cls_tgts.append(cls_t)
        box_tgts.append(box_t)
        mask_pos.append(pos_m)
    return cls_tgts, box_tgts, mask_pos


def train_one(backbone_name: str, device: str = None):
    # device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    model = TIMMYolo(
        backbone_name=backbone_name, num_classes=NUM_CLASSES, pretrained=True
    ).to(device)

    train_ds = CocoLikeDataset(
        str(IMG_ROOT), str(TRAIN_JSON), transforms=get_train_transforms(IMG_SIZE)
    )
    val_ds = CocoLikeDataset(
        str(IMG_ROOT), str(VAL_JSON), transforms=get_val_transforms(IMG_SIZE)
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    cls_loss_fn = FocalLoss()
    # Use new torch.amp GradScaler API (replaces torch.cuda.amp.GradScaler)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    save_dir = RUNS_DIR / backbone_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"[{backbone_name}] epoch {epoch}/{EPOCHS}")
        avg_loss = 0.0

        for imgs, targets in pbar:
            imgs = imgs.to(device)
            # Use new torch.amp.autocast API (replaces torch.cuda.amp.autocast)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                cls_outs, reg_outs = model(imgs)
                cls_tgts, box_tgts, mask_pos = assign_targets(
                    cls_outs, reg_outs, targets, num_classes=NUM_CLASSES
                )

                cls_loss, reg_loss = 0.0, 0.0
                for cl, rg, ct, bt, mp in zip(
                    cls_outs, reg_outs, cls_tgts, box_tgts, mask_pos
                ):
                    B, C, H, W = cl.shape
                    cls_loss += cls_loss_fn(cl.view(B, C, -1), ct.view(B, C, -1))
                    pos = mp.bool().expand(B, 4, H, W)
                    if pos.any():
                        reg_loss += giou_loss(rg[pos].view(-1, 4), bt[pos].view(-1, 4))

                loss = cls_loss + reg_loss

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            avg_loss = 0.9 * avg_loss + 0.1 * loss.item() if avg_loss else loss.item()
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                cls=f"{float(cls_loss):.3f}",
                box=f"{float(reg_loss):.3f}",
            )

        # Evaluation: save metrics and curves per epoch using utils.metrics
        metrics_dir = save_dir / "metrics" / f"epoch_{epoch:02d}"
        metrics = evaluate_map50_aucroc(
            model=model,
            val_dl=val_dl,
            device=device,
            save_dir=metrics_dir,
            iou_thr=0.5,
            nms_iou=0.6,
        )
        print(
            f"[val] mAP@0.5: {metrics['mAP50']:.4f} | AUC-ROC: {metrics['AUC_ROC']:.4f}"
        )

        ckpt = save_dir / f"epoch_{epoch:02d}.pt"
        torch.save(model.state_dict(), ckpt)

    print(f"✅ finished: {backbone_name}. checkpoints in: {save_dir}")


if __name__ == "__main__":
    # 1) SWIN baseline
    train_one("swin_tiny_patch4_window7_224")
    # 2) ConvNeXt baseline
    train_one("convnext_tiny")
