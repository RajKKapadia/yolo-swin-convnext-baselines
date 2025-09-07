import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CocoLikeDataset, collate_fn
from data.transforms import get_train_transforms, get_val_transforms
from models import TIMMYolo
from utils.losses import FocalLoss, giou_loss
from utils.metrics import evaluate_map50_aucroc


# Default paths
IMG_ROOT = Path("dataset")  # because file_name in COCO is like "EuroCityPersons/prague/xxx.png"
ANN_DIR = IMG_ROOT / "COCO" / "annotations"
TRAIN_JSON = ANN_DIR / "train.json"
VAL_JSON = ANN_DIR / "val.json"
RUNS_DIR = Path("runs")


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


def train_one(
    backbone_name: str,
    img_root: Path,
    train_json: Path,
    val_json: Path,
    runs_dir: Path,
    epochs: int,
    batch_size: int,
    img_size: int,
    lr: float,
    weight_decay: float,
    num_classes: int,
    num_workers: int,
    device: str,
    focal_alpha: float,
    focal_gamma: float,
    eval_iou_thr: float,
    eval_nms_iou: float,
    pretrained: bool,
):
    model = TIMMYolo(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        img_size=img_size,
    ).to(device)

    train_ds = CocoLikeDataset(
        str(img_root), str(train_json), transforms=get_train_transforms(img_size)
    )
    val_ds = CocoLikeDataset(
        str(img_root), str(val_json), transforms=get_val_transforms(img_size)
    )

    pin = device == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cls_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    save_dir = runs_dir / backbone_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"[{backbone_name}] epoch {epoch}/{epochs}")
        avg_loss = 0.0

        for imgs, targets in pbar:
            imgs = imgs.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                cls_outs, reg_outs = model(imgs)
                cls_tgts, box_tgts, mask_pos = assign_targets(
                    cls_outs, reg_outs, targets, num_classes=num_classes
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
            iou_thr=eval_iou_thr,
            nms_iou=eval_nms_iou,
        )
        print(
            f"[val] mAP@0.5: {metrics['mAP50']:.4f} | AUC-ROC: {metrics['AUC_ROC']:.4f}"
        )

        ckpt = save_dir / f"epoch_{epoch:02d}.pt"
        torch.save(model.state_dict(), ckpt)

    print(f"✅ finished: {backbone_name}. checkpoints in: {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TIMMYolo with configurable hyperparameters")
    parser.add_argument("--img-root", type=str, default=str(IMG_ROOT), help="Image root directory")
    parser.add_argument("--train-json", type=str, default=str(TRAIN_JSON), help="Path to COCO train.json")
    parser.add_argument("--val-json", type=str, default=str(VAL_JSON), help="Path to COCO val.json")
    parser.add_argument("--runs-dir", type=str, default=str(RUNS_DIR), help="Output runs directory")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    parser.add_argument(
        "--backbones",
        type=str,
        nargs="+",
        default=["swin_tiny_patch4_window7_224", "convnext_tiny"],
        help="One or more timm backbone names",
    )
    # Pretrained flag (enabled by default); use --no-pretrained to disable
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)

    # Loss
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    # Evaluation / NMS thresholds
    parser.add_argument("--eval-iou-thr", type=float, default=0.5, help="IoU threshold for mAP50 evaluation")
    parser.add_argument("--nms-iou", type=float, default=0.6, help="IoU for NMS during evaluation")

    return parser.parse_args()


def main():
    args = parse_args()

    # Device resolution
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Seeding (basic)
    torch.manual_seed(args.seed)

    img_root = Path(args.img_root)
    train_json = Path(args.train_json)
    val_json = Path(args.val_json)
    runs_dir = Path(args.runs_dir)

    for backbone in args.backbones:
        train_one(
            backbone_name=backbone,
            img_root=img_root,
            train_json=train_json,
            val_json=val_json,
            runs_dir=runs_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_classes=args.num_classes,
            num_workers=args.workers,
            device=device,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            eval_iou_thr=args.eval_iou_thr,
            eval_nms_iou=args.nms_iou,
            pretrained=args.pretrained,
        )


if __name__ == "__main__":
    main()
