import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from models import TIMMYolo
from data.dataset import CocoLikeDataset, collate_fn
from data.transforms import get_val_transforms
from utils.metrics import evaluate_map50_aucroc


RUNS_DIR = Path("runs")
DEFAULT_IMG_ROOT = Path("dataset")
DEFAULT_VAL_JSON = DEFAULT_IMG_ROOT / "COCO" / "annotations" / "val.json"


def _categories_from_coco(ann_path: Path) -> int:
    with open(ann_path, "r") as f:
        coco = json.load(f)
    cats = coco.get("categories", [])
    if not cats:
        raise RuntimeError(f"No categories found in {ann_path}")
    return len(cats)


def _latest_checkpoint(run_dir: Path) -> Optional[Path]:
    if not run_dir.exists():
        return None
    ckpts = sorted(run_dir.glob("epoch_*.pt"))
    if not ckpts:
        return None

    # Prefer by epoch number if parsable, else by mtime
    def _epoch_num(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except Exception:
            return -1

    ckpts_num = sorted(ckpts, key=lambda p: (_epoch_num(p), p.stat().st_mtime))
    return ckpts_num[-1]


def _latest_checkpoint_any(runs_dir: Path) -> Optional[Tuple[str, Path]]:
    if not runs_dir.exists():
        return None
    best: Optional[Tuple[str, Path, float]] = None  # (backbone, path, mtime)
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        ckpt = _latest_checkpoint(d)
        if ckpt is None:
            continue
        mtime = ckpt.stat().st_mtime
        if best is None or mtime > best[2]:
            best = (d.name, ckpt, mtime)
    if best is None:
        return None
    return best[0], best[1]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate latest checkpoint on validation set"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Backbone run directory name under runs/ (e.g., swin_tiny_patch4_window7_224)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint .pt to evaluate",
    )
    parser.add_argument(
        "--img-root",
        type=str,
        default=str(DEFAULT_IMG_ROOT),
        help="Image root directory (default: dataset)",
    )
    parser.add_argument(
        "--val-json",
        type=str,
        default=str(DEFAULT_VAL_JSON),
        help="Path to COCO val.json",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Validation image size (for transforms/backbone init)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()

    # device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    img_root = Path(args.img_root)
    val_json = Path(args.val_json)
    assert val_json.exists(), f"Missing val annotations: {val_json}"

    # Determine checkpoint and backbone
    backbone_name = args.backbone
    ckpt_path: Optional[Path] = Path(args.checkpoint) if args.checkpoint else None

    if ckpt_path is None:
        if backbone_name is not None:
            run_dir = RUNS_DIR / backbone_name
            ckpt_path = _latest_checkpoint(run_dir)
            if ckpt_path is None:
                raise RuntimeError(f"No checkpoints found under {run_dir}")
        else:
            latest = _latest_checkpoint_any(RUNS_DIR)
            if latest is None:
                raise RuntimeError(f"No checkpoints found under {RUNS_DIR}")
            backbone_name, ckpt_path = latest

    assert ckpt_path is not None
    if backbone_name is None:
        # infer backbone from path: runs/<backbone>/epoch_xx.pt
        try:
            backbone_name = ckpt_path.parent.name
        except Exception:
            backbone_name = "unknown_backbone"

    print(f"Using checkpoint: {ckpt_path}")
    print(f"Backbone: {backbone_name}")

    # Derive num_classes from val.json categories
    num_classes = _categories_from_coco(val_json)
    print(f"Detected num_classes from {val_json}: {num_classes}")

    # Build model and load weights
    model = TIMMYolo(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=False,
        img_size=args.img_size,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)

    # Data
    val_ds = CocoLikeDataset(
        str(img_root), str(val_json), transforms=get_val_transforms(args.img_size)
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    # Save metrics under the run directory
    save_dir = RUNS_DIR / backbone_name / "metrics" / "test_last"
    metrics = evaluate_map50_aucroc(model, val_dl, device, save_dir)
    print(f"mAP@0.5: {metrics['mAP50']:.4f} | AUC-ROC: {metrics['AUC_ROC']:.4f}")


if __name__ == "__main__":
    main()
