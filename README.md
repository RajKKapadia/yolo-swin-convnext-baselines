## Environment

- Create a new virtual environment
```
python -m venv .venv && source .venv/bin/activate
```

- Install dependencies
```
pip install --upgrade pip wheel setuptools
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Note: Uses PyTorch AMP via `torch.amp.autocast` and `torch.amp.GradScaler`. Runs on CPU or CUDA.

## Dataset Layout

Before COCO conversion, arrange EuroCityPersons like:
```
./
├── data/
│   ├── dataset.py
│   ├── eurocity_to_coco.py
│   └── transforms.py
├── dataset/
│   └── EuroCityPersons/
│       └── prague/
│           ├── prague_00000.png
│           └── prague_00000.json
├── models/
│   ├── backbones.py
│   ├── fpn.py
│   ├── __init__.py
│   └── yolo_head.py
├── train.py
└── utils/
    ├── box_ops.py
    ├── losses.py
    └── metrics.py
```

## Convert EuroCityPersons → COCO

Place per-city folders under `dataset/EuroCityPersons/<city>/` with matching image (`.png`/`.jpg`/`.jpeg`) and JSON files.

Run the converter:
```
python data/eurocity_to_coco.py
```

Outputs:
```
dataset/COCO/annotations/train.json
dataset/COCO/annotations/val.json
```
If `val.json` is missing at train time, `train.py` will auto-split `train.json` 90/10 to create it.

## Train

`train.py` supports CLI hyperparameters, so you don’t need to edit code.

- Train both included backbones (Swin Tiny and ConvNeXt Tiny):
```
python train.py \
  --img-root dataset \
  --train-json dataset/COCO/annotations/train.json \
  --val-json dataset/COCO/annotations/val.json \
  --epochs 1 \
  --batch-size 8 \
  --img-size 640 \
  --lr 1e-3 \
  --weight-decay 0.05 \
  --num-classes 1 \
  --workers 8 \
  --device auto \
  --backbones swin_tiny_patch4_window7_224 convnext_tiny \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --eval-iou-thr 0.5 \
  --nms-iou 0.6
```

Common options:
```
  --epochs INT                number of training epochs
  --batch-size INT            training/validation batch size
  --img-size INT              image size used by transforms/backbone
  --lr FLOAT                  AdamW learning rate
  --weight-decay FLOAT        AdamW weight decay
  --num-classes INT           number of classes in dataset
  --workers INT               dataloader workers
  --device {auto,cpu,cuda}    select device (auto picks CUDA if available)
  --backbones NAMES...        one or more timm backbones to train
  --pretrained/--no-pretrained  enable/disable pretrained backbone (default: enabled)
  --focal-alpha FLOAT         focal loss alpha (default 0.25)
  --focal-gamma FLOAT         focal loss gamma (default 2.0)
  --eval-iou-thr FLOAT        IoU threshold for mAP@0.5 computation
  --nms-iou FLOAT             IoU used by NMS in evaluation
```

During training, checkpoints are saved to:
```
runs/<backbone>/epoch_XX.pt
```

After each epoch, validation metrics are saved to:
```
runs/<backbone>/metrics/epoch_XX/
  ├── metrics.json   # {mAP50, AUC_ROC, num_gt, ...}
  ├── pr_curve.csv   # recall,precision
  └── roc_curve.csv  # fpr,tpr
```

## Evaluate Last Checkpoint

Use `test.py` to evaluate the latest checkpoint on the validation set and print/save mAP@0.5 and AUC‑ROC.

- Auto-detect latest checkpoint under `runs/*/`:
```
python test.py
```

- Evaluate a specific backbone’s latest:
```
python test.py --backbone swin_tiny_patch4_window7_224
```

- Evaluate a specific checkpoint file:
```
python test.py --checkpoint runs/convnext_tiny/epoch_50.pt
```

Options:
```
  --img-root dataset \
  --val-json dataset/COCO/annotations/val.json \
  --img-size 640 \
  --batch-size 8 \
  --workers 8
```

Results are saved to:
```
runs/<backbone>/metrics/test_last/
```

## Multi-class (Occlusion-as-class)

To train 4 classes (`no`, `light`, `moderate`, `heavy`) based on occlusion:
- Set `USE_OCCLUSION_AS_CLASS = True` in `data/eurocity_to_coco.py` and reconvert the dataset.
- Pass `--num-classes 4` to `train.py` when training.

## Notes & Tips

- Device: training uses AMP on CUDA automatically; otherwise runs on CPU (slower). Select with `--device`.
- Dataloader: increase `--workers` for faster data loading; `pin_memory` is enabled only when on CUDA.
- Normalization: current transforms scale to `[0,1]`. You may add mean/std normalization in `data/transforms.py` for timm backbones.

## Hyperparameters at a Glance

- Optimizer: AdamW (`--lr`, `--weight-decay`)
- Loss: Focal Loss (`--focal-alpha`, `--focal-gamma`)
- Image size/augs: `--img-size` (see `data/transforms.py` for aug ops/probabilities)
- Evaluation thresholds: `--eval-iou-thr` and `--nms-iou`
