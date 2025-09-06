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

Adjust training settings at the top of `train.py` (e.g., `EPOCHS`, `BATCH_SIZE`, `NUM_CLASSES`).

Train both included backbones (Swin Tiny and ConvNeXt Tiny):
```
python train.py
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
- Set `NUM_CLASSES = 4` in `train.py` before training.

## Notes & Tips

- Device: training uses AMP on CUDA automatically; otherwise runs on CPU (slower). You can adjust `device` logic in `train_one` if needed.
- Dataloader: increase `--workers` for faster data loading; `pin_memory` is enabled only when on CUDA.
- Normalization: current transforms scale to `[0,1]`. You may add mean/std normalization in `data/transforms.py` for timm backbones.

