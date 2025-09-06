import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

# -------------------- CONFIG --------------------
IMG_EXTS = [".png", ".jpg", ".jpeg"]
RANDOM_SEED = 42
TRAIN_SPLIT = 0.9  # 90/10 split
USE_OCCLUSION_AS_CLASS = False  # True => 4 classes: no/light/moderate/heavy
OCC_LABELS = ["no", "light", "moderate", "heavy"]
# ------------------------------------------------


def _infer_bbox(ann: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Return COCO-style [x, y, w, h].
    Supports:
      - 'bbox': [x, y, w, h]
      - (x1, y1, x2, y2)
      - (x, y, w, h)
    """
    # [x,y,w,h]
    if (
        "bbox" in ann
        and isinstance(ann["bbox"], (list, tuple))
        and len(ann["bbox"]) == 4
    ):
        x, y, w, h = ann["bbox"]
        return float(x), float(y), float(w), float(h)

    # (x1,y1,x2,y2)
    keys = ann.keys()
    if all(k in keys for k in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = (
            float(ann["x1"]),
            float(ann["y1"]),
            float(ann["x2"]),
            float(ann["y2"]),
        )
        return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)

    # (x,y,w,h)
    if all(k in keys for k in ("x", "y", "w", "h")):
        return float(ann["x"]), float(ann["y"]), float(ann["w"]), float(ann["h"])

    raise KeyError(
        "Could not infer bbox from annotation. Expected 'bbox' or (x1,y1,x2,y2) or (x,y,w,h)."
    )


def _bucket_occ(vnum: float) -> str:
    """Map numeric occlusion to buckets using your thresholds."""
    # If 0..1 given, treat as fraction and scale to percent.
    if 0.0 <= vnum <= 1.0:
        vnum *= 100.0
    if vnum > 80:
        return "heavy"
    elif vnum > 40:
        return "moderate"
    elif vnum > 10:
        return "light"
    else:
        return "no"


def _infer_occlusion(ann: Dict[str, Any]) -> str:
    """
    Map various occlusion encodings to one of OCC_LABELS.
    Accepts:
      - 'occlusion'/'occ'/'occl'/'occlusion_level' as string label or numeric (%, fraction).
    Default 'no' if absent/unknown.
    """
    for k in ("occlusion", "occ", "occl", "occlusion_level"):
        if k in ann:
            v = ann[k]
            # label as string
            if isinstance(v, str):
                v_norm = v.strip().lower().replace("%", "")
                if v_norm in OCC_LABELS:
                    return v_norm
                # numeric in string
                try:
                    return _bucket_occ(float(v_norm))
                except Exception:
                    return "no"
            # numeric
            try:
                return _bucket_occ(float(v))
            except Exception:
                return "no"
    return "no"


def _image_size_from_json_or_file(
    img_path: Path, meta: Dict[str, Any]
) -> Tuple[int, int]:
    """
    Try to read width/height from JSON. If missing, fall back to reading the image (OpenCV).
    Returns (width, height); (0,0) if unknown.
    """
    w = meta.get("width") or meta.get("img_width") or meta.get("w") or 0
    h = meta.get("height") or meta.get("img_height") or meta.get("h") or 0
    w, h = int(w), int(h)
    if w == 0 or h == 0:
        try:
            import cv2

            im = cv2.imread(str(img_path))
            if im is not None:
                h, w = im.shape[:2]
        except Exception:
            pass
    return w, h


def _collect_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """
    Walk dataset/EuroCityPersons/<city>/, pair *.json with same-named image.
    Skip if no image exists. Prefer .png, else .jpg/.jpeg.
    Returns list of (img_path, json_path).
    """
    euro_dir = root / "EuroCityPersons"
    if not euro_dir.exists():
        raise FileNotFoundError(f"Expected directory: {euro_dir}")

    pairs: List[Tuple[Path, Path]] = []
    for city in sorted([p for p in euro_dir.iterdir() if p.is_dir()]):
        for j in sorted(city.glob("*.json")):
            base = j.stem
            found_img = None
            for ext in IMG_EXTS:
                cand = city / f"{base}{ext}"
                if cand.exists():
                    found_img = cand
                    break
            if found_img is None:
                print(f"⚠️  Skipping {j} (no matching image found among {IMG_EXTS})")
                continue
            pairs.append((found_img, j))
    return pairs


def _choose_ann_list(meta: Any) -> List[Dict[str, Any]]:
    """
    Find the list of objects/boxes in a JSON. Tries common keys; falls back to top-level list.
    """
    if isinstance(meta, list):
        return meta
    for k in ("objects", "annotations", "boxes", "labels"):
        if k in meta and isinstance(meta[k], list):
            return meta[k]
    return []


def _build_coco(
    pairs_subset: List[Tuple[Path, Path]],
    root: Path,
) -> Dict[str, Any]:
    """
    Build a COCO dict from (img_path, json_path) pairs.
    file_name is stored relative to `root` (the dataset directory).
    """
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    ann_id = 1
    img_id = 1

    if USE_OCCLUSION_AS_CLASS:
        categories = [
            {"id": i + 1, "name": lab, "supercategory": "person"}
            for i, lab in enumerate(OCC_LABELS)
        ]
        name_to_cid = {lab: i + 1 for i, lab in enumerate(OCC_LABELS)}
    else:
        categories = [{"id": 1, "name": "pedestrian", "supercategory": "person"}]

    for img_path, json_path in pairs_subset:
        if not img_path.exists():
            print(f"⚠️  Skipping {json_path} (image missing at runtime).")
            continue

        with open(json_path, "r") as f:
            meta = json.load(f)

        w, h = _image_size_from_json_or_file(img_path, meta)

        images.append(
            {
                "id": img_id,
                # relative path from dataset/ so that IMG_ROOT="dataset" works
                "file_name": str(img_path.relative_to(root)),
                "width": w,
                "height": h,
            }
        )

        objs = _choose_ann_list(meta)
        for obj in objs:
            try:
                x, y, bw, bh = _infer_bbox(obj)
            except KeyError:
                continue

            if USE_OCCLUSION_AS_CLASS:
                occ = _infer_occlusion(obj)
                cid = name_to_cid[occ]
            else:
                cid = 1

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [float(x), float(y), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        img_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def convert(root_dir: str, out_dir: str) -> None:
    """
    root_dir should be 'dataset'
    out_dir will be 'dataset/COCO/annotations'
    """
    root = Path(root_dir).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    pairs = _collect_pairs(root)
    if not pairs:
        raise RuntimeError(
            f"No valid (image,json) pairs found under {root}/EuroCityPersons/<city>/"
        )

    random.seed(RANDOM_SEED)
    random.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_SPLIT)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    train_coco = _build_coco(train_pairs, root)
    val_coco = _build_coco(val_pairs, root)

    with open(out / "train.json", "w") as f:
        json.dump(train_coco, f)
    with open(out / "val.json", "w") as f:
        json.dump(val_coco, f)

    print(
        f"✅ Wrote {out / 'train.json'}  — {len(train_coco['images'])} images, {len(train_coco['annotations'])} anns"
    )
    print(
        f"✅ Wrote {out / 'val.json'}    — {len(val_coco['images'])} images, {len(val_coco['annotations'])} anns"
    )


if __name__ == "__main__":
    # Example usage for your structure:
    # dataset/
    # ├─ EuroCityPersons/<city>/{*.png,*.json}
    # └─ COCO/annotations/
    convert(root_dir="dataset", out_dir="dataset/COCO/annotations")
