import json
import os

import cv2
import torch
from torch.utils.data import Dataset


class CocoLikeDataset(Dataset):
    def __init__(self, img_root, ann_path, transforms=None):
        self.img_root = img_root
        self.transforms = transforms
        with open(ann_path, "r") as f:
            coco = json.load(f)
        self.images = {img["id"]: img for img in coco["images"]}
        self.anns_by_img = {img_id: [] for img_id in self.images.keys()}
        for ann in coco["annotations"]:
            self.anns_by_img[ann["image_id"]].append(ann)
        self.ids = list(self.images.keys())
        # class mapping
        cats = sorted(coco["categories"], key=lambda c: c["id"])
        self.cat_ids = [c["id"] for c in cats]
        self.id_to_idx = {cid: i for i, cid in enumerate(self.cat_ids)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        meta = self.images[img_id]
        path = os.path.join(self.img_root, meta["file_name"])
        img = cv2.imread(path)
        assert img is not None, f"Missing image: {path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, labels = [], []
        for ann in self.anns_by_img[img_id]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # convert to x1,y1,x2,y2
            labels.append(self.id_to_idx[ann["category_id"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
            if labels
            else torch.zeros((0,), dtype=torch.long),
            "image_id": torch.tensor([img_id]),
        }
        if self.transforms:
            augmented = self.transforms(
                image=img,
                bboxes=target["boxes"].tolist(),
                class_labels=target["labels"].tolist(),
            )
            img = augmented["image"]
            if augmented["bboxes"]:
                target["boxes"] = torch.tensor(augmented["bboxes"], dtype=torch.float32)
                target["labels"] = torch.tensor(
                    augmented["class_labels"], dtype=torch.long
                )
            else:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros((0,), dtype=torch.long)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, dim=0), list(targets)
