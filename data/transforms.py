import albumentations as A


def get_train_transforms(img_size=640):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"], min_visibility=0.2
        ),
    )


def get_val_transforms(img_size=640):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )
