import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))


from src.data import (
    get_loaders,
    build_split
)

import json


# ======================
# create split if missing
# ======================

split_file = (
    ROOT
    /
    "splits"
    /
    "split_seed42.json"
)


if not split_file.exists():

    print("split file not found.")
    print("building split...")

    build_split()



# ======================
# load manifest
# ======================

with open(split_file) as f:

    manifest = json.load(f)



print("Tiny ImageNet data check")


print(
    "classes=",
    len(manifest["classes"])
)


print(
    "train total=",
    len(manifest["train"])
)


print(
    "validation total=",
    len(manifest["validation"])
)


print(
    "test total=",
    len(manifest["test"])
)



# ======================
# dataloader check
# ======================

train_loader, val_loader, test_loader = get_loaders(
    batch_size=8
)



for name, loader in [
    ("train", train_loader),
    ("validation", val_loader),
    ("test", test_loader)
]:

    images, labels, rgb_mean = next(iter(loader))


    print(
        f"[{name}]"
    )


    print(
        "batch_images_shape=",
        images.shape
    )


    print(
        "batch_labels_shape=",
        labels.shape
    )


    print(
        "batch_rgb_mean_shape=",
        rgb_mean.shape
    )



print("data_check_passed")