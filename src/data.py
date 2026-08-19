import json
import hashlib
import random
from pathlib import Path
from typing import List, Dict

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ==========================
# Path configuration
# ==========================

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / "data" / "tiny-imagenet-200"

SPLIT_ROOT = ROOT / "splits"


SEED = 42


# ==========================
# Utils
# ==========================

def sha256_string(text: str):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def get_wnids():

    wnids_file = DATA_ROOT / "wnids.txt"

    with open(wnids_file) as f:
        wnids = [
            x.strip()
            for x in f.readlines()
        ]

    return sorted(wnids)



# ==========================
# Class mapping
# ==========================

def build_class_mapping(
        num_classes=100
):

    wnids = get_wnids()


    random.seed(SEED)

    selected = random.sample(
        wnids,
        num_classes
    )

    selected = sorted(selected)


    class_to_idx = {
        wnid:i
        for i,wnid in enumerate(selected)
    }


    SPLIT_ROOT.mkdir(
        exist_ok=True
    )


    with open(
        SPLIT_ROOT/"classes_seed42.txt",
        "w"
    ) as f:

        for c in selected:
            f.write(c+"\n")


    with open(
        SPLIT_ROOT/"class_to_idx_seed42.json",
        "w"
    ) as f:

        json.dump(
            class_to_idx,
            f,
            indent=2
        )


    return class_to_idx



# ==========================
# Split generation
# ==========================

def build_split():

    class_to_idx = build_class_mapping()


    train=[]
    val=[]
    test=[]


    for wnid,class_id in class_to_idx.items():


        train_dir = (
            DATA_ROOT
            /
            "train"
            /
            wnid
            /
            "images"
        )


        images = sorted(
            train_dir.glob("*.JPEG")
        )


        records=[]


        for img in images:


            relative = img.relative_to(
                DATA_ROOT
            ).as_posix()


            key = (
                f"split-v1|{SEED}|"
                f"{relative}"
            )


            h = sha256_string(key)


            records.append(
                (
                    h,
                    relative,
                    {
                        "relative_path":relative,
                        "WNID":wnid,
                        "class_id":class_id
                    }
                )
            )


        records.sort(
            key=lambda x:(x[0],x[1])
        )


        for _,_,item in records[:450]:
            train.append(item)


        for _,_,item in records[450:500]:
            val.append(item)



    # official val as test

    ann = (
        DATA_ROOT
        /
        "val"
        /
        "val_annotations.txt"
    )


    with open(ann) as f:

        for line in f:

            img,wnid,*_ = line.strip().split("\t")


            if wnid in class_to_idx:


                test.append(
                    {
                        "relative_path":
                        f"val/images/{img}",

                        "WNID":
                        wnid,

                        "class_id":
                        class_to_idx[wnid]
                    }
                )



    manifest={

        "seed":SEED,

        "configuration":"lite",

        "classes":
            list(class_to_idx.keys()),

        "class_to_idx":
            class_to_idx,

        "train":
            sorted(
                train,
                key=lambda x:
                (
                    x["class_id"],
                    x["relative_path"]
                )
            ),

        "validation":
            sorted(
                val,
                key=lambda x:
                (
                    x["class_id"],
                    x["relative_path"]
                )
            ),

        "test":
            sorted(
                test,
                key=lambda x:
                (
                    x["class_id"],
                    x["relative_path"]
                )
            )

    }



    with open(
        SPLIT_ROOT/"split_seed42.json",
        "w"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )


    checksum = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",",":")
        ).encode("utf-8")
    ).hexdigest()


    with open(
        SPLIT_ROOT/"split_seed42.sha256",
        "w"
    ) as f:

        f.write(checksum)


    return manifest



# ==========================
# Dataset
# ==========================


class TinyImageNetDataset(Dataset):


    def __init__(
        self,
        records,
        transform=None
    ):

        self.records=records
        self.transform=transform



    def __len__(self):

        return len(self.records)



    def __getitem__(self,index):

        item=self.records[index]


        path = (
            DATA_ROOT
            /
            item["relative_path"]
        )


        img = Image.open(path).convert("RGB")


        base_tensor = transforms.ToTensor()(img)


        rgb_mean = base_tensor.mean(
            dim=(1,2)
        )


        if self.transform:

            img=self.transform(img)


        return (
            img,
            item["class_id"],
            rgb_mean
        )



# ==========================
# DataLoader
# ==========================


def get_loaders(
    batch_size=8
):


    manifest_file = (
        SPLIT_ROOT
        /
        "split_seed42.json"
    )


    if not manifest_file.exists():

        build_split()


    with open(manifest_file) as f:

        manifest=json.load(f)



    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485,0.456,0.406),
                (0.229,0.224,0.225)
            )
        ]
    )


    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485,0.456,0.406),
                (0.229,0.224,0.225)
            )
        ]
    )


    train_ds = TinyImageNetDataset(
        manifest["train"],
        train_tf
    )


    val_ds = TinyImageNetDataset(
        manifest["validation"],
        test_tf
    )


    test_ds = TinyImageNetDataset(
        manifest["test"],
        test_tf
    )


    return (

        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True
        ),

        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False
        ),

        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False
        )

    )