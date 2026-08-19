import sys
from pathlib import Path


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import torch
import torch.nn as nn
from torch.optim import AdamW


from src.data import get_loaders
from models.mini_vit_full import MiniViTFull


# ============================================================
# Configuration
# ============================================================

EPOCHS = 20

BATCH_SIZE = 64

LR = 3e-4

WEIGHT_DECAY = 0.05


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Output paths
# ============================================================

SAVE_DIR = (
    ROOT
    / "outputs"
    / "mini_vit_full"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BEST_PATH = (
    SAVE_DIR
    / "best.pt"
)


LOG_PATH = (
    SAVE_DIR
    / "train_log.txt"
)


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0


    for images, labels, _ in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad()


        logits = model(
            images
        )


        loss = criterion(
            logits,
            labels
        )


        loss.backward()


        optimizer.step()


        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )


        predictions = logits.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += batch_size


    epoch_loss = (
        total_loss / total
    )

    epoch_accuracy = (
        correct / total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# Validation
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels, _ in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            logits = model(
                images
            )


            predictions = logits.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    return correct / total


# ============================================================
# Main
# ============================================================

def main():

    print(
        f"Device: {DEVICE}"
    )


    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, val_loader, _ = get_loaders(
        batch_size=BATCH_SIZE
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MiniViTFull(
        img_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=192,
        num_heads=4,
        depth=4,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=100
    )


    model = model.to(
        DEVICE
    )


    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_acc = 0.0


    with open(
        LOG_PATH,
        "w",
        encoding="utf-8"
    ) as log_file:

        log_file.write(
            f"Device: {DEVICE}\n"
        )


        for epoch in range(
            EPOCHS
        ):

            train_loss, train_acc = (
                train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer
                )
            )


            val_acc = evaluate(
                model,
                val_loader
            )


            log_line = (
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Train Loss:{train_loss:.4f} "
                f"Train Acc:{train_acc:.4f} "
                f"Val Acc:{val_acc:.4f}"
            )


            print(
                log_line
            )


            log_file.write(
                log_line + "\n"
            )


            # ------------------------------------------------
            # Save best checkpoint
            # ------------------------------------------------

            if val_acc > best_val_acc:

                best_val_acc = val_acc


                torch.save(
                    model.state_dict(),
                    BEST_PATH
                )


                print(
                    "Saved best checkpoint."
                )


        log_file.write(
            f"\nBest Val Acc:{best_val_acc:.4f}\n"
        )


    print(
        f"Best Val Acc: {best_val_acc:.4f}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()