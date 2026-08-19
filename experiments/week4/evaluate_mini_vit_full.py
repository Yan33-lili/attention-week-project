import sys
from pathlib import Path

import torch
import torch.nn as nn


# =====================================================
# Project root
# =====================================================

ROOT = Path(__file__).resolve().parents[2]


if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT)
    )


# =====================================================
# Import
# =====================================================

from src.data import get_loaders
from models.mini_vit_full import MiniViTFull



# =====================================================
# Config
# =====================================================

BATCH_SIZE = 64

NUM_CLASSES = 100


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



CHECKPOINT_PATH = (
    ROOT
    / "outputs"
    / "mini_vit_full"
    / "best.pt"
)


RESULT_DIR = (
    ROOT
    / "outputs"
    / "metrics"
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULT_PATH = (
    RESULT_DIR
    / "mini_vit_full_test_result.txt"
)



# =====================================================
# Evaluation
# =====================================================

def evaluate():

    print(
        f"Device: {DEVICE}"
    )


    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )


    # -------------------------------------------------
    # Check checkpoint
    # -------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )


    # -------------------------------------------------
    # Load test data
    # -------------------------------------------------

    train_loader, val_loader, test_loader = get_loaders(
        batch_size=BATCH_SIZE
    )


    # -------------------------------------------------
    # Create model
    # -------------------------------------------------

    model = MiniViTFull(
        img_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=192,
        num_heads=4,
        depth=4,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=NUM_CLASSES
    )


    model = model.to(
        DEVICE
    )


    # -------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )


    # Your training saves:
    #
    # torch.save(model.state_dict(), path)
    #

    model.load_state_dict(
        checkpoint
    )


    model.eval()


    # -------------------------------------------------
    # Loss
    # -------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    total_loss = 0.0

    correct = 0

    total = 0



    # -------------------------------------------------
    # Test
    # -------------------------------------------------

    with torch.no_grad():

        for images, labels, _ in test_loader:


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


            loss = criterion(
                logits,
                labels
            )


            batch_size = labels.size(0)


            total_loss += (
                loss.item()
                *
                batch_size
            )


            predictions = torch.argmax(
                logits,
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += batch_size



    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    test_loss = (
        total_loss / total
    )


    test_accuracy = (
        correct / total
    )


    accuracy_percentage = (
        test_accuracy * 100
    )



    # -------------------------------------------------
    # Print
    # -------------------------------------------------

    print()

    print(
        "MiniViT-Full Test Evaluation"
    )

    print()


    print(
        "Device:"
    )

    print(
        DEVICE
    )

    print()


    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print()


    print(
        "Test Loss:"
    )

    print(
        f"{test_loss:.4f}"
    )

    print()


    print(
        "Test Accuracy:"
    )

    print(
        f"{test_accuracy:.4f}"
    )

    print()


    print(
        "Accuracy Percentage:"
    )

    print(
        f"{accuracy_percentage:.2f}%"
    )


    # -------------------------------------------------
    # Save result
    # -------------------------------------------------

    result_text = (
        "MiniViT-Full Test Evaluation\n\n"

        f"Device:\n"
        f"{DEVICE}\n\n"

        f"Checkpoint:\n"
        f"{CHECKPOINT_PATH}\n\n"

        f"Test Loss:\n"
        f"{test_loss:.4f}\n\n"

        f"Test Accuracy:\n"
        f"{test_accuracy:.4f}\n\n"

        f"Accuracy Percentage:\n"
        f"{accuracy_percentage:.2f}%\n"
    )


    RESULT_PATH.write_text(
        result_text,
        encoding="utf-8"
    )


    print()

    print(
        "Saved result:"
    )

    print(
        RESULT_PATH
    )



# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    evaluate()