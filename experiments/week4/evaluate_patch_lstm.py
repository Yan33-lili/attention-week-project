import sys
from pathlib import Path

import torch
import torch.nn as nn


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Imports
# ============================================================

from src.data import get_loaders
from models.patch_lstm import PatchLSTM


# ============================================================
# Config
# ============================================================

BATCH_SIZE = 64

NUM_CLASSES = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


CHECKPOINT_PATH = (
    ROOT
    / "outputs"
    / "patch_lstm"
    / "best.pt"
)


METRICS_DIR = (
    ROOT
    / "outputs"
    / "metrics"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULT_PATH = (
    METRICS_DIR
    / "patch_lstm_test_result.txt"
)


# ============================================================
# Evaluation
# ============================================================

def evaluate():

    print(f"Device: {DEVICE}")

    print(f"Checkpoint:")
    print(CHECKPOINT_PATH)

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Load test loader
    # --------------------------------------------------------

    loaders = get_loaders(
        batch_size=BATCH_SIZE
    )

    # get_loaders() returns:
    # train_loader, val_loader, test_loader

    train_loader, val_loader, test_loader = loaders

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = PatchLSTM(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    # Support both:
    #
    # 1. torch.save(model.state_dict(), path)
    #
    # 2. torch.save(
    #        {"model_state_dict": model.state_dict()},
    #        path
    #    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            # Assume the dictionary itself is state_dict
            state_dict = checkpoint

    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict
    )

    model.eval()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    with torch.no_grad():

        for batch in test_loader:

            # Your Tiny ImageNet pipeline returns:
            #
            # images
            # labels
            # rgb_mean
            #
            # We only need images and labels here.

            images = batch[0]
            labels = batch[1]

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            # Forward
            logits = model(images)

            # Loss
            loss = criterion(
                logits,
                labels
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            # Prediction
            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    test_loss = (
        total_loss / total
    )

    test_accuracy = (
        correct / total
    )

    accuracy_percentage = (
        test_accuracy * 100
    )


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("Patch-LSTM Test Evaluation")
    print()
    print("Device:")
    print(DEVICE)
    print()

    print("Checkpoint:")
    print(CHECKPOINT_PATH)
    print()

    print("Test Loss:")
    print(f"{test_loss:.4f}")
    print()

    print("Test Accuracy:")
    print(f"{test_accuracy:.4f}")
    print()

    print("Accuracy Percentage:")
    print(f"{accuracy_percentage:.2f}%")
    print()


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result_text = (
        "Patch-LSTM Test Evaluation\n\n"

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


    print("Saved result:")
    print(RESULT_PATH)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    evaluate()