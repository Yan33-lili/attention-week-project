import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.data import get_loaders
from models.mini_vit_full import MiniViTFull


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Configuration
# ============================================================

SEED = 42

EPOCHS = 20

BATCH_SIZE = 64

LR = 3e-4

WEIGHT_DECAY = 0.05

IMG_SIZE = 64

PATCH_SIZE = 8

IN_CHANNELS = 3

EMBED_DIM = 192

NUM_HEADS = 4

DEPTH = 4

MLP_RATIO = 4.0

DROPOUT = 0.1

NUM_CLASSES = 100


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Paths
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "positional_ablation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULT_PATH = (
    OUTPUT_DIR
    / "results.json"
)


PARAM_CHECK_PATH = (
    OUTPUT_DIR
    / "parameter_check.json"
)


ABS_CHECKPOINT = (
    ROOT
    / "outputs"
    / "mini_vit_full"
    / "best.pt"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior.
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# Model factory
# ============================================================

def create_model(position_mode):

    model = MiniViTFull(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
        position_mode=position_mode
    )

    return model.to(DEVICE)


# ============================================================
# Parameter utilities
# ============================================================

def parameter_dict(model):

    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
    }


def compare_shared_parameters(
    model_a,
    model_b
):

    params_a = parameter_dict(
        model_a
    )

    params_b = parameter_dict(
        model_b
    )

    common_names = sorted(
        set(params_a.keys())
        & set(params_b.keys())
    )

    mismatches = []

    max_abs_diff = 0.0

    for name in common_names:

        a = params_a[name]

        b = params_b[name]

        if a.shape != b.shape:

            mismatches.append(
                {
                    "name": name,
                    "reason": "shape mismatch",
                    "shape_a": list(a.shape),
                    "shape_b": list(b.shape)
                }
            )

            continue

        diff = (
            a - b
        ).abs().max().item()

        max_abs_diff = max(
            max_abs_diff,
            diff
        )

        if not torch.equal(
            a,
            b
        ):

            mismatches.append(
                {
                    "name": name,
                    "reason": "value mismatch",
                    "max_abs_diff": diff
                }
            )

    return {
        "common_parameter_count": len(
            common_names
        ),
        "mismatch_count": len(
            mismatches
        ),
        "max_abs_diff": max_abs_diff,
        "mismatches": mismatches
    }


# ============================================================
# Train
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
# Evaluation
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
# Train one variant
# ============================================================

def train_variant(
    model,
    train_loader,
    val_loader,
    variant_name
):

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_acc = 0.0

    best_state = None

    history = []

    print()
    print(
        "=" * 60
    )

    print(
        f"Training variant: {variant_name}"
    )

    print(
        "=" * 60
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

        log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        }

        history.append(
            log
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return {
        "best_val_acc": best_val_acc,
        "history": history
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "Task 7: Position Encoding Ablation"
    )

    print(
        "=" * 60
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Seed: {SEED}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print()


    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not ABS_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Absolute-position checkpoint not found:\n"
            f"{ABS_CHECKPOINT}"
        )


    # --------------------------------------------------------
    # Seed before data loader creation
    # --------------------------------------------------------

    set_seed(
        SEED
    )


    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, val_loader, test_loader = (
        get_loaders(
            batch_size=BATCH_SIZE
        )
    )


    # --------------------------------------------------------
    # Create models from identical seed
    # --------------------------------------------------------

    set_seed(
        SEED
    )

    no_pos_model = create_model(
        "none"
    )


    set_seed(
        SEED
    )

    abs_pos_model = create_model(
        "absolute"
    )


    set_seed(
        SEED
    )

    rope_model = create_model(
        "rope2d"
    )


    # --------------------------------------------------------
    # Parameter consistency check
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "Parameter consistency check"
    )

    print(
        "=" * 60
    )


    no_vs_rope = compare_shared_parameters(
        no_pos_model,
        rope_model
    )


    no_vs_abs = compare_shared_parameters(
        no_pos_model,
        abs_pos_model
    )


    parameter_check = {
        "seed": SEED,
        "no_pos_vs_rope2d": no_vs_rope,
        "no_pos_vs_absolute": no_vs_abs
    }


    with open(
        PARAM_CHECK_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            parameter_check,
            f,
            indent=4
        )


    print(
        json.dumps(
            parameter_check,
            indent=2
        )
    )


    # --------------------------------------------------------
    # Train NoPos
    # --------------------------------------------------------

    no_pos_result = train_variant(
        no_pos_model,
        train_loader,
        val_loader,
        "NoPos"
    )


    # --------------------------------------------------------
    # Train RoPE2D
    # --------------------------------------------------------

    rope_result = train_variant(
        rope_model,
        train_loader,
        val_loader,
        "RoPE2D"
    )


    # --------------------------------------------------------
    # Load existing AbsPos checkpoint
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "Loading existing AbsPos checkpoint"
    )

    print(
        ABS_CHECKPOINT
    )

    print(
        "=" * 60
    )


    checkpoint = torch.load(
        ABS_CHECKPOINT,
        map_location=DEVICE
    )


    # The existing checkpoint belongs to
    # the original PyTorch TransformerEncoder
    # implementation.
    #
    # Therefore, it cannot be loaded directly
    # into the hand-written TransformerBlock model.
    #
    # We intentionally evaluate the original
    # MiniViTFull checkpoint separately below.

    from models.mini_vit_full import MiniViTFull as OriginalMiniViTFull

    abs_model = OriginalMiniViTFull(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES
    ).to(DEVICE)


    abs_model.load_state_dict(
        checkpoint
    )


    abs_model.eval()


    abs_test_acc = evaluate(
        abs_model,
        test_loader
    )


    # --------------------------------------------------------
    # Test NoPos
    # --------------------------------------------------------

    no_pos_test_acc = evaluate(
        no_pos_model,
        test_loader
    )


    # --------------------------------------------------------
    # Test RoPE2D
    # --------------------------------------------------------

    rope_test_acc = evaluate(
        rope_model,
        test_loader
    )


    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    results = {

        "task": "Task 7 Position Encoding Ablation",

        "device": DEVICE,

        "seed": SEED,

        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "img_size": IMG_SIZE,
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "depth": DEPTH,
            "mlp_ratio": MLP_RATIO,
            "dropout": DROPOUT,
            "num_classes": NUM_CLASSES,
            "grid_size": [
                IMG_SIZE // PATCH_SIZE,
                IMG_SIZE // PATCH_SIZE
            ]
        },

        "variants": {

            "NoPos": {
                "best_val_acc":
                    no_pos_result[
                        "best_val_acc"
                    ],
                "test_acc":
                    no_pos_test_acc,
                "history":
                    no_pos_result[
                        "history"
                    ]
            },

            "AbsPos": {
                "checkpoint":
                    str(
                        ABS_CHECKPOINT
                    ),
                "test_acc":
                    abs_test_acc
            },

            "RoPE2D": {
                "best_val_acc":
                    rope_result[
                        "best_val_acc"
                    ],
                "test_acc":
                    rope_test_acc,
                "history":
                    rope_result[
                        "history"
                    ]
            }
        }
    }


    # --------------------------------------------------------
    # Accuracy differences
    # --------------------------------------------------------

    results["comparison"] = {

        "NoPos_minus_AbsPos":
            no_pos_test_acc
            - abs_test_acc,

        "RoPE2D_minus_AbsPos":
            rope_test_acc
            - abs_test_acc,

        "RoPE2D_minus_NoPos":
            rope_test_acc
            - no_pos_test_acc
    }


    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )


    # --------------------------------------------------------
    # Print final table
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"{'Variant':<15}"
        f"{'Test Acc':>15}"
    )

    print(
        "-" * 30
    )

    print(
        f"{'NoPos':<15}"
        f"{no_pos_test_acc:>15.4f}"
    )

    print(
        f"{'AbsPos':<15}"
        f"{abs_test_acc:>15.4f}"
    )

    print(
        f"{'RoPE2D':<15}"
        f"{rope_test_acc:>15.4f}"
    )

    print()

    print(
        f"Results saved to:\n"
        f"{RESULT_PATH}"
    )

    print()

    print(
        f"Parameter check saved to:\n"
        f"{PARAM_CHECK_PATH}"
    )


if __name__ == "__main__":

    main()