from pathlib import Path
import re
import matplotlib.pyplot as plt


# =====================================================
# Path
# =====================================================

ROOT = Path(__file__).resolve().parents[2]


LOG_PATH = (
    ROOT
    / "outputs"
    / "mini_vit_window"
    / "train_log.txt"
)


SAVE_DIR = (
    ROOT
    / "outputs"
    / "figures"
)


SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LOSS_PATH = (
    SAVE_DIR
    / "mini_vit_window_loss_curve.png"
)


ACC_PATH = (
    SAVE_DIR
    / "mini_vit_window_acc_curve.png"
)



# =====================================================
# Read training log
# =====================================================

epochs = []

train_losses = []

train_accs = []

val_accs = []



with open(
    LOG_PATH,
    "r",
    encoding="utf-8"
) as f:

    lines = f.readlines()



for line in lines:


    match = re.search(
        r"Epoch \[(\d+)/\d+\]\s+"
        r"Train Loss:(\d+\.\d+)\s+"
        r"Train Acc:(\d+\.\d+)\s+"
        r"Val Acc:(\d+\.\d+)",
        line
    )


    if match:


        epochs.append(
            int(match.group(1))
        )


        train_losses.append(
            float(match.group(2))
        )


        train_accs.append(
            float(match.group(3))
        )


        val_accs.append(
            float(match.group(4))
        )



print(
    f"Loaded {len(epochs)} epochs"
)



# =====================================================
# Loss Curve
# =====================================================

plt.figure(
    figsize=(8,5)
)


plt.plot(
    epochs,
    train_losses,
    marker="o"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Train Loss"
)


plt.title(
    "MiniViT-Window Training Loss"
)


plt.grid(
    True
)


plt.savefig(
    LOSS_PATH,
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# =====================================================
# Accuracy Curve
# =====================================================

plt.figure(
    figsize=(8,5)
)


plt.plot(
    epochs,
    train_accs,
    marker="o",
    label="Train Accuracy"
)


plt.plot(
    epochs,
    val_accs,
    marker="o",
    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "MiniViT-Window Accuracy"
)


plt.legend()


plt.grid(
    True
)


plt.savefig(
    ACC_PATH,
    dpi=300,
    bbox_inches="tight"
)


plt.close()



print(
    "Saved:"
)

print(
    LOSS_PATH
)

print(
    ACC_PATH
)