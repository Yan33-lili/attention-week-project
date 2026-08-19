import re
from pathlib import Path
import matplotlib.pyplot as plt


# =====================================================
# Path
# =====================================================

ROOT = Path(__file__).resolve().parents[2]


LOG_PATH = (
    ROOT
    / "outputs"
    / "mini_vit_full"
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
    / "mini_vit_full_loss_curve.png"
)


ACC_PATH = (
    SAVE_DIR
    / "mini_vit_full_acc_curve.png"
)


# =====================================================
# Read log
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

    # Match:
    #
    # Epoch [1/20] Train Loss:4.0906 Train Acc:0.0762 Val Acc:0.1188

    match = re.search(
        r"Epoch \[(\d+)/\d+\]\s+"
        r"Train Loss:(\d+\.\d+)\s+"
        r"Train Acc:(\d+\.\d+)\s+"
        r"Val Acc:(\d+\.\d+)",
        line
    )


    if match:

        epoch = int(
            match.group(1)
        )

        loss = float(
            match.group(2)
        )

        train_acc = float(
            match.group(3)
        )

        val_acc = float(
            match.group(4)
        )


        epochs.append(epoch)
        train_losses.append(loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)



print(
    f"Loaded {len(epochs)} epochs"
)



# =====================================================
# Loss curve
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
    "MiniViT-Full Training Loss"
)


plt.grid(True)


plt.savefig(
    LOSS_PATH,
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# =====================================================
# Accuracy curve
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
    "MiniViT-Full Accuracy"
)


plt.legend()

plt.grid(True)


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