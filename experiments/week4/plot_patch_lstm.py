import re
from pathlib import Path
import matplotlib.pyplot as plt


# =========================
# Path config
# =========================

ROOT = Path(__file__).resolve().parents[2]


LOG_PATH = (
    ROOT
    / "outputs"
    / "patch_lstm"
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


LOSS_SAVE = (
    SAVE_DIR
    / "patch_lstm_loss_curve.png"
)

ACC_SAVE = (
    SAVE_DIR
    / "patch_lstm_acc_curve.png"
)


# =========================
# Read log
# =========================

epochs = []
train_losses = []
train_accs = []
val_accs = []


with open(LOG_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()


for line in lines:

    # example:
    # Epoch [1/20] Train Loss:2.xxxx Train Acc:0.xxxx Val Acc:0.xxxx

    match = re.search(
        r"Epoch \[(\d+)/\d+\].*Train Loss:(\d+\.\d+).*Train Acc:(\d+\.\d+).*Val Acc:(\d+\.\d+)",
        line
    )

    if match:

        epoch = int(match.group(1))
        loss = float(match.group(2))
        train_acc = float(match.group(3))
        val_acc = float(match.group(4))


        epochs.append(epoch)
        train_losses.append(loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)



print("Loaded epochs:", len(epochs))


# =========================
# Loss curve
# =========================

plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    train_losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Train Loss")

plt.title(
    "Patch-LSTM Training Loss"
)

plt.grid(True)

plt.savefig(
    LOSS_SAVE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# =========================
# Accuracy curve
# =========================

plt.figure(figsize=(8,5))


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


plt.xlabel("Epoch")

plt.ylabel("Accuracy")


plt.title(
    "Patch-LSTM Accuracy"
)


plt.legend()

plt.grid(True)


plt.savefig(
    ACC_SAVE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()



print("Saved:")
print(LOSS_SAVE)
print(ACC_SAVE)