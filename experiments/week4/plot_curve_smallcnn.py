from pathlib import Path
import re
import matplotlib.pyplot as plt


# ======================
# paths
# ======================

ROOT = Path(__file__).resolve().parents[2]

LOG_PATH = ROOT / "outputs" / "smallcnn" / "train_log.txt"

SAVE_DIR = ROOT / "outputs" / "figures"

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================
# read log
# ======================

epochs = []
train_losses = []
train_accs = []
val_accs = []


with open(LOG_PATH, "r", encoding="utf-8") as f:

    for line in f:

        if "Epoch" in line:

            result = re.search(
                r"Epoch \[(\d+)/\d+\] Train Loss:(\d+\.\d+) Train Acc:(\d+\.\d+) Val Acc:(\d+\.\d+)",
                line
            )

            if result:

                epoch = int(result.group(1))
                loss = float(result.group(2))
                train_acc = float(result.group(3))
                val_acc = float(result.group(4))


                epochs.append(epoch)
                train_losses.append(loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)



print("Loaded epochs:", len(epochs))


# ======================
# loss curve
# ======================

plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    train_losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("SmallCNN Training Loss")

plt.grid()

plt.savefig(
    SAVE_DIR / "smallcnn_loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# ======================
# accuracy curve
# ======================

plt.figure(figsize=(8,5))


plt.plot(
    epochs,
    train_accs,
    marker="o",
    label="Train Acc"
)


plt.plot(
    epochs,
    val_accs,
    marker="o",
    label="Val Acc"
)


plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("SmallCNN Accuracy")

plt.legend()

plt.grid()


plt.savefig(
    SAVE_DIR / "smallcnn_acc_curve.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("Curves saved.")