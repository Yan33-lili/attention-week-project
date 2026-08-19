import sys
from pathlib import Path

# add project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


import torch
import torch.nn as nn

from src.data import get_loaders
from models.small_cnn import SmallCNN


# ======================
# config
# ======================

BATCH_SIZE = 64

CHECKPOINT = (
    ROOT
    / "outputs"
    / "smallcnn"
    / "best.pt"
)

SAVE_PATH = (
    ROOT
    / "outputs"
    / "metrics"
    / "smallcnn_test_result.txt"
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# evaluate
# ======================

def evaluate():

    _, _, test_loader = get_loaders(
        batch_size=BATCH_SIZE
    )


    model = SmallCNN(
        num_classes=100
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(DEVICE)
    model.eval()


    criterion = nn.CrossEntropyLoss()


    total_loss = 0
    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels, _ in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)


            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


            total_loss += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()



    test_loss = total_loss / len(test_loader)

    test_acc = (
        correct / total
    )



    # ======================
    # save result
    # ======================

    SAVE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    result = f"""
SmallCNN Test Evaluation

Device:
{DEVICE}

Checkpoint:
{CHECKPOINT}

Test Loss:
{test_loss:.4f}

Test Accuracy:
{test_acc:.4f}

Accuracy Percentage:
{test_acc*100:.2f}%

"""


    print(result)


    with open(
        SAVE_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(result)



if __name__ == "__main__":
    evaluate()