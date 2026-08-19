import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


import torch
import torch.nn as nn
from torch.optim import Adam

from src.data import get_loaders
from models.small_cnn import SmallCNN



# ======================
# config
# ======================

EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


SAVE_PATH = (
    ROOT
    /
    "outputs"
    /
    "smallcnn"
    /
    "best.pt"
)


SAVE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)



# ======================
# train one epoch
# ======================

def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0


    for images, labels, _ in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        total_loss += loss.item()


        preds = outputs.argmax(
            dim=1
        )

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)



    return (
        total_loss / len(loader),
        correct / total
    )



# ======================
# validation
# ======================

def evaluate(
        model,
        loader,
        criterion):

    model.eval()


    total_loss = 0
    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels, _ in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            total_loss += loss.item()


            preds = outputs.argmax(
                dim=1
            )


            correct += (
                preds == labels
            ).sum().item()


            total += labels.size(0)



    return (
        total_loss / len(loader),
        correct / total
    )



# ======================
# main
# ======================

def main():

    print(
        "Device:",
        DEVICE
    )


    train_loader, val_loader, test_loader = get_loaders(
        batch_size=BATCH_SIZE
    )


    model = SmallCNN(
        num_classes=100
    ).to(DEVICE)



    criterion = nn.CrossEntropyLoss()


    optimizer = Adam(
        model.parameters(),
        lr=LR
    )


    best_acc = 0



    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )


        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion
        )


        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss:{train_loss:.4f} "
            f"Train Acc:{train_acc:.4f} "
            f"Val Acc:{val_acc:.4f}"
        )


        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                SAVE_PATH
            )


            print(
                "Saved best checkpoint."
            )



    print(
        "Best Val Acc:",
        best_acc
    )



if __name__ == "__main__":
    main()