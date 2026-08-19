import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(ROOT)
)


import torch
import torch.nn as nn
from torch.optim import Adam


from src.data import get_loaders

from models.patch_lstm import PatchLSTM



EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



SAVE_DIR = (
    ROOT /
    "outputs" /
    "patch_lstm"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BEST_PATH = (
    SAVE_DIR /
    "best.pt"
)


LOG_PATH = (
    SAVE_DIR /
    "train_log.txt"
)



def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss=0
    correct=0
    total=0


    for images,labels,_ in loader:

        images=images.to(DEVICE)
        labels=labels.to(DEVICE)


        optimizer.zero_grad()


        outputs=model(images)


        loss=criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()



        total_loss += loss.item()


        pred=torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            pred==labels
        ).sum().item()


        total += labels.size(0)



    return (
        total_loss/len(loader),
        correct/total
    )




def evaluate(
    model,
    loader
):

    model.eval()


    correct=0
    total=0


    with torch.no_grad():

        for images,labels,_ in loader:

            images=images.to(DEVICE)
            labels=labels.to(DEVICE)


            outputs=model(images)


            pred=torch.argmax(
                outputs,
                dim=1
            )


            correct += (
                pred==labels
            ).sum().item()


            total += labels.size(0)


    return correct/total




def main():

    train_loader,val_loader,_ = get_loaders(
        batch_size=BATCH_SIZE
    )


    model=PatchLSTM()

    model.to(DEVICE)


    criterion=nn.CrossEntropyLoss()

    optimizer=Adam(
        model.parameters(),
        lr=LR
    )


    best_acc=0



    with open(
        LOG_PATH,
        "w"
    ) as f:


        f.write(
            f"Device:{DEVICE}\n"
        )


        for epoch in range(EPOCHS):


            loss,acc=train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )


            val_acc=evaluate(
                model,
                val_loader
            )


            text=(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Train Loss:{loss:.4f} "
                f"Train Acc:{acc:.4f} "
                f"Val Acc:{val_acc:.4f}"
            )


            print(text)

            f.write(text+"\n")



            if val_acc>best_acc:

                best_acc=val_acc

                torch.save(
                    model.state_dict(),
                    BEST_PATH
                )


                print(
                    "Saved best checkpoint."
                )


        f.write(
            f"\nBest Val Acc:{best_acc:.4f}"
        )



if __name__=="__main__":
    main()