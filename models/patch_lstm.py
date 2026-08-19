import torch
import torch.nn as nn


class PatchLSTM(nn.Module):

    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=3,
        hidden_dim=256,
        num_layers=2,
        num_classes=100
    ):
        super().__init__()


        self.patch_size = patch_size


        # patch数量
        self.num_patches = (
            img_size // patch_size
        ) ** 2


        # 每个patch展开后的维度
        patch_dim = (
            patch_size *
            patch_size *
            in_channels
        )


        self.lstm = nn.LSTM(
            input_size=patch_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )


        self.classifier = nn.Linear(
            hidden_dim,
            num_classes
        )



    def forward(self, x):

        B,C,H,W = x.shape


        # unfold image into patches
        patches = x.unfold(
            2,
            self.patch_size,
            self.patch_size
        )

        patches = patches.unfold(
            3,
            self.patch_size,
            self.patch_size
        )


        # B,C,8,8,8,8

        patches = patches.permute(
            0,2,3,1,4,5
        )


        # B,8,8,C,8,8


        patches = patches.reshape(
            B,
            self.num_patches,
            -1
        )


        # B,64,192


        output,(h,c)=self.lstm(
            patches
        )


        # use last hidden state

        feature = h[-1]


        logits = self.classifier(
            feature
        )


        return logits