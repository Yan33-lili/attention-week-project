import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    Simple CNN baseline for Tiny ImageNet-200

    Input:
        (B, 3, 64, 64)

    Output:
        (B, 100)
    """

    def __init__(self, num_classes=100):
        super().__init__()

        self.features = nn.Sequential(

            # block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),
            # 64x64 -> 32x32


            # block 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),
            # 32x32 -> 16x16


            # block 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2)
            # 16x16 -> 8x8
        )


        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(),

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x



if __name__ == "__main__":

    model = SmallCNN()

    x = torch.randn(
        4,
        3,
        64,
        64
    )

    y = model(x)

    print(y.shape)