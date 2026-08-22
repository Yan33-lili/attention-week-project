import torch
import torch.nn as nn


class LearnedAbsolutePositionEmbedding(nn.Module):
    """
    Learned Absolute Position Embedding

    P:
        [1, N, C]
    """

    def __init__(
        self,
        num_tokens,
        dim
    ):
        super().__init__()

        self.position = nn.Parameter(
            torch.zeros(
                1,
                num_tokens,
                dim
            )
        )


        nn.init.trunc_normal_(
            self.position,
            std=0.02
        )


    def forward(self, x):

        return x + self.position



class NoPositionEmbedding(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self,x):

        return x