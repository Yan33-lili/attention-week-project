import torch
import torch.nn as nn
import torch.nn.functional as F



def window_partition(
    x,
    window_size
):
    """
    Split feature map into windows.

    Input:
        x:
        (B,H,W,C)

    Output:
        windows:
        (num_windows*B,ws,ws,C)

    """

    B,H,W,C = x.shape


    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C
    )


    windows = x.permute(
        0,
        1,
        3,
        2,
        4,
        5
    )


    windows = windows.reshape(
        -1,
        window_size,
        window_size,
        C
    )


    return windows





def window_reverse(
    windows,
    window_size,
    H,
    W
):
    """
    Merge windows back.

    Input:
        windows:
        (num_windows*B,ws,ws,C)

    Output:
        x:
        (B,H,W,C)
    """


    B = int(
        windows.shape[0]
        /
        (
            H*W/window_size/window_size
        )
    )


    x = windows.view(
        B,
        H//window_size,
        W//window_size,
        window_size,
        window_size,
        -1
    )


    x = x.permute(
        0,
        1,
        3,
        2,
        4,
        5
    )


    x = x.reshape(
        B,
        H,
        W,
        -1
    )


    return x






class WindowAttention(nn.Module):
    """
    Self-Attention inside local windows.
    """


    def __init__(
        self,
        dim,
        window_size,
        num_heads
    ):
        super().__init__()


        self.dim = dim

        self.window_size = window_size

        self.num_heads = num_heads


        self.head_dim = dim // num_heads


        self.scale = (
            self.head_dim ** -0.5
        )


        self.qkv = nn.Linear(
            dim,
            dim*3
        )


        self.proj = nn.Linear(
            dim,
            dim
        )



    def forward(
        self,
        x
    ):
        """
        x:

        (B_,N,C)

        N=window_size^2
        """


        B,N,C = x.shape


        qkv = self.qkv(x)


        qkv = qkv.reshape(
            B,
            N,
            3,
            self.num_heads,
            self.head_dim
        )


        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4
        )


        q,k,v = qkv


        attn = (
            q @
            k.transpose(-2,-1)
        )


        attn = attn * self.scale


        attn = F.softmax(
            attn,
            dim=-1
        )


        out = (
            attn @ v
        )


        out = out.transpose(
            1,
            2
        )


        out = out.reshape(
            B,
            N,
            C
        )


        out = self.proj(out)


        return out, attn