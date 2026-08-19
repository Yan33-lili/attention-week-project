import torch

from models.window_attention import (
    window_partition,
    window_reverse,
    WindowAttention
)



def main():

    B = 2

    H = 8

    W = 8

    C = 64

    window_size = 4



    x = torch.randn(
        B,
        H,
        W,
        C
    )


    print("Input:")
    print(x.shape)



    windows = window_partition(
        x,
        window_size
    )


    print("\nWindows:")
    print(
        windows.shape
    )


    # flatten window tokens

    windows_flat = windows.reshape(
        -1,
        window_size*window_size,
        C
    )


    attention = WindowAttention(
        dim=C,
        window_size=window_size,
        num_heads=8
    )


    out, attn = attention(
        windows_flat
    )


    print("\nAttention output:")
    print(out.shape)


    print("\nAttention map:")
    print(attn.shape)



if __name__=="__main__":

    main()