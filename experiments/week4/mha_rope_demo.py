import torch

from models.attention import MultiHeadSelfAttention



def main():

    B = 2

    patches = 64

    embed_dim = 256


    x = torch.randn(
        B,
        patches,
        embed_dim
    )


    model = MultiHeadSelfAttention(
        embed_dim=embed_dim,
        num_heads=8,
        use_rope=True,
        grid_size=8
    )


    output, attn = model(x)


    print("Input:")
    print(x.shape)


    print("\nOutput:")
    print(output.shape)


    print("\nAttention map:")
    print(attn.shape)



if __name__ == "__main__":
    main()