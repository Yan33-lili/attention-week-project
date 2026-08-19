import torch

from models.attention import MultiHeadSelfAttention



def main():

    # simulate ViT tokens

    batch_size = 2

    num_patches = 64

    embed_dim = 256


    x = torch.randn(
        batch_size,
        num_patches,
        embed_dim
    )


    print("Input:")
    print(x.shape)



    model = MultiHeadSelfAttention(
        embed_dim=embed_dim,
        num_heads=8
    )


    output, attn = model(x)


    print("\nOutput:")
    print(output.shape)


    print("\nAttention map:")
    print(attn.shape)



if __name__ == "__main__":
    main()