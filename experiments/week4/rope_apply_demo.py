import torch

from models.rope import AxialRoPE2D



def main():

    B = 2
    heads = 8
    grid_size = 8
    tokens = grid_size * grid_size
    dim = 32


    q = torch.randn(
        B,
        heads,
        tokens,
        dim
    )

    k = torch.randn_like(q)


    rope = AxialRoPE2D(
        dim=dim,
        grid_size=grid_size
    )


    q_out, k_out = rope(
        q,
        k
    )


    print("Input Q:")
    print(q.shape)


    print("Output Q:")
    print(q_out.shape)


    print("Input K:")
    print(k.shape)


    print("Output K:")
    print(k_out.shape)



if __name__ == "__main__":
    main()