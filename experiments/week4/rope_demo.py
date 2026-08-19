import torch

from models.rope import AxialRoPE2D



def main():

    # Tiny ImageNet example:
    # 8x8 patch grid

    rope = AxialRoPE2D(
        dim=256,
        grid_size=8
    )


    positions = rope()


    print(
        "Number of patch positions:"
    )

    print(
        len(positions)
    )


    print(
        "First 10 positions:"
    )

    print(
        positions[:10]
    )


if __name__=="__main__":

    main()