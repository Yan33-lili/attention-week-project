from pathlib import Path
import json

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


from models.position_embedding import (
    LearnedAbsolutePositionEmbedding,
    NoPositionEmbedding
)


from models.attention import MultiHeadSelfAttention



# ===============================
# Config
# ===============================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


B = 2

N = 64

C = 256

HEADS = 8



OUTPUT_DIR = Path(
    "outputs/figures/position_encoding"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



RESULT_PATH = (
    OUTPUT_DIR /
    "position_results.json"
)



FIGURE_PATH = (
    OUTPUT_DIR /
    "position_entropy.png"
)



# ===============================
# Attention entropy
# ===============================

def attention_entropy(attn):

    """
    attn:

    [B,H,N,N]

    """

    eps = 1e-8


    entropy = (
        -attn *
        torch.log(
            attn + eps
        )
    ).sum(
        dim=-1
    )


    entropy = entropy.mean()


    return entropy.item()



# ===============================
# Main
# ===============================

def main():


    print(
        "Device:",
        DEVICE
    )


    x = torch.randn(
        B,
        N,
        C,
        device=DEVICE
    )



    attention = MultiHeadSelfAttention(
        dim=C,
        num_heads=HEADS
    ).to(DEVICE)



    methods = {}


    # ---------------------------
    # No Position
    # ---------------------------

    methods["No Position"] = (
        NoPositionEmbedding()
    )



    # ---------------------------
    # Learned Absolute PE
    # ---------------------------

    methods["Learned Absolute"] = (
        LearnedAbsolutePositionEmbedding(
            N,
            C
        ).to(DEVICE)
    )



    results = {}



    for name, pos in methods.items():


        print(
            "\nRunning:",
            name
        )


        x_pos = pos(x)


        output, attn_map = attention(
            x_pos,
            return_attention=True
        )


        entropy = attention_entropy(
            attn_map
        )


        results[name] = {

            "output_mean":
                output.mean().item(),

            "output_std":
                output.std().item(),

            "attention_entropy":
                entropy
        }


        print(
            results[name]
        )



    # ---------------------------
    # Save
    # ---------------------------

    with open(
        RESULT_PATH,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )



    # ---------------------------
    # Plot
    # ---------------------------


    names = list(
        results.keys()
    )


    entropy = [
        results[n]["attention_entropy"]
        for n in names
    ]


    plt.figure(
        figsize=(7,4)
    )


    plt.bar(
        names,
        entropy
    )


    plt.ylabel(
        "Attention Entropy"
    )


    plt.title(
        "Effect of Position Encoding"
    )


    plt.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()



    print(
        "\nSaved:"
    )

    print(
        RESULT_PATH
    )



if __name__=="__main__":

    main()