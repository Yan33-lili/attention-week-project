import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt



def compute_attention(
    Q,
    K,
    scaled=True
):
    """
    Compute attention scores.

    Q:
    (N,d)

    K:
    (N,d)
    """


    scores = Q @ K.T


    if scaled:

        d_k = Q.shape[-1]

        scores = scores / (
            d_k ** 0.5
        )


    attention = F.softmax(
        scores,
        dim=-1
    )


    return scores, attention





def entropy(prob):

    """
    Attention entropy

    H=-sum(p log p)
    """

    eps = 1e-9


    return (
        -(
            prob *
            torch.log(
                prob + eps
            )
        )
        .sum(dim=-1)
    )





def main():

    torch.manual_seed(42)


    N = 64

    d_k = 64


    # random Q,K

    Q = torch.randn(
        N,
        d_k
    )


    K = torch.randn(
        N,
        d_k
    )


    # unscaled

    score_unscaled, attn_unscaled = compute_attention(
        Q,
        K,
        scaled=False
    )


    # scaled

    score_scaled, attn_scaled = compute_attention(
        Q,
        K,
        scaled=True
    )



    print(
        "Unscaled score std:",
        score_unscaled.std().item()
    )


    print(
        "Scaled score std:",
        score_scaled.std().item()
    )



    print(
        "Unscaled entropy:",
        entropy(attn_unscaled).mean().item()
    )


    print(
        "Scaled entropy:",
        entropy(attn_scaled).mean().item()
    )



    # -------------------------
    # Figure 1
    # Score distribution
    # -------------------------

    plt.figure(figsize=(8,5))


    plt.hist(
        score_unscaled.flatten().numpy(),
        bins=100,
        alpha=0.5,
        label="Unscaled"
    )


    plt.hist(
        score_scaled.flatten().numpy(),
        bins=100,
        alpha=0.5,
        label="Scaled"
    )


    plt.legend()

    plt.title(
        "Attention Score Distribution"
    )

    plt.xlabel(
        "score"
    )

    plt.ylabel(
        "frequency"
    )


    plt.savefig(
        "outputs/figures/score_distribution.png"
    )

    plt.close()



    # -------------------------
    # Figure 2
    # Attention distribution
    # -------------------------

    plt.figure(figsize=(8,5))


    plt.hist(
        attn_unscaled.flatten().detach().numpy(),
        bins=100,
        alpha=0.5,
        label="Unscaled"
    )


    plt.hist(
        attn_scaled.flatten().detach().numpy(),
        bins=100,
        alpha=0.5,
        label="Scaled"
    )


    plt.legend()


    plt.title(
        "Attention Weight Distribution"
    )


    plt.xlabel(
        "attention weight"
    )


    plt.ylabel(
        "frequency"
    )


    plt.savefig(
        "outputs/figures/attention_distribution.png"
    )

    plt.close()



    # -------------------------
    # Figure 3
    # Entropy comparison
    # -------------------------

    entropy_values = [
        entropy(attn_unscaled).mean().item(),
        entropy(attn_scaled).mean().item()
    ]


    plt.figure(figsize=(5,5))


    plt.bar(
        [
            "Unscaled",
            "Scaled"
        ],
        entropy_values
    )


    plt.title(
        "Attention Entropy Comparison"
    )


    plt.ylabel(
        "Entropy"
    )


    plt.savefig(
        "outputs/figures/attention_entropy.png"
    )

    plt.close()





if __name__ == "__main__":

    main()