from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import torch


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "attention_scaling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUTPUT_DIR / "attention_scaling_results.json"
FIGURE_PATH = OUTPUT_DIR / "attention_scaling_three_panel.png"


# ============================================================
# Experiment configuration
# ============================================================

SEED = 42

# Different key/query dimensions
D_K_VALUES = [16, 32, 64, 128, 256, 512]

# Number of query/key positions
SEQ_LEN = 64

# Number of random trials for each d_k
NUM_TRIALS = 20

# Small epsilon for finite-difference Jacobian validation
FD_EPS = 1e-5


# ============================================================
# Attention score
# ============================================================

def compute_scores(q, k, scaled):
    """
    Compute scaled or unscaled attention scores.

    q:
        [L, d_k]

    k:
        [L, d_k]

    return:
        [L, L]
    """

    scores = q @ k.transpose(-2, -1)

    if scaled:
        d_k = q.shape[-1]
        scores = scores / math.sqrt(d_k)

    return scores


# ============================================================
# Normalized entropy
# ============================================================

def normalized_entropy(probabilities):
    """
    Compute normalized entropy.

    H(p) = -sum p log(p)

    H_norm = H(p) / log(L)

    probabilities:
        [..., L]
    """

    eps = torch.finfo(probabilities.dtype).eps

    entropy = -(
        probabilities *
        torch.log(probabilities.clamp_min(eps))
    ).sum(dim=-1)

    length = probabilities.shape[-1]

    return entropy / math.log(length)


# ============================================================
# Softmax Jacobian
# ============================================================

def softmax_jacobian(probabilities):
    """
    Analytical softmax Jacobian:

        J(p) = diag(p) - p p^T

    probabilities:
        [L]

    return:
        [L, L]
    """

    p = probabilities

    diagonal = torch.diag(p)

    outer_product = torch.outer(p, p)

    return diagonal - outer_product


# ============================================================
# Finite-difference Jacobian
# ============================================================

def finite_difference_jacobian(logits, epsilon=1e-5):
    """
    Numerically approximate the Jacobian of softmax.

    For each input dimension i:

        J[:, i]
        ≈
        [softmax(s + eps*e_i) - softmax(s)] / eps

    logits:
        [L]

    return:
        [L, L]
    """

    base_probability = torch.softmax(logits, dim=-1)

    length = logits.numel()

    jacobian = torch.zeros(
        length,
        length,
        dtype=logits.dtype,
        device=logits.device,
    )

    for i in range(length):

        perturbation = torch.zeros_like(logits)

        perturbation[i] = epsilon

        perturbed_probability = torch.softmax(
            logits + perturbation,
            dim=-1,
        )

        jacobian[:, i] = (
            perturbed_probability -
            base_probability
        ) / epsilon

    return jacobian


# ============================================================
# Finite-difference validation
# ============================================================

def validate_jacobian():

    torch.manual_seed(SEED)

    # Small vector for easy numerical verification
    logits = torch.randn(
        8,
        dtype=torch.float64,
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    analytical = softmax_jacobian(
        probabilities
    )

    numerical = finite_difference_jacobian(
        logits,
        epsilon=FD_EPS,
    )

    absolute_error = (
        analytical - numerical
    ).abs()

    max_error = absolute_error.max().item()

    mean_error = absolute_error.mean().item()

    frobenius_error = torch.linalg.matrix_norm(
        analytical - numerical,
        ord="fro",
    ).item()

    print("\nFinite-difference Jacobian validation")
    print("-------------------------------------")

    print(
        f"Max absolute error: "
        f"{max_error:.8e}"
    )

    print(
        f"Mean absolute error: "
        f"{mean_error:.8e}"
    )

    print(
        f"Frobenius error: "
        f"{frobenius_error:.8e}"
    )

    return {
        "max_absolute_error": max_error,
        "mean_absolute_error": mean_error,
        "frobenius_error": frobenius_error,
    }


# ============================================================
# Run experiment for one d_k
# ============================================================

def run_single_dimension(d_k):
    """
    Run scaled/unscaled comparison for one d_k.
    """

    unscaled_score_stds = []
    scaled_score_stds = []

    unscaled_entropies = []
    scaled_entropies = []

    unscaled_jacobian_norms = []
    scaled_jacobian_norms = []

    for _ in range(NUM_TRIALS):

        q = torch.randn(
            SEQ_LEN,
            d_k,
        )

        k = torch.randn(
            SEQ_LEN,
            d_k,
        )

        # ----------------------------------------------------
        # Unscaled
        # ----------------------------------------------------

        unscaled_scores = compute_scores(
            q,
            k,
            scaled=False,
        )

        unscaled_probabilities = torch.softmax(
            unscaled_scores,
            dim=-1,
        )

        unscaled_score_stds.append(
            unscaled_scores.std().item()
        )

        unscaled_entropies.append(
            normalized_entropy(
                unscaled_probabilities
            ).mean().item()
        )

        # Jacobian is computed row by row.
        unscaled_norms = []

        # Use only the first few rows to keep the
        # numerical experiment lightweight.
        num_rows_for_jacobian = min(
            8,
            SEQ_LEN,
        )

        for row in unscaled_probabilities[
            :num_rows_for_jacobian
        ]:

            jacobian = softmax_jacobian(row)

            norm = torch.linalg.matrix_norm(
                jacobian,
                ord="fro",
            )

            unscaled_norms.append(
                norm.item()
            )

        unscaled_jacobian_norms.append(
            sum(unscaled_norms) /
            len(unscaled_norms)
        )

        # ----------------------------------------------------
        # Scaled
        # ----------------------------------------------------

        scaled_scores = compute_scores(
            q,
            k,
            scaled=True,
        )

        scaled_probabilities = torch.softmax(
            scaled_scores,
            dim=-1,
        )

        scaled_score_stds.append(
            scaled_scores.std().item()
        )

        scaled_entropies.append(
            normalized_entropy(
                scaled_probabilities
            ).mean().item()
        )

        scaled_norms = []

        for row in scaled_probabilities[
            :num_rows_for_jacobian
        ]:

            jacobian = softmax_jacobian(row)

            norm = torch.linalg.matrix_norm(
                jacobian,
                ord="fro",
            )

            scaled_norms.append(
                norm.item()
            )

        scaled_jacobian_norms.append(
            sum(scaled_norms) /
            len(scaled_norms)
        )

    return {
        "d_k": d_k,

        "unscaled": {
            "score_std": (
                sum(unscaled_score_stds) /
                len(unscaled_score_stds)
            ),

            "normalized_entropy": (
                sum(unscaled_entropies) /
                len(unscaled_entropies)
            ),

            "jacobian_frobenius_norm": (
                sum(unscaled_jacobian_norms) /
                len(unscaled_jacobian_norms)
            ),
        },

        "scaled": {
            "score_std": (
                sum(scaled_score_stds) /
                len(scaled_score_stds)
            ),

            "normalized_entropy": (
                sum(scaled_entropies) /
                len(scaled_entropies)
            ),

            "jacobian_frobenius_norm": (
                sum(scaled_jacobian_norms) /
                len(scaled_jacobian_norms)
            ),
        },
    }


# ============================================================
# Plot
# ============================================================

def plot_results(results):
    """
    Generate the required three-panel figure:

    1. Score standard deviation
    2. Normalized entropy
    3. Jacobian sensitivity
    """

    d_values = [
        item["d_k"]
        for item in results
    ]

    # --------------------------------------------------------
    # Extract values
    # --------------------------------------------------------

    unscaled_score_std = [
        item["unscaled"]["score_std"]
        for item in results
    ]

    scaled_score_std = [
        item["scaled"]["score_std"]
        for item in results
    ]

    unscaled_entropy = [
        item["unscaled"]["normalized_entropy"]
        for item in results
    ]

    scaled_entropy = [
        item["scaled"]["normalized_entropy"]
        for item in results
    ]

    unscaled_jacobian = [
        item["unscaled"]["jacobian_frobenius_norm"]
        for item in results
    ]

    scaled_jacobian = [
        item["scaled"]["jacobian_frobenius_norm"]
        for item in results
    ]

    # --------------------------------------------------------
    # Create three-panel figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    # ========================================================
    # Panel 1: Score standard deviation
    # ========================================================

    axes[0].plot(
        d_values,
        unscaled_score_std,
        marker="o",
        label="Unscaled",
    )

    axes[0].plot(
        d_values,
        scaled_score_std,
        marker="o",
        label="Scaled",
    )

    axes[0].set_title(
        "Attention Score Standard Deviation"
    )

    axes[0].set_xlabel(
        "d_k"
    )

    axes[0].set_ylabel(
        "Std of QK^T"
    )

    axes[0].set_xscale(
        "log",
        base=2,
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[0].legend()

    # ========================================================
    # Panel 2: Normalized entropy
    # ========================================================

    axes[1].plot(
        d_values,
        unscaled_entropy,
        marker="o",
        label="Unscaled",
    )

    axes[1].plot(
        d_values,
        scaled_entropy,
        marker="o",
        label="Scaled",
    )

    axes[1].set_title(
        "Normalized Attention Entropy"
    )

    axes[1].set_xlabel(
        "d_k"
    )

    axes[1].set_ylabel(
        "Normalized Entropy"
    )

    axes[1].set_xscale(
        "log",
        base=2,
    )

    axes[1].set_ylim(
        0,
        1.05,
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    axes[1].legend()

    # ========================================================
    # Panel 3: Jacobian sensitivity
    # ========================================================

    axes[2].plot(
        d_values,
        unscaled_jacobian,
        marker="o",
        label="Unscaled",
    )

    axes[2].plot(
        d_values,
        scaled_jacobian,
        marker="o",
        label="Scaled",
    )

    axes[2].set_title(
        "Softmax Jacobian Sensitivity"
    )

    axes[2].set_xlabel(
        "d_k"
    )

    axes[2].set_ylabel(
        r"$||J(p)||_F$"
    )

    axes[2].set_xscale(
        "log",
        base=2,
    )

    axes[2].grid(
        True,
        alpha=0.3,
    )

    axes[2].legend()

    # ========================================================
    # Save
    # ========================================================

    fig.suptitle(
        "Scaled vs Unscaled Attention",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nFigure saved to:\n"
        f"{FIGURE_PATH}"
    )


# ============================================================
# Main
# ============================================================

def main():

    torch.manual_seed(SEED)

    print("=" * 60)
    print("Scaled vs Unscaled Attention Experiment")
    print("=" * 60)

    print(
        f"\nSequence length: {SEQ_LEN}"
    )

    print(
        f"d_k values: {D_K_VALUES}"
    )

    print(
        f"Trials per d_k: {NUM_TRIALS}"
    )

    # --------------------------------------------------------
    # Jacobian validation
    # --------------------------------------------------------

    jacobian_validation = validate_jacobian()

    # --------------------------------------------------------
    # Main experiment
    # --------------------------------------------------------

    results = []

    print("\nMain experiment")
    print("-------------------------------------")

    for d_k in D_K_VALUES:

        result = run_single_dimension(
            d_k
        )

        results.append(result)

        print(
            f"\nd_k = {d_k}"
        )

        print(
            "  Unscaled:"
        )

        print(
            f"    score std       = "
            f"{result['unscaled']['score_std']:.6f}"
        )

        print(
            f"    norm entropy    = "
            f"{result['unscaled']['normalized_entropy']:.6f}"
        )

        print(
            f"    Jacobian ||J||  = "
            f"{result['unscaled']['jacobian_frobenius_norm']:.6f}"
        )

        print(
            "  Scaled:"
        )

        print(
            f"    score std       = "
            f"{result['scaled']['score_std']:.6f}"
        )

        print(
            f"    norm entropy    = "
            f"{result['scaled']['normalized_entropy']:.6f}"
        )

        print(
            f"    Jacobian ||J||  = "
            f"{result['scaled']['jacobian_frobenius_norm']:.6f}"
        )

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    output = {
        "experiment": (
            "Scaled vs Unscaled Attention"
        ),

        "seed": SEED,

        "sequence_length": SEQ_LEN,

        "d_k_values": D_K_VALUES,

        "num_trials": NUM_TRIALS,

        "finite_difference_epsilon": FD_EPS,

        "jacobian_validation": (
            jacobian_validation
        ),

        "results": results,
    }

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"\nResults saved to:\n"
        f"{RESULT_PATH}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_results(results)

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()