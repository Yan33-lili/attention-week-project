from pathlib import Path
import time
import json

import torch
import matplotlib.pyplot as plt


# =====================================================
# Paths
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "figures"
    / "attention_efficiency"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULT_PATH = (
    OUTPUT_DIR
    / "efficiency_results.json"
)

FIGURE_PATH = (
    OUTPUT_DIR
    / "attention_efficiency.png"
)


# =====================================================
# Config
# =====================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


BATCH = 1

CHANNEL = 64

WINDOW_SIZE = 4


# Patch grid:
#
# 8x8     = 64 tokens
# 8x16    = 128 tokens
# 16x16   = 256 tokens
# 16x32   = 512 tokens
# 32x32   = 1024 tokens
#

TOKEN_GRIDS = [
    (8, 8),
    (8, 16),
    (16, 16),
    (16, 32),
    (32, 32),
]


ITERATIONS = 50



# =====================================================
# Full Attention
# =====================================================

def full_attention(x):
    """
    Full Self Attention

    x:
        [B,N,C]

    Attention map:
        [B,N,N]
    """

    q = x
    k = x


    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    )


    attn = torch.softmax(
        scores,
        dim=-1
    )


    out = torch.matmul(
        attn,
        x
    )


    return out



# =====================================================
# Window Attention
# =====================================================

def window_attention(
        x,
        height,
        width,
        window_size
):
    """
    Window Self Attention

    Input:
        x:
        [B,N,C]

    Output:
        window outputs
    """

    B, N, C = x.shape


    assert (
        N == height * width
    )


    assert (
        height % window_size == 0
    )

    assert (
        width % window_size == 0
    )


    # -------------------------------------------------
    # [B,N,C]
    #
    # ->
    #
    # [B,H,W,C]
    # -------------------------------------------------

    x = x.reshape(
        B,
        height,
        width,
        C
    )


    # -------------------------------------------------
    # Split windows
    #
    # [B,H,W,C]
    #
    # ->
    #
    # [B,
    #  H/M,
    #  M,
    #  W/M,
    #  M,
    #  C]
    # -------------------------------------------------

    x = x.reshape(
        B,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        C
    )


    # -------------------------------------------------
    # Rearrange
    #
    # ->
    #
    # [B,
    #  H/M,
    #  W/M,
    #  M,
    #  M,
    #  C]
    # -------------------------------------------------

    x = x.permute(
        0,
        1,
        3,
        2,
        4,
        5
    )


    # -------------------------------------------------
    # Merge windows
    #
    # ->
    #
    # [num_windows*B,
    #  M*M,
    #  C]
    # -------------------------------------------------

    windows = x.reshape(
        -1,
        window_size * window_size,
        C
    )


    q = windows
    k = windows


    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    )


    attn = torch.softmax(
        scores,
        dim=-1
    )


    out = torch.matmul(
        attn,
        windows
    )


    return out



# =====================================================
# Runtime benchmark
# =====================================================

def benchmark(
        func,
        x
):

    if DEVICE == "cuda":

        torch.cuda.empty_cache()

        torch.cuda.synchronize()


    start = time.perf_counter()


    for _ in range(ITERATIONS):

        func(x)


    if DEVICE == "cuda":

        torch.cuda.synchronize()


    end = time.perf_counter()


    return (
        end - start
    ) / ITERATIONS



# =====================================================
# Memory estimation
# =====================================================

def full_attention_memory(N):

    """
    Full attention map:

    N*N

    float32:
        4 bytes
    """

    elements = (
        N * N
    )


    bytes = (
        elements * 4
    )


    return bytes / 1024**2



def window_attention_memory(
        N,
        window_size
):

    """
    Window attention map:

    number_of_windows * M^4
    """

    windows = (
        N //
        (window_size ** 2)
    )


    elements = (
        windows *
        window_size**4
    )


    bytes = (
        elements * 4
    )


    return bytes / 1024**2



# =====================================================
# FLOPs estimation
# =====================================================

def full_attention_flops(
        N,
        C
):

    """
    QK^T complexity:

    O(N^2 C)
    """

    return (
        2 *
        N *
        N *
        C
    )



def window_attention_flops(
        N,
        C,
        window_size
):

    """
    O(N M^2 C)
    """

    return (
        2 *
        N *
        window_size**2 *
        C
    )



# =====================================================
# Plot
# =====================================================

def plot_results(results):


    tokens = [
        r["tokens"]
        for r in results
    ]


    full_time = [
        r["full_time"]
        for r in results
    ]


    window_time = [
        r["window_time"]
        for r in results
    ]


    full_memory = [
        r["full_memory"]
        for r in results
    ]


    window_memory = [
        r["window_memory"]
        for r in results
    ]


    plt.figure(
        figsize=(8,5)
    )


    plt.plot(
        tokens,
        full_time,
        marker="o",
        label="Full Attention"
    )


    plt.plot(
        tokens,
        window_time,
        marker="o",
        label="Window Attention"
    )


    plt.xlabel(
        "Number of Tokens"
    )


    plt.ylabel(
        "Runtime (s)"
    )


    plt.title(
        "Full vs Window Attention Runtime"
    )


    plt.grid(
        True
    )


    plt.legend()


    plt.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()



# =====================================================
# Main
# =====================================================

def main():


    print(
        "=" * 60
    )

    print(
        "Full Attention vs Window Attention Benchmark"
    )

    print(
        "=" * 60
    )


    print(
        "\nDevice:",
        DEVICE
    )


    results = []


    for height, width in TOKEN_GRIDS:


        N = (
            height *
            width
        )


        print(
            f"\nTokens: {N}"
        )


        print(
            f"Grid: {height} x {width}"
        )


        x = torch.randn(
            BATCH,
            N,
            CHANNEL,
            device=DEVICE
        )


        # -------------------------------
        # Runtime
        # -------------------------------

        full_time = benchmark(
            full_attention,
            x
        )


        window_time = benchmark(
            lambda y:
            window_attention(
                y,
                height,
                width,
                WINDOW_SIZE
            ),
            x
        )


        result = {

            "tokens": N,

            "height": height,

            "width": width,


            "full_time":
                full_time,


            "window_time":
                window_time,


            "full_memory":
                full_attention_memory(
                    N
                ),


            "window_memory":
                window_attention_memory(
                    N,
                    WINDOW_SIZE
                ),


            "full_flops":
                full_attention_flops(
                    N,
                    CHANNEL
                ),


            "window_flops":
                window_attention_flops(
                    N,
                    CHANNEL,
                    WINDOW_SIZE
                )
        }


        results.append(
            result
        )


        print(
            result
        )


    # ---------------------------------
    # Save json
    # ---------------------------------

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )


    print(
        "\nResults saved:"
    )

    print(
        RESULT_PATH
    )


    # ---------------------------------
    # Plot
    # ---------------------------------

    plot_results(
        results
    )


    print(
        "\nFigure saved:"
    )

    print(
        FIGURE_PATH
    )


    print(
        "\nBenchmark completed."
    )



if __name__ == "__main__":

    main()