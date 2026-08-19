import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rope import AxialRoPE2D



class MultiHeadSelfAttention(nn.Module):
    """
    Hand-written Multi-Head Self-Attention
    with optional 2D Axial RoPE.

    Input:
        x:
        (B, N, D)

    Output:
        out:
        (B, N, D)

        attention_weights:
        (B, heads, N, N)
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        use_rope=False,
        grid_size=None
    ):
        super().__init__()


        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"


        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads


        self.use_rope = use_rope



        # -------------------------
        # Q,K,V projection
        # -------------------------

        self.q_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.k_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.v_proj = nn.Linear(
            embed_dim,
            embed_dim
        )


        # -------------------------
        # 2D Axial RoPE
        # -------------------------

        if use_rope:

            assert grid_size is not None, \
                "grid_size is required when using RoPE"


            self.rope = AxialRoPE2D(
                dim=self.head_dim,
                grid_size=grid_size
            )



        # -------------------------
        # Output projection
        # -------------------------

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim
        )



    def forward(
        self,
        x
    ):

        B, N, D = x.shape



        # =========================
        # 1. Generate Q,K,V
        # =========================

        Q = self.q_proj(x)

        K = self.k_proj(x)

        V = self.v_proj(x)



        # shape:
        # (B,N,D)



        # =========================
        # 2. Split into heads
        # =========================

        Q = Q.reshape(
            B,
            N,
            self.num_heads,
            self.head_dim
        )

        K = K.reshape(
            B,
            N,
            self.num_heads,
            self.head_dim
        )

        V = V.reshape(
            B,
            N,
            self.num_heads,
            self.head_dim
        )


        # move heads dimension forward

        Q = Q.transpose(
            1,
            2
        )

        K = K.transpose(
            1,
            2
        )

        V = V.transpose(
            1,
            2
        )


        # Now:
        #
        # Q,K,V:
        #
        # (B, heads, N, head_dim)



        # =========================
        # 3. Apply 2D Axial RoPE
        # =========================

        if self.use_rope:

            Q, K = self.rope(
                Q,
                K
            )



        # =========================
        # 4. Attention Score
        # =========================

        scores = torch.matmul(
            Q,
            K.transpose(
                -2,
                -1
            )
        )


        # scaled dot product

        scores = scores / (
            self.head_dim ** 0.5
        )



        # =========================
        # 5. Softmax
        # =========================

        attention_weights = F.softmax(
            scores,
            dim=-1
        )



        # =========================
        # 6. Weighted sum of V
        # =========================

        out = torch.matmul(
            attention_weights,
            V
        )



        # =========================
        # 7. Merge heads
        # =========================

        out = out.transpose(
            1,
            2
        )


        out = out.reshape(
            B,
            N,
            D
        )



        # =========================
        # 8. Output projection
        # =========================

        out = self.out_proj(
            out
        )


        return out, attention_weights