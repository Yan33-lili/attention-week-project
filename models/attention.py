import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Hand-written Multi-Head Self-Attention

    Input:
        x: (B, N, D)

    B: batch size
    N: number of tokens
    D: embedding dimension
    """

    def __init__(
        self,
        embed_dim,
        num_heads
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # dimension per head
        self.head_dim = embed_dim // num_heads


        # Linear projection for Q,K,V

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


        # output projection

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim
        )


    def forward(self, x):

        B, N, D = x.shape


        # ==========================
        # 1. Generate Q,K,V
        # ==========================

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)


        # shape:
        # B,N,D


        # ==========================
        # 2. Split heads
        # ==========================

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


        # move heads forward

        Q = Q.transpose(1,2)

        K = K.transpose(1,2)

        V = V.transpose(1,2)


        # Now:

        # Q,K,V:
        # B, heads, N, head_dim



        # ==========================
        # 3. Attention score
        # ==========================

        scores = torch.matmul(
            Q,
            K.transpose(-2,-1)
        )


        # divide sqrt(dk)

        scores = scores / (
            self.head_dim ** 0.5
        )


        # ==========================
        # 4. Softmax
        # ==========================

        attention_weights = F.softmax(
            scores,
            dim=-1
        )


        # ==========================
        # 5. Weighted sum of V
        # ==========================

        out = torch.matmul(
            attention_weights,
            V
        )


        # ==========================
        # 6. Merge heads
        # ==========================

        out = out.transpose(
            1,
            2
        )


        out = out.reshape(
            B,
            N,
            D
        )


        # ==========================
        # 7. Final projection
        # ==========================

        out = self.out_proj(out)


        return out, attention_weights