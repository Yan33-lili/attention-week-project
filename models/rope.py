import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    1D Rotary Position Embedding (RoPE)

    Used to generate cos and sin values
    according to token positions.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=512,
        base=10000
    ):
        super().__init__()

        self.dim = dim

        # theta_i = 1 / base^(2i/d)
        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0,
                    dim,
                    2
                ).float()
                / dim
            )
        )

        position = torch.arange(
            max_position_embeddings
        ).float()


        # position * theta

        freqs = torch.einsum(
            "i,j->ij",
            position,
            inv_freq
        )


        self.register_buffer(
            "cos",
            freqs.cos()
        )

        self.register_buffer(
            "sin",
            freqs.sin()
        )


    def forward(
        self,
        positions
    ):
        """
        positions:
        (N,)

        return:
        cos, sin
        """

        cos = self.cos[positions]

        sin = self.sin[positions]


    # Expand from d/2 frequency values
    # to d rotary dimensions
        cos = torch.repeat_interleave(
            cos,
            2,
            dim=-1
        )

        sin = torch.repeat_interleave(
            sin,
            2,
            dim=-1
        )   


        return cos, sin



def rotate_half(x):
    """
    Rotate pairs of dimensions.

    Example:

    [x1,x2,x3,x4]

    becomes:

    [-x2,x1,-x4,x3]
    """

    x1 = x[..., ::2]

    x2 = x[..., 1::2]


    x = torch.stack(
        (-x2, x1),
        dim=-1
    )


    return x.flatten(-2)



def apply_rotary_emb(
    x,
    cos,
    sin
):
    """
    Apply RoPE to tensor.

    x:
    (..., dim)

    cos/sin:
    broadcastable with x
    """

    return (
        x * cos
        +
        rotate_half(x) * sin
    )



class AxialRoPE2D(nn.Module):
    """
    2D Axial Rotary Position Embedding

    Designed for image patch grids.

    Example:

    8x8 patch grid

    positions:

    (0,0)
    (0,1)
    ...
    (7,7)

    The embedding dimension is split into:

    - row axis
    - column axis
    """

    def __init__(
        self,
        dim,
        grid_size,
        base=10000
    ):
        super().__init__()


        assert dim % 4 == 0, \
            "dimension must be divisible by 4"


        self.dim = dim

        self.grid_size = grid_size


        # Half dimensions for row and column

        axis_dim = dim // 2


        self.row_rope = RotaryEmbedding(
            axis_dim,
            max_position_embeddings=grid_size,
            base=base
        )


        self.col_rope = RotaryEmbedding(
            axis_dim,
            max_position_embeddings=grid_size,
            base=base
        )



    def forward(
        self,
        q,
        k
    ):
        """
        Apply 2D axial RoPE.

        q,k:

        (B, heads, N, dim)

        N = grid_size * grid_size
        """


        B, heads, N, D = q.shape


        assert N == (
            self.grid_size *
            self.grid_size
        )


        assert D == self.dim



        # Split embedding dimension

        q_row = q[..., :D//2]

        q_col = q[..., D//2:]


        k_row = k[..., :D//2]

        k_col = k[..., D//2:]



        # Generate 2D positions

        positions = []

        for i in range(self.grid_size):

            for j in range(self.grid_size):

                positions.append(
                    (i,j)
                )


        rows = torch.tensor(
            [
                p[0]
                for p in positions
            ],
            device=q.device
        )


        cols = torch.tensor(
            [
                p[1]
                for p in positions
            ],
            device=q.device
        )


        # Get row and column rotary factors

        row_cos, row_sin = self.row_rope(
            rows
        )


        col_cos, col_sin = self.col_rope(
            cols
        )


        # reshape for broadcasting

        row_cos = row_cos[None,None,:,:]

        row_sin = row_sin[None,None,:,:]


        col_cos = col_cos[None,None,:,:]

        col_sin = col_sin[None,None,:,:]



        # Apply RoPE separately

        q_row = apply_rotary_emb(
            q_row,
            row_cos,
            row_sin
        )


        k_row = apply_rotary_emb(
            k_row,
            row_cos,
            row_sin
        )


        q_col = apply_rotary_emb(
            q_col,
            col_cos,
            col_sin
        )


        k_col = apply_rotary_emb(
            k_col,
            col_cos,
            col_sin
        )


        # Merge row and column dimensions

        q_out = torch.cat(
            [
                q_row,
                q_col
            ],
            dim=-1
        )


        k_out = torch.cat(
            [
                k_row,
                k_col
            ],
            dim=-1
        )


        return q_out, k_out