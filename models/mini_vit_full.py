import torch
import torch.nn as nn

from models.attention import MultiHeadSelfAttention


class MLP(nn.Module):
    """
    Transformer MLP block.

    embed_dim -> hidden_dim -> embed_dim
    """

    def __init__(
        self,
        embed_dim,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        hidden_dim = int(
            embed_dim * mlp_ratio
        )

        self.fc1 = nn.Linear(
            embed_dim,
            hidden_dim
        )

        self.act = nn.GELU()

        self.fc2 = nn.Linear(
            hidden_dim,
            embed_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        x = self.fc1(x)

        x = self.act(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer block.

    x
      |
      +---- LayerNorm -> MHA -> residual
      |
      +---- LayerNorm -> MLP -> residual
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        use_rope=False,
        grid_size=None
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_rope=use_rope,
            grid_size=grid_size
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        # --------------------------------
        # Self-Attention
        # --------------------------------

        x_norm = self.norm1(x)

        attn_out, _ = self.attn(
            x_norm
        )

        x = x + self.dropout(
            attn_out
        )

        # --------------------------------
        # MLP
        # --------------------------------

        x_norm = self.norm2(x)

        mlp_out = self.mlp(
            x_norm
        )

        x = x + mlp_out

        return x


class MiniViTFull(nn.Module):

    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=192,
        num_heads=4,
        depth=4,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=100,
        position_mode="absolute"
    ):
        super().__init__()

        assert (
            img_size % patch_size == 0
        )

        assert position_mode in {
            "none",
            "absolute",
            "rope2d"
        }

        self.img_size = img_size

        self.patch_size = patch_size

        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.depth = depth

        self.position_mode = position_mode

        # --------------------------------
        # Patch grid
        # --------------------------------

        self.grid_size = (
            img_size // patch_size,
            img_size // patch_size
        )

        self.num_patches = (
            self.grid_size[0]
            * self.grid_size[1]
        )

        patch_dim = (
            patch_size
            * patch_size
            * in_channels
        )

        # --------------------------------
        # Patch embedding
        # --------------------------------

        self.patch_embed = nn.Linear(
            patch_dim,
            embed_dim
        )

        # --------------------------------
        # CLS token
        # --------------------------------

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim
            )
        )

        # --------------------------------
        # Learned absolute position
        # --------------------------------

        if position_mode == "absolute":

            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    self.num_patches + 1,
                    embed_dim
                )
            )

            nn.init.trunc_normal_(
                self.pos_embed,
                std=0.02
            )

        else:

            # Keep attribute absent for
            # NoPos / RoPE2D.
            self.pos_embed = None

        self.pos_dropout = nn.Dropout(
            dropout
        )

        # --------------------------------
        # Transformer blocks
        # --------------------------------

        use_rope = (
            position_mode == "rope2d"
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_rope=use_rope,
                    grid_size=self.grid_size
                )
                for _ in range(depth)
            ]
        )

        # --------------------------------
        # Final normalization
        # --------------------------------

        self.norm = nn.LayerNorm(
            embed_dim
        )

        # --------------------------------
        # Classification head
        # --------------------------------

        self.head = nn.Linear(
            embed_dim,
            num_classes
        )

        # --------------------------------
        # Initialization
        # --------------------------------

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02
        )

    # ==========================================================
    # Patchify
    # ==========================================================

    def patchify(self, x):

        B, C, H, W = x.shape

        patches = x.unfold(
            2,
            self.patch_size,
            self.patch_size
        )

        patches = patches.unfold(
            3,
            self.patch_size,
            self.patch_size
        )

        # B, C, num_h, num_w, patch_h, patch_w

        patches = patches.permute(
            0,
            2,
            3,
            1,
            4,
            5
        )

        # B, num_h, num_w, C, patch_h, patch_w

        patches = patches.reshape(
            B,
            self.num_patches,
            -1
        )

        return patches

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(self, x):

        # --------------------------------
        # Image -> patches
        # --------------------------------

        patches = self.patchify(
            x
        )

        # --------------------------------
        # Patch embedding
        # --------------------------------

        tokens = self.patch_embed(
            patches
        )

        # B, 64, 192

        # --------------------------------
        # Add CLS token
        # --------------------------------

        B = tokens.size(0)

        cls_tokens = self.cls_token.expand(
            B,
            -1,
            -1
        )

        tokens = torch.cat(
            [
                cls_tokens,
                tokens
            ],
            dim=1
        )

        # B, 65, 192

        # --------------------------------
        # Position encoding
        # --------------------------------

        if self.position_mode == "absolute":

            tokens = (
                tokens
                + self.pos_embed
            )

        # NoPos:
        # do nothing

        # RoPE2D:
        # do nothing here.
        #
        # RoPE is applied directly to
        # Q/K inside MultiHeadSelfAttention.

        tokens = self.pos_dropout(
            tokens
        )

        # --------------------------------
        # Transformer blocks
        # --------------------------------

        for block in self.blocks:

            tokens = block(
                tokens
            )

        # --------------------------------
        # CLS token
        # --------------------------------

        cls_feature = tokens[:, 0]

        cls_feature = self.norm(
            cls_feature
        )

        # --------------------------------
        # Classification
        # --------------------------------

        logits = self.head(
            cls_feature
        )

        return logits