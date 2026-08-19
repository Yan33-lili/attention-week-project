import torch
import torch.nn as nn


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
        num_classes=100
    ):
        super().__init__()

        # --------------------------------------------------
        # Patch configuration
        # --------------------------------------------------

        assert img_size % patch_size == 0

        self.img_size = img_size
        self.patch_size = patch_size

        self.num_patches = (
            img_size // patch_size
        ) ** 2

        patch_dim = (
            patch_size
            * patch_size
            * in_channels
        )

        # --------------------------------------------------
        # Patch embedding
        # --------------------------------------------------

        self.patch_embed = nn.Linear(
            patch_dim,
            embed_dim
        )

        # --------------------------------------------------
        # CLS token
        # --------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim
            )
        )

        # --------------------------------------------------
        # Positional embedding
        # --------------------------------------------------

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches + 1,
                embed_dim
            )
        )

        self.pos_dropout = nn.Dropout(
            dropout
        )

        # --------------------------------------------------
        # Transformer encoder
        # --------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(
                embed_dim * mlp_ratio
            ),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        # --------------------------------------------------
        # Classification head
        # --------------------------------------------------

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.head = nn.Linear(
            embed_dim,
            num_classes
        )

        # --------------------------------------------------
        # Initialization
        # --------------------------------------------------

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02
        )

    # ======================================================
    # Patchify
    # ======================================================

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

        # B, num_patches, patch_dim

        return patches

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, x):

        # --------------------------------------------------
        # Image → patches
        # --------------------------------------------------

        patches = self.patchify(x)

        # --------------------------------------------------
        # Patch embedding
        # --------------------------------------------------

        tokens = self.patch_embed(
            patches
        )

        # B, num_patches, embed_dim

        # --------------------------------------------------
        # Add CLS token
        # --------------------------------------------------

        B = tokens.size(0)

        cls_tokens = self.cls_token.expand(
            B,
            -1,
            -1
        )

        tokens = torch.cat(
            [cls_tokens, tokens],
            dim=1
        )

        # B, num_patches + 1, embed_dim

        # --------------------------------------------------
        # Positional embedding
        # --------------------------------------------------

        tokens = tokens + self.pos_embed

        tokens = self.pos_dropout(
            tokens
        )

        # --------------------------------------------------
        # Full self-attention
        # --------------------------------------------------

        tokens = self.encoder(
            tokens
        )

        # --------------------------------------------------
        # CLS token
        # --------------------------------------------------

        cls_feature = tokens[:, 0]

        cls_feature = self.norm(
            cls_feature
        )

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        logits = self.head(
            cls_feature
        )

        return logits