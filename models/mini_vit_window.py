import torch
import torch.nn as nn


# =========================================================
# Window Attention
# =========================================================

class WindowAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )


    def forward(self, x):

        shortcut = x

        x = self.norm(x)

        x, _ = self.attn(
            x,
            x,
            x
        )

        return x + shortcut



# =========================================================
# MLP
# =========================================================

class MLP(nn.Module):

    def __init__(
        self,
        dim,
        ratio=4,
        dropout=0.1
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.LayerNorm(dim),

            nn.Linear(
                dim,
                dim * ratio
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                dim * ratio,
                dim
            ),

            nn.Dropout(dropout)
        )


    def forward(self,x):

        return self.net(x)



# =========================================================
# Transformer Block
# =========================================================

class WindowBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        shift=False,
        window_size=2
    ):

        super().__init__()

        self.shift = shift

        self.window_size = window_size


        self.attn = WindowAttention(
            dim,
            num_heads
        )


        self.mlp = MLP(
            dim
        )


    def forward(self,x):

        # x:
        # B,H,W,C

        B,H,W,C=x.shape


        # shifted window

        if self.shift:

            x=torch.roll(
                x,
                shifts=(
                    -self.window_size,
                    -self.window_size
                ),
                dims=(
                    1,
                    2
                )
            )


        windows = self.window_partition(
            x
        )


        windows = self.attn(
            windows
        )


        x=self.window_reverse(
            windows,
            B,
            H,
            W,
            C
        )


        if self.shift:

            x=torch.roll(
                x,
                shifts=(
                    self.window_size,
                    self.window_size
                ),
                dims=(
                    1,
                    2
                )
            )


        # flatten

        x=x.reshape(
            B,
            H*W,
            C
        )


        x=x+self.mlp(x)


        return x.reshape(
            B,
            H,
            W,
            C
        )



    def window_partition(self,x):

        B,H,W,C=x.shape

        ws=self.window_size


        windows=x.reshape(
            B,
            H//ws,
            ws,
            W//ws,
            ws,
            C
        )


        windows=windows.permute(
            0,
            1,
            3,
            2,
            4,
            5
        )


        windows=windows.reshape(
            -1,
            ws*ws,
            C
        )


        return windows



    def window_reverse(
        self,
        windows,
        B,
        H,
        W,
        C
    ):

        ws=self.window_size


        x=windows.reshape(
            B,
            H//ws,
            W//ws,
            ws,
            ws,
            C
        )


        x=x.permute(
            0,
            1,
            3,
            2,
            4,
            5
        )


        x=x.reshape(
            B,
            H,
            W,
            C
        )


        return x



# =========================================================
# MiniViT Window
# =========================================================

class MiniViTWindow(nn.Module):

    def __init__(
        self,
        img_size=64,
        patch_size=8,
        window_size=2,
        in_channels=3,
        embed_dim=192,
        num_heads=4,
        depth=4,
        num_classes=100,
        dropout=0.1
    ):

        super().__init__()


        self.patch_size=patch_size


        self.grid_size = (
            img_size // patch_size
        )


        self.num_patches = (
            self.grid_size ** 2
        )


        patch_dim = (
            patch_size
            *
            patch_size
            *
            in_channels
        )


        # patch embedding

        self.patch_embed=nn.Linear(
            patch_dim,
            embed_dim
        )


        # CLS

        self.cls_token=nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim
            )
        )


        self.pos_embed=nn.Parameter(
            torch.zeros(
                1,
                self.num_patches+1,
                embed_dim
            )
        )


        # window blocks

        self.blocks=nn.ModuleList()


        for i in range(depth):

            self.blocks.append(

                WindowBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    shift=(i%2==1),
                    window_size=window_size
                )

            )


        self.norm=nn.LayerNorm(
            embed_dim
        )


        self.head=nn.Linear(
            embed_dim,
            num_classes
        )



    # -----------------------------------------------------
    # patchify
    # -----------------------------------------------------

    def patchify(self,x):

        B,C,H,W=x.shape


        x=x.unfold(
            2,
            self.patch_size,
            self.patch_size
        )


        x=x.unfold(
            3,
            self.patch_size,
            self.patch_size
        )


        x=x.permute(
            0,
            2,
            3,
            1,
            4,
            5
        )


        x=x.reshape(
            B,
            self.num_patches,
            -1
        )


        return x



    # -----------------------------------------------------
    # forward
    # -----------------------------------------------------

    def forward(self,x):

        B=x.size(0)


        x=self.patchify(x)


        x=self.patch_embed(
            x
        )


        x=x+self.pos_embed[:,1:]


        x=x.reshape(
            B,
            self.grid_size,
            self.grid_size,
            -1
        )


        for block in self.blocks:

            x=block(x)



        x=x.reshape(
            B,
            self.num_patches,
            -1
        )


        # CLS pooling

        feature=x.mean(
            dim=1
        )


        feature=self.norm(
            feature
        )


        logits=self.head(
            feature
        )


        return logits