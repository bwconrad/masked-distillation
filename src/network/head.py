import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.layers.weight_init import trunc_normal_
from timm.models.vision_transformer import Block


class VitDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_patches: int = 196,
        depth: int = 8,
        embed_dim: int = 512,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),  # type:ignore
        act_layer: nn.Module = nn.GELU,  # type:ignore
        use_fixed_sin_cos_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Projection from encoder to decoder dim
        self.embed = nn.Linear(in_dim, self.embed_dim, bias=True)

        # Position embedding
        if use_fixed_sin_cos_pos_embed:
            # Fixed position embedding
            self.pos_embed = self.build_2d_sincos_position_embedding()
        else:
            # Learnable position embedding
            self.pos_embed = nn.Parameter(
                torch.zeros((1, num_patches + 1, self.embed_dim)), requires_grad=True
            )
            trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,  # type:ignore
                    act_layer=act_layer,  # type:ignore
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, out_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_2d_sincos_position_embedding(self, temperature=10000.0) -> nn.Parameter:
        """From: https://github.com/facebookresearch/moco-v3/blob/main/vits.py"""
        h = w = int(math.sqrt(self.num_patches))
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert (
            self.embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        # assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        embed.requires_grad = False

        return embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project to decoder embed size
        x = self.embed(x)

        # Add pos embed
        x = x + self.pos_embed

        # Apply transformer layers
        x = self.blocks(x)

        # Apply prediction head
        x = self.head(self.norm(x))

        return x[:, 1:, :]  # Drop cls token


class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)[:, 1:, :]  # Drop cls token


def build_head(model: str, **kwargs) -> nn.Module:
    try:
        model_fn = MODEL_DICT[model]
    except:
        raise ValueError(
            f"{model} is not an available head. Should be one of {[k for k in MODEL_DICT.keys()]}"
        )

    return model_fn(**kwargs)


MODEL_DICT = {"vit": VitDecoder, "fc": Linear}
