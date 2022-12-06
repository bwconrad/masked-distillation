from typing import Any, Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.beit import Beit


class VisionTransformer(Beit):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.patch_size = self.patch_embed.patch_size[0]  # type:ignore

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Patch embed image
        x = self.patch_embed(x)
        b, n, d = x.size()

        # Replace masked out patches with the mask token
        mask_token = repeat(self.mask_token, "1 1 d -> b n d", b=b, n=n)
        w = rearrange(mask, "b h w -> b (h w) 1").type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        # Add cls token
        cls_token = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cls_token, x], dim=1)

        # Add position embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through transformer blocks
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, shared_rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        return x


def build_student(model: str, **kwargs) -> nn.Module:
    try:
        model_fn = MODEL_DICT[model]
    except:
        raise ValueError(
            f"{model} is not an available student. Should be one of {[k for k in MODEL_DICT.keys()]}"
        )
    return model_fn(**kwargs)


def default(val: Any, default: Any) -> Any:
    """If val is None set to default value"""
    return val if val is not None else default


def vit_tiny_patch16(
    use_abs_pos_emb: bool = True,
    use_rel_pos_bias: bool = True,
    init_values: float = 0.1,
    drop_path_rate: float = 0.1,
    **kwargs,
) -> VisionTransformer:
    init_values = default(init_values, 0.0)
    drop_path_rate = default(drop_path_rate, 0.0)

    return VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        use_abs_pos_emb=use_abs_pos_emb,
        use_rel_pos_bias=use_rel_pos_bias,
        init_values=init_values,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )


def vit_small_patch16(
    use_abs_pos_emb: bool = True,
    use_rel_pos_bias: bool = True,
    init_values: float = 0.1,
    drop_path_rate: float = 0.1,
    **kwargs,
) -> VisionTransformer:
    init_values = default(init_values, 0.0)
    drop_path_rate = default(drop_path_rate, 0.0)

    return VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        use_abs_pos_emb=use_abs_pos_emb,
        use_rel_pos_bias=use_rel_pos_bias,
        init_values=init_values,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )


def vit_base_patch16(
    use_abs_pos_emb: bool = True,
    use_rel_pos_bias: bool = True,
    init_values: Optional[float] = 0.1,
    drop_path_rate: Optional[float] = 0.1,
    **kwargs,
) -> VisionTransformer:
    init_values = default(init_values, 0.1)
    drop_path_rate = default(drop_path_rate, 0.1)

    return VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        use_abs_pos_emb=use_abs_pos_emb,
        use_rel_pos_bias=use_rel_pos_bias,
        init_values=init_values,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )


def vit_large_patch16(
    use_abs_pos_emb: bool = True,
    use_rel_pos_bias: bool = True,
    init_values: Optional[float] = 1e-5,
    drop_path_rate: Optional[float] = 0.2,
    **kwargs,
) -> VisionTransformer:
    init_values = default(init_values, 1e-5)
    drop_path_rate = default(drop_path_rate, 0.2)

    return VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        use_abs_pos_emb=use_abs_pos_emb,
        use_rel_pos_bias=use_rel_pos_bias,
        init_values=init_values,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )


def vit_huge_patch14(
    use_abs_pos_emb: bool = True,
    use_rel_pos_bias: bool = True,
    init_values: Optional[float] = 1e-5,
    drop_path_rate: Optional[float] = 0.2,
    **kwargs,
) -> VisionTransformer:
    init_values = default(init_values, 1e-5)
    drop_path_rate = default(drop_path_rate, 0.2)

    return VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        use_abs_pos_emb=use_abs_pos_emb,
        use_rel_pos_bias=use_rel_pos_bias,
        init_values=init_values,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )


MODEL_DICT = {
    "vit_tiny_patch16": vit_tiny_patch16,
    "vit_small_patch16": vit_small_patch16,
    "vit_base_patch16": vit_base_patch16,
    "vit_large_patch16": vit_large_patch16,
    "vit_huge_patch14": vit_huge_patch14,
}
