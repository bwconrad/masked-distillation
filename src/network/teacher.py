import math
from functools import partial
from typing import Tuple

import clip
import timm
import torch
import torch.nn as nn
from einops import rearrange


class ClipModel(nn.Module):
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__()
        self.net = clip.load(model, device="cpu")[0].visual
        self.patch_size = self.net.conv1.kernel_size[0]
        self.embed_dim = self.net.conv1.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Change the image encoding forward pass to not discard the per-patch features"""
        w, h = x.shape[2:]
        x = self.net.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.net.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        x = self.net.ln_pre(x)  # type:ignore

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.net.transformer(x)  # type:ignore
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x[:, 1:, :]  # Drop cls token

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Interpolate position embeddings when input is a different spatial resolution
        than what was used during pretraining (i.e. 224x224)

        Adapted from: https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L174
        """
        num_patches_x = x.shape[1] - 1
        num_patches_embed = self.net.positional_embedding.shape[0] - 1  # type:ignore
        if num_patches_x == num_patches_embed and w == h:
            return self.net.positional_embedding  # type:ignore

        # Separate the cls and patch embeddings
        class_positional_embedding = self.net.positional_embedding[:1]  # type:ignore
        patch_positional_embedding = self.net.positional_embedding[1:]  # type:ignore

        # Calculate patch grid size
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        # We add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        # Interpolate position embeddings
        patch_positional_embedding = nn.functional.interpolate(
            rearrange(  # Reshape from 1d to 2d grid
                patch_positional_embedding,
                "(h w) d -> 1 d h w",
                h=int(math.sqrt(num_patches_embed)),
                w=int(math.sqrt(num_patches_embed)),
            ),
            scale_factor=(
                h0 / math.sqrt(num_patches_embed),
                w0 / math.sqrt(num_patches_embed),
            ),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_positional_embedding.shape[-1]
            and int(h0) == patch_positional_embedding.shape[-2]
        )

        # Reshape back to 1d
        patch_positional_embedding = rearrange(
            patch_positional_embedding, "1 d h w -> (h w) d"
        )

        return torch.cat(
            (class_positional_embedding, patch_positional_embedding), dim=0
        )


class TimmModel(nn.Module):
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__()
        kwargs.pop("student_patch_size")
        self.net = timm.create_model(model, pretrained=True, **kwargs)
        self.patch_size = self.net.patch_embed.patch_size[0]
        self.embed_dim = self.net.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward_features(x)[:, 1:, :]  # Drop cls token


class PixelModel(nn.Module):
    def __init__(self, student_patch_size: int, **kwargs) -> None:
        super().__init__()
        self.patch_size = student_patch_size
        self.embed_dim = self.patch_size**2 * 3  # Number of pixels per patch

    def patchify(self, x) -> torch.Tensor:
        """Rearrange image into patches
        (b, 3, h, w) -> (b, n, patch_size^2 * 3)
        """
        assert x.shape[2] == x.shape[3] and x.shape[2] % self.patch_size == 0

        return rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patchify(x)


def build_teacher(
    model: str, size_ratio: float = 1.0, **kwargs
) -> Tuple[nn.Module, int]:
    try:
        model_fn = MODEL_DICT[model]
    except:
        raise ValueError(
            f"{model} is not an available teacher. Should be one of {[k for k in MODEL_DICT.keys()]}"
        )

    # Calculate the adjusted image size
    if model != "pixel":
        patch_size = int(model[-2:])  # Infer patch size from model string
    else:
        # With pixel teacher targets just use the same patch size as the student
        patch_size = kwargs["student_patch_size"]
    kwargs["img_size"] = int(size_ratio * patch_size)

    model = model_fn(**kwargs)

    # Freeze the teacher's weights
    for child in model.children():  # type:ignore
        for param in child.parameters():
            param.requires_grad = False

    return model, kwargs["img_size"]  # type:ignore


MODEL_DICT = {
    "clip_vit_base_patch32": partial(ClipModel, "ViT-B/32"),
    "clip_vit_base_patch16": partial(ClipModel, "ViT-B/16"),
    "clip_vit_large_patch14": partial(ClipModel, "ViT-L/14"),
    "openclip_vit_base_patch32": partial(
        TimmModel, "vit_base_patch32_224_clip_laion2b"
    ),
    "openclip_vit_large_patch14": partial(
        TimmModel, "vit_large_patch14_224_clip_laion2b"
    ),
    "openclip_vit_giant_patch14": partial(
        TimmModel, "vit_giant_patch14_224_clip_laion2b"
    ),
    "openclip_vit_huge_patch14": partial(
        TimmModel, "vit_huge_patch14_224_clip_laion2b"
    ),
    "pixel": PixelModel,
}
