from typing import Any, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from timm.optim.optim_factory import param_groups_weight_decay
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.loss import get_loss_fn
from src.network.head import build_head
from src.network.student import build_student
from src.network.teacher import build_teacher


class MaskDistillModel(pl.LightningModule):
    def __init__(
        self,
        img_size: int = 224,
        student_model: str = "vit_base_patch16",
        teacher_model: str = "clip_vit_base_patch16",
        use_abs_pos_emb: bool = True,
        use_rel_pos_bias: bool = True,
        layer_scale_init_val: Optional[float] = None,
        drop_path_rate: Optional[float] = None,
        window_size: int = 7,
        head_model: str = "fc",
        head_embed_dim: int = 512,
        head_depth: int = 8,
        head_num_heads: int = 16,
        head_fixed_pos_embed: bool = True,
        normalize_targets: bool = True,
        loss_type: str = "smooth_l1",
        smooth_l1_beta: float = 2.0,
        lr: float = 2.5e-4,
        min_lr: float = 1e-5,
        warmup_init_lr: float = 1e-6,
        optimizer: str = "adamw",
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.05,
        momentum: float = 0.9,
        scheduler: str = "cosine",
        warmup_epochs: float = 0.0,
    ):
        """Masked Distillation Pretraining Model

        Args:
            img_size: Size of input image
            student_model: Name of student model. One of [vit_tiny_patch16, vit_small_patch16,
                vit_base_patch16, vit_large_patch16, vit_huge_patch14]
            use_abs_pos_emb: Add learnable position embeddings to student
            use_rel_pos_bias: Add relative position biases to student (only for ViT student)
            layer_scale_init_val: Layer scale initialization value of student (None uses model default) (only for ViT student)
            drop_path_rate: Drop path rate of student (None uses model default)
            window_size: Swin window size
            teacher_model: Name of teacher model. One of [clip_vit_base_patch32, clip_vit_base_patch16,
                clip_vit_large_patch14, openclip_vit_base_patch32, openclip_vit_large_patch14,
                openclip_vit_giant_patch14, openclip_vit_huge_patch14]
            head_model: Decoder head type. One of [fc, vit]
            head_embed_dim: Embed dim of ViT head
            head_depth: Number of transformer blocks in the ViT head
            head_num_heads: Number of attention heads in the ViT head
            head_fixed_pos_embed: Use fixed sin-cos position embeddings in ViT decoder head. False use learnable embeddings.
            normalize_targets: Per-patch normalize the teacher's feature targets
            loss_type: Name of loss function. One of [l1, l2, smooth_l1]
            smooth_l1_beta: Beta value for smooth L1 loss
            lr: Learning rate (should be scaled with batch size. i.e. lr = base_lr*batch_size/256)
            min_lr: Lower learning rate bound in cosine schedule
            warmup_init_lr: Initial learning rate during warm up
            optimizer: Name of optimizer. One of [adam, adamw, sgd]
            betas: Adam beta parameters
            weight_decay: Optimizer weight decay
            momentum: SGD momentum parameter
            scheduler: Name of learning rate scheduler. One of [cosine, none]
            warmup_epochs: Number of warmup epochs (can be a float)
        """
        super().__init__()
        self.save_hyperparameters()
        self.img_size = img_size
        self.student_model = student_model
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_rel_pos_bias = use_rel_pos_bias
        self.layer_scale = layer_scale_init_val
        self.drop_path_rate = drop_path_rate
        self.window_size = window_size
        self.teacher_model = teacher_model
        self.head_model = head_model
        self.head_embed_dim = head_embed_dim
        self.head_depth = head_depth
        self.head_num_heads = head_num_heads
        self.head_fixed_pos_embed = head_fixed_pos_embed
        self.normalize_targets = normalize_targets
        self.loss_type = loss_type
        self.smooth_l1_beta = smooth_l1_beta
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_init_lr = warmup_init_lr
        self.optimizer = optimizer
        self.betas = betas
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs

        # Initialize networks
        self.student = build_student(
            self.student_model,
            img_size=self.img_size,
            use_abs_pos_emb=self.use_abs_pos_emb,
            use_rel_pos_bias=self.use_rel_pos_bias,
            init_values=self.layer_scale,
            drop_path_rate=self.drop_path_rate,
            window_size=self.window_size,
        )
        self.teacher, self.teacher_img_size = build_teacher(
            self.teacher_model,
            size_ratio=self.img_size / self.student.patch_size,  # type:ignore
            student_patch_size=self.student.patch_size,
        )
        self.head = build_head(
            self.head_model,
            in_dim=self.student.embed_dim,
            out_dim=self.teacher.embed_dim,
            num_patches=self.student.patch_embed.num_patches,  # type:ignore
            embed_dim=self.head_embed_dim,
            depth=self.head_depth,
            num_heads=self.head_num_heads,
            use_fixed_sin_cos_pos_embed=self.head_fixed_pos_embed,
            drop_cls_token=self.student.has_cls_token,
        )

        # Define loss function
        self.loss_fn = get_loss_fn(self.loss_type, self.smooth_l1_beta)

    def shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], mode: str = "train"
    ):
        x, mask = batch

        # Pass masked image through student and prediction head
        z = self.student(x, mask)
        pred = self.head(z)

        # Adjust image size for teacher
        if self.img_size != self.teacher_img_size:
            x_t = F.interpolate(x, self.teacher_img_size, mode="bilinear")
        else:
            x_t = x

        # Pass full image through teacher
        target = self.teacher(x_t)
        if self.normalize_targets:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # Calculate loss
        loss = self.loss_fn(pred, target, mask.flatten(1))

        # Log
        self.log(f"{mode}_loss", loss)

        return {"loss": loss}

    def training_step(self, x, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return self.shared_step(x, mode="train")

    def validation_step(self, x, _):
        return self.shared_step(x, mode="val")

    def configure_optimizers(self):
        """Initialize optimizer and learning rate schedule"""
        # Set weight decay to 0 for bias and norm layers
        params = param_groups_weight_decay(
            self.student, self.weight_decay
        ) + param_groups_weight_decay(self.head, self.weight_decay)

        # Optimizer
        if self.optimizer == "adam":
            optimizer = Adam(
                params,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                params,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        # Learning rate schedule
        if self.scheduler == "cosine":
            per_epoch_steps = (
                self.trainer.estimated_stepping_batches
                // self.trainer.max_epochs  # type:ignore
            )
            warmup_steps = per_epoch_steps * self.warmup_epochs
            scheduler = CosineLRScheduler(
                optimizer,
                self.trainer.estimated_stepping_batches,  # type:ignore
                lr_min=self.min_lr,
                warmup_t=warmup_steps,
                warmup_lr_init=self.warmup_init_lr,  # type:ignore
            )
        elif self.scheduler == "none":
            scheduler = LambdaLR(optimizer, lambda _: 1)
        else:
            raise ValueError(
                f"{self.scheduler} is not an available optimizer. Should be one of ['cosine', 'none']"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def lr_scheduler_step(
        self,
        scheduler,
        _,
        metric: Optional[Any],
    ):
        if metric is None:
            scheduler.step(self.global_step)  # type: ignore
        else:
            scheduler.step(self.global_step, metric=metric)
