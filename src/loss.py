from functools import partial
from typing import Callable

import torch


def masked_l2_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """L2/MSE loss on masked patches

    Args:
        pred: B x num_patches x D tensor of predict patches
        target: B x num_patches x D tensor of target patch values
        mask: B x num_patches binary mask with masked patches marked with 1

    Return:
        loss: Masked L2 loss
    """

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # Per patch loss
    loss = (loss * mask).sum() / mask.sum()  # Mean of masked patches

    return loss


def masked_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """L1/MAE loss on masked patches

    Args:
        pred: B x num_patches x D tensor of predict patches
        target: B x num_patches x D tensor of target patch values
        mask: B x num_patches binary mask with masked patches marked with 1

    Return:
        loss: Masked L1 loss
    """

    loss = (pred - target).abs()
    loss = loss.mean(dim=-1)  # Per patch loss
    loss = (loss * mask).sum() / mask.sum()  # Mean of masked patches

    return loss


def masked_smooth_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """Smooth L1 loss on masked patches

    Args:
        pred: B x num_patches x D tensor of predict patches
        target: B x num_patches x D tensor of target patch values
        mask: B x num_patches binary mask with masked patches marked with 1
        beta: Float value of L1 to L2 change point

    Return:
        loss: Masked smooth L1 loss
    """
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    loss = loss.mean(dim=-1)  # Per patch loss
    loss = (loss * mask).sum() / mask.sum()  # Mean of masked patches

    return loss


def get_loss_fn(name: str, beta: float) -> Callable:
    if name == "l1":
        return masked_l1_loss
    elif name == "l2":
        return masked_l2_loss
    elif name == "smooth_l1":
        return partial(masked_smooth_l1_loss, beta=beta)
    else:
        raise ValueError(
            f"{name} is not an loss function type. Should be one of ['l1', 'l2', 'smooth_l1']"
        )
