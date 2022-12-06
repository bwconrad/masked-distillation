"""
Script to extract the student's state_dict from a checkpoint file
"""

from argparse import ArgumentParser

import torch

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="weights.pt", help="Output path of weights"
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default="student",
        help="Parameter prefix of student model used in state dict",
    )
    parser.add_argument(
        "--keep_layer_scale_gamma",
        action="store_true",
        help="Keep layer scale gamma parameters",
    )
    parser.add_argument(
        "--keep_rel_pos_bias",
        action="store_true",
        help="Keep relative position biases",
    )

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint = checkpoint["state_dict"]

    newmodel = {}
    for k, v in checkpoint.items():
        if not k.startswith(args.prefix) or "mask_token" in k:
            continue
        if not args.keep_layer_scale_gamma and "gamma" in k:
            continue
        if not args.keep_rel_pos_bias and "relative_position" in k:
            continue

        k = k.replace(args.prefix + ".", "")
        newmodel[k] = v

    with open(args.output, "wb") as f:
        torch.save(newmodel, f)
