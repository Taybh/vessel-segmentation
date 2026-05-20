import os
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import SamProcessor

from dataset import (
    load_nifti_image,
    normalize_image,
    get_bounding_box,
    find_data_mask_pairs,
    strip_nii_suffix,
)
from model import build_sam_finetune_mask_decoder, build_sam_finetune_lora
from utils import load_config


def find_best_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))

    if len(checkpoint_files) == 0:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    best_file = None
    best_val_loss = float("inf")
    best_epoch = float("inf")

    for ckpt_file in checkpoint_files:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        val_loss = ckpt["val_loss"]
        epoch = ckpt["epoch"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_file = ckpt_file

        elif val_loss == best_val_loss and epoch < best_epoch:
            best_epoch = epoch
            best_file = ckpt_file

    return best_file


def load_model_for_visualization(model_name, fine_tune_method, checkpoint_dir, device):
    checkpoint_file = find_best_checkpoint(checkpoint_dir)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    if fine_tune_method == "lora":
        model = build_sam_finetune_lora(model_name)

    elif fine_tune_method == "mask_decoder":
        model = build_sam_finetune_mask_decoder(model_name)

    else:
        raise ValueError(f"Unknown fine-tuning method: {fine_tune_method}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_file}")

    return model


def select_slice(image_volume, mask_volume, slice_idx=None, auto_select=False):
    if slice_idx is not None:
        return slice_idx

    if auto_select:
        mask_sums = [
            np.sum(mask_volume[:, :, z])
            for z in range(mask_volume.shape[2])
        ]
        return int(np.argmax(mask_sums))

    return image_volume.shape[2] // 2


def predict_full_slice(
    model,
    processor,
    image_volume,
    mask_volume,
    slice_idx,
    device,
    threshold=0.5,
):
    image_slice = image_volume[:, :, slice_idx]
    gt_slice = mask_volume[:, :, slice_idx]

    original_h, original_w = image_slice.shape

    image_slice = normalize_image(image_slice)

    image_for_processor = np.stack(
        [image_slice] * 3,
        axis=-1
    )

    prompt = get_bounding_box(gt_slice)

    inputs = processor(
        image_for_processor,
        input_boxes=[[prompt]],
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    model.eval()

    with torch.no_grad():
        outputs = model(
            pixel_values=inputs["pixel_values"],
            input_boxes=inputs["input_boxes"],
            multimask_output=False
        )

    pred_mask = outputs.pred_masks

    if pred_mask.ndim == 5:
        pred_mask = pred_mask.squeeze(2)

    pred_mask = F.interpolate(
        pred_mask,
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=False
    )

    pred_prob = torch.sigmoid(pred_mask)
    pred_bin = (pred_prob > threshold).float()

    pred_slice = pred_bin.squeeze().detach().cpu().numpy()

    return image_slice, gt_slice, pred_slice

def save_full_slice_visualization(
    image_slice,
    gt_slice,
    pred_slice,
    save_path,
    title=""
):
    image_norm = normalize_image(image_slice)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].imshow(image_norm, cmap="gray")
    axes[0].set_title("Input slice")

    axes[1].imshow(gt_slice, cmap="gray")
    axes[1].set_title("Ground truth")

    axes[2].imshow(pred_slice, cmap="gray")
    axes[2].set_title("Prediction")

    axes[3].imshow(image_norm, cmap="gray")
    axes[3].imshow(gt_slice, cmap="Greens", alpha=0.4)
    axes[3].imshow(pred_slice, cmap="Reds", alpha=0.4)
    axes[3].set_title("Overlay: GT=green, Pred=red")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    config = load_config("config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    processor = SamProcessor.from_pretrained(config["model_name"])

    test_pairs = find_data_mask_pairs(config["test_dir"])

    if len(test_pairs) == 0:
        raise RuntimeError(f"No test image-mask pairs found in {config['test_dir']}")

    models_to_visualize = config.get(
        "models_to_visualize",
        [
            ["lora", "dice_focal"],
            ["mask_decoder", "dice_focal"],
        ]
    )

    output_root = config.get(
        "visualization_dir",
        "prediction_visualizations"
    )

    for fine_tune_method, loss_name in models_to_visualize:
        print(f"\nLoading model: {fine_tune_method} + {loss_name}")

        checkpoint_dir = os.path.join(
            config["checkpoint_dir"],
            fine_tune_method,
            loss_name,
            "best"
        )

        model = load_model_for_visualization(
            model_name=config["model_name"],
            fine_tune_method=fine_tune_method,
            checkpoint_dir=checkpoint_dir,
            device=device
        )

        for image_path, mask_path in test_pairs:
            print(f"\nVisualizing image: {image_path.name}")

            image_volume = load_nifti_image(image_path, is_mask=False)
            mask_volume = load_nifti_image(mask_path, is_mask=True)

            slice_idx = select_slice(
                image_volume=image_volume,
                mask_volume=mask_volume,
                slice_idx=config.get("visualization_slice_idx", None),
                auto_select=config.get("auto_select_slice", False)
            )

            print(f"Selected slice: {slice_idx}")
            print(f"Vessel pixels in slice: {np.sum(mask_volume[:, :, slice_idx])}")

            image_slice, gt_slice, pred_slice = predict_full_slice(
                model=model,
                processor=processor,
                image_volume=image_volume,
                mask_volume=mask_volume,
                slice_idx=slice_idx,
                device=device,
                threshold=config.get("threshold", 0.5),
            )

            case_name = strip_nii_suffix(image_path.name)

            output_dir = os.path.join(
                output_root,
                fine_tune_method,
                loss_name,
                case_name
            )

            os.makedirs(output_dir, exist_ok=True)

            save_path = os.path.join(
                output_dir,
                f"slice_{slice_idx}.png"
            )

            save_full_slice_visualization(
                image_slice=image_slice,
                gt_slice=gt_slice,
                pred_slice=pred_slice,
                save_path=save_path,
                title=f"{case_name} | {fine_tune_method} + {loss_name} | slice {slice_idx}"
            )

            print(f"Saved: {save_path}")

            del image_volume
            del mask_volume
            del image_slice
            del gt_slice
            del pred_slice

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()