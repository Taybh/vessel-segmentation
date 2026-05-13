import os
import glob
import csv
import torch
import torch.nn.functional as F
from transformers import SamModel, SamProcessor
from peft import PeftModel
from tqdm import tqdm

from dataset import PatchDataset, find_data_mask_pairs
from losses import get_loss_function
from utils import load_config

from dataset import get_bounding_box

def prepare_masks(pred_masks, gt_masks):
    if pred_masks.ndim == 5:
        pred_masks = pred_masks.squeeze(2)

    if gt_masks.ndim == 3:
        gt_masks = gt_masks.unsqueeze(1)

    if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
        pred_masks = F.interpolate(
            pred_masks,
            size=gt_masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    return pred_masks, gt_masks


def compute_metrics(pred_logits, gt_masks, threshold=0.5, eps=1e-7):
    pred_probs = torch.sigmoid(pred_logits)
    pred_bin = (pred_probs > threshold).float()
    gt_masks = gt_masks.float()

    intersection = (pred_bin * gt_masks).sum(dim=(1, 2, 3))
    pred_sum = pred_bin.sum(dim=(1, 2, 3))
    gt_sum = gt_masks.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (pred_sum + gt_sum + eps)
    iou = (intersection + eps) / (pred_sum + gt_sum - intersection + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (gt_sum + eps)

    return {
        "dice": dice.detach().cpu(),
        "iou": iou.detach().cpu(),
        "precision": precision.detach().cpu(),
        "recall": recall.detach().cpu(),
    }


def find_best_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))

    if len(checkpoint_files) == 0:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    best_file = None
    best_val_loss = float("inf")

    for ckpt_file in checkpoint_files:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        val_loss = ckpt.get("val_loss", float("inf"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_file = ckpt_file

    return best_file


def load_model_for_testing(model_name, fine_tune_method, checkpoint_dir, device):
    if fine_tune_method == "lora":
        base_model = SamModel.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)

    elif fine_tune_method == "mask_decoder":
        model = SamModel.from_pretrained(model_name)

        checkpoint_file = find_best_checkpoint(checkpoint_dir)
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    else:
        raise ValueError(f"Unknown fine-tuning method: {fine_tune_method}")

    model.to(device)
    model.eval()

    return model


def evaluate_model(model, test_loader, device, threshold=0.5):
    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            gt_masks = batch["ground_truth_mask"].float().to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False
            )

            pred_masks = outputs.pred_masks
            pred_masks, gt_masks = prepare_masks(pred_masks, gt_masks)

            metrics = compute_metrics(
                pred_logits=pred_masks,
                gt_masks=gt_masks,
                threshold=threshold
            )

            all_dice.extend(metrics["dice"].tolist())
            all_iou.extend(metrics["iou"].tolist())
            all_precision.extend(metrics["precision"].tolist())
            all_recall.extend(metrics["recall"].tolist())

    return {
        "dice_mean": sum(all_dice) / len(all_dice),
        "iou_mean": sum(all_iou) / len(all_iou),
        "precision_mean": sum(all_precision) / len(all_precision),
        "recall_mean": sum(all_recall) / len(all_recall),
        "dice_std": torch.tensor(all_dice).std().item(),
        "iou_std": torch.tensor(all_iou).std().item(),
        "precision_std": torch.tensor(all_precision).std().item(),
        "recall_std": torch.tensor(all_recall).std().item(),
    }


def main():
    config = load_config("config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    processor = SamProcessor.from_pretrained(config["model_name"])

    test_pairs = find_data_mask_pairs(config["test_dir"])

    test_dataset = PatchDataset(
        pairs=test_pairs,
        processor=processor,
        get_bounding_box_fn=get_bounding_box,  # we are promting while testing as well. change to None inorder to have fully automated segmentation
        patch_size=config["patch_size"],
        step=config["step"],
        normalize=True,
        positive_only=config.get("test_positive_only", False),
        min_mask_sum=config.get("min_mask_sum", 1),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    results = []

    for fine_tune_method in config["fine_tune_methods"]:
        for loss_name in config["loss_names"]:

            print(f"\nEvaluating: {fine_tune_method} + {loss_name}")

            checkpoint_dir = os.path.join(
                config["checkpoint_dir"],
                fine_tune_method,
                loss_name
            )

            model = load_model_for_testing(
                model_name=config["model_name"],
                fine_tune_method=fine_tune_method,
                checkpoint_dir=checkpoint_dir,
                device=device
            )

            metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                device=device,
                threshold=config.get("threshold", 0.5)
            )

            row = {
                "fine_tune_method": fine_tune_method,
                "loss_name": loss_name,
                **metrics
            }

            results.append(row)

            print(row)

    output_csv = config.get("test_results_csv", "test_results.csv")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fine_tune_method",
                "loss_name",
                "dice_mean",
                "dice_std",
                "iou_mean",
                "iou_std",
                "precision_mean",
                "precision_std",
                "recall_mean",
                "recall_std",
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved test results to: {output_csv}")


if __name__ == "__main__":
    main()