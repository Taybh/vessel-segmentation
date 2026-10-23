"""
Evaluate trained SAM models on the held-out TEST set (patch level).

- Test set   = all volumes in config["test_dir"] (separate folder, never used in training).
- Val set    = the same 20 % validation volumes as in train.py (same split + seed);
               used ONLY to choose the probability threshold.
- Prompts    = tight ground-truth bounding boxes (no random jitter), same as validation.
- Checkpoint = best_model.pt of every (fine_tune_method, loss) in config.yaml.

Outputs (in test_output_dir):
  test_summary.csv          one row per model: test metrics at threshold 0.5 and at the
                            threshold tuned on validation
  threshold_sweep.csv       val + test metrics for every threshold in eval_thresholds
  per_patch/<method>_<loss>.csv   metrics for every test patch (for statistics / visualisation)
"""

import csv
import os
from functools import partial

# On Apple GPUs (MPS), fall back to CPU for unsupported ops. Must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamProcessor

from dataset import (
    PatchDataset,
    find_data_mask_pairs,
    split_train_val,
    get_bounding_box,
    strip_nii_suffix,
)
from model import (
    build_sam_finetune_mask_decoder,
    build_sam_finetune_lora,
    build_sam_finetune_adapter,
)
from utils import load_config, set_seed, get_device, load_checkpoint


MODEL_BUILDERS = {
    "mask_decoder": build_sam_finetune_mask_decoder,
    "lora": build_sam_finetune_lora,
    "adapter": build_sam_finetune_adapter,
}

METRICS = ["dice", "iou", "precision", "recall", "cldice", "pred_fg", "gt_fg"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def volume_names(pairs):
    return {strip_nii_suffix(img.name) for img, _ in pairs}


def check_no_overlap(trainval_pairs, test_pairs):
    """Make sure no volume appears in both the train/val folder and the test folder."""
    overlap = volume_names(trainval_pairs) & volume_names(test_pairs)
    if overlap:
        raise ValueError(
            f"{len(overlap)} volume(s) are in BOTH data_dir and test_dir: {sorted(overlap)[:10]} ..."
        )


def build_eval_dataset(pairs, processor, config, preload):
    return PatchDataset(
        pairs=pairs,
        processor=processor,
        get_bounding_box_fn=partial(get_bounding_box, perturb=False),   # fixed, tight boxes
        patch_size=config["patch_size"],
        step=config["step"],
        normalize=True,
        positive_only=config["positive_only"],
        min_mask_sum=config["min_mask_sum"],
        preload=preload,
    )


def build_loader(dataset, config):
    num_workers = config["num_workers"]
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,                      # keep order -> patch i = dataset.patch_index[i]
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Metrics (per patch, binary numpy masks)
# ---------------------------------------------------------------------------

def cldice_score(pred, gt):
    """clDice on binary 2D masks (skeleton-based topology metric)."""
    skel_pred = skeletonize(pred)
    skel_gt = skeletonize(gt)

    if skel_pred.sum() == 0 and skel_gt.sum() == 0:
        return 1.0
    if skel_pred.sum() == 0 or skel_gt.sum() == 0:
        return 0.0

    tprec = np.logical_and(skel_pred, gt).sum() / skel_pred.sum()   # topology precision
    tsens = np.logical_and(skel_gt, pred).sum() / skel_gt.sum()     # topology sensitivity
    if tprec + tsens == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))


def patch_metrics(pred, gt):
    """
    pred, gt: boolean arrays (H, W).
    Empty cases are handled explicitly, so an EMPTY prediction gets precision 0
    (not 1 as with an eps-smoothed formula).
    """
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    pred_sum, gt_sum = tp + fp, tp + fn

    if pred_sum + gt_sum == 0:
        dice = iou = 1.0
    else:
        dice = 2 * tp / (pred_sum + gt_sum)
        iou = tp / (tp + fp + fn)

    precision = (1.0 if gt_sum == 0 else 0.0) if pred_sum == 0 else tp / pred_sum
    recall = (1.0 if pred_sum == 0 else 0.0) if gt_sum == 0 else tp / gt_sum

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "cldice": cldice_score(pred, gt),
        "pred_fg": pred_sum / pred.size,
        "gt_fg": gt_sum / gt.size,
        "tp": tp, "fp": fp, "fn": fn,
    }


def summarize(rows):
    """Mean +- std over patches, plus pooled ('global') metrics over all pixels."""
    out = {"n_patches": len(rows)}
    for k in METRICS:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0

    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    out["dice_global"] = 2 * tp / max(2 * tp + fp + fn, 1)
    out["precision_global"] = tp / max(tp + fp, 1)
    out["recall_global"] = tp / max(tp + fn, 1)
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, dataset, device, thresholds, desc):
    """Run the model once and compute patch metrics for every threshold."""
    model.eval()
    records = {t: [] for t in thresholds}
    i = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        gt_masks = batch["ground_truth_mask"]

        outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
        pred = outputs.pred_masks
        if pred.ndim == 5:
            pred = pred.squeeze(2)
        if gt_masks.ndim == 3:
            gt_masks = gt_masks.unsqueeze(1)
        if pred.shape[-2:] != gt_masks.shape[-2:]:
            pred = F.interpolate(pred, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False)

        probs = torch.sigmoid(pred).float().cpu().numpy()[:, 0]
        gts = gt_masks.numpy()[:, 0] > 0.5

        for prob, gt in zip(probs, gts):
            vol_idx, z, x, y = dataset.patch_index[i]
            info = {"volume": strip_nii_suffix(dataset.pairs[vol_idx][0].name), "z": z, "x": x, "y": y}
            i += 1
            for t in thresholds:
                records[t].append({**info, **patch_metrics(prob > t, gt)})

    return records


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def short(s):
    return (f"Dice {s['dice_mean']:.4f}+-{s['dice_std']:.3f} | IoU {s['iou_mean']:.4f} | "
            f"Prec {s['precision_mean']:.4f} | Rec {s['recall_mean']:.4f} | "
            f"clDice {s['cldice_mean']:.4f} | pred fg {100 * s['pred_fg_mean']:.2f}% "
            f"(GT {100 * s['gt_fg_mean']:.2f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config("config.yaml")
    set_seed(config["seed"])
    device = get_device()

    preload_setting = config.get("preload_volumes", "auto")
    preload = (device == "cuda") if preload_setting == "auto" else bool(preload_setting)

    default_t = float(config.get("threshold", 0.5))
    thresholds = sorted({float(t) for t in config.get("eval_thresholds", [default_t])} | {default_t})
    tune = bool(config.get("tune_threshold_on_val", True))

    out_dir = config.get("test_output_dir", os.path.join(config["checkpoint_dir"], "test_results"))
    os.makedirs(os.path.join(out_dir, "per_patch"), exist_ok=True)

    processor = SamProcessor.from_pretrained(config["model_name"])

    # ---------------- data ----------------
    trainval_pairs = find_data_mask_pairs(config["data_dir"])
    test_pairs = find_data_mask_pairs(config["test_dir"])
    check_no_overlap(trainval_pairs, test_pairs)
    _, val_pairs = split_train_val(trainval_pairs, val_ratio=config["val_ratio"], seed=config["seed"])

    print(f"Test volumes: {len(test_pairs)} (from {config['test_dir']})")
    test_ds = build_eval_dataset(test_pairs, processor, config, preload)
    test_loader = build_loader(test_ds, config)

    if tune:
        print(f"Val volumes:  {len(val_pairs)} (same split as training, used only for threshold tuning)")
        val_ds = build_eval_dataset(val_pairs, processor, config, preload)
        val_loader = build_loader(val_ds, config)

    summary_rows, sweep_rows = [], []

    # ---------------- models ----------------
    for method in config["fine_tune_methods"]:
        for loss_name in config["loss_names"]:
            ckpt_path = os.path.join(config["checkpoint_dir"], method, loss_name, "best_model.pt")
            if not os.path.exists(ckpt_path):
                print(f"\nSKIP {method} + {loss_name}: {ckpt_path} not found")
                continue

            print(f"\n=== {method} + {loss_name} ===")
            ckpt = load_checkpoint(ckpt_path)
            model = MODEL_BUILDERS[method](config["model_name"])
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            print(f"Checkpoint: epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.6f}")

            # threshold tuning on validation
            best_t = default_t
            if tune:
                val_rec = evaluate(model, val_loader, val_ds, device, thresholds, "Validation")
                for t in thresholds:
                    sweep_rows.append({"fine_tune_method": method, "loss_name": loss_name,
                                       "split": "val", "threshold": t, **summarize(val_rec[t])})
                best_t = max(thresholds, key=lambda t: np.mean([r["dice"] for r in val_rec[t]]))
                print(f"Threshold tuned on val: {best_t}")

            # test
            test_rec = evaluate(model, test_loader, test_ds, device, thresholds, "Test")
            for t in thresholds:
                sweep_rows.append({"fine_tune_method": method, "loss_name": loss_name,
                                   "split": "test", "threshold": t, **summarize(test_rec[t])})

            base = {"fine_tune_method": method, "loss_name": loss_name,
                    "checkpoint_epoch": ckpt["epoch"], "checkpoint_val_loss": ckpt["val_loss"]}

            s_default = summarize(test_rec[default_t])
            summary_rows.append({**base, "threshold": default_t, "threshold_source": "default", **s_default})
            print(f"TEST @ {default_t}: {short(s_default)}")

            if best_t != default_t:
                s_tuned = summarize(test_rec[best_t])
                summary_rows.append({**base, "threshold": best_t, "threshold_source": "tuned_on_val", **s_tuned})
                print(f"TEST @ {best_t}: {short(s_tuned)}")

            write_csv(os.path.join(out_dir, "per_patch", f"{method}_{loss_name}.csv"),
                      [{"threshold": best_t, **r} for r in test_rec[best_t]])

            # save after every model, so partial results survive a crash
            write_csv(os.path.join(out_dir, "test_summary.csv"), summary_rows)
            write_csv(os.path.join(out_dir, "threshold_sweep.csv"), sweep_rows)

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---------------- overview ----------------
    print("\n" + "=" * 110)
    print(f"{'method':<8}{'loss':<15}{'thr':>5}{'Dice':>9}{'IoU':>9}{'Prec':>9}{'Rec':>9}"
          f"{'clDice':>9}{'predFG%':>9}")
    print("-" * 110)
    for r in summary_rows:
        print(f"{r['fine_tune_method']:<8}{r['loss_name']:<15}{r['threshold']:>5}"
              f"{r['dice_mean']:>9.4f}{r['iou_mean']:>9.4f}{r['precision_mean']:>9.4f}"
              f"{r['recall_mean']:>9.4f}{r['cldice_mean']:>9.4f}{100 * r['pred_fg_mean']:>9.2f}")
    print("=" * 110)
    print(f"Results saved in {out_dir}")


if __name__ == "__main__":
    main()
