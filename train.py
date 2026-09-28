import csv
import json
import math
import os
import time
from datetime import datetime

# On Apple GPUs (MPS), fall back to CPU for the few operations MPS does not
# support instead of crashing. Must be set BEFORE torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SamProcessor

from dataset import create_datasets_and_loaders
from model import (
    build_sam_finetune_mask_decoder,
    build_sam_finetune_lora,
    build_sam_finetune_adapter,
)
from losses import get_loss_function, dice_score, iou_score
from utils import (
    load_config,
    set_seed,
    get_device,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    set_rng_state,
)


MODEL_BUILDERS = {
    "mask_decoder": build_sam_finetune_mask_decoder,
    "lora": build_sam_finetune_lora,
    "adapter": build_sam_finetune_adapter,
}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class RunLogger:
    """Prints to the console (without breaking tqdm bars) and appends to a log file."""

    def __init__(self, log_file, fresh=True):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file
        if fresh:
            open(log_file, "w").close()   # start a clean log for a new run

    def __call__(self, message=""):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}" if message else ""
        tqdm.write(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")


def write_history_csv(path, history):
    if not history:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_curves(history, best_epoch, out_path, title):
    """Loss curves + validation Dice/IoU, with the best epoch marked."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [r["epoch"] for r in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, [r["train_loss"] for r in history], label="train loss")
    ax1.plot(epochs, [r["val_loss"] for r in history], label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")

    ax2.plot(epochs, [r["val_dice"] for r in history], label="val Dice")
    ax2.plot(epochs, [r["val_iou"] for r in history], label="val IoU")
    ax2.set_xlabel("epoch")
    ax2.set_ylim(0, 1)
    ax2.set_title("Validation overlap")

    for ax in (ax1, ax2):
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", color="gray",
                       label=f"best checkpoint (epoch {best_epoch})")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def update_experiments_summary(path, summary):
    """One row per (method, loss). Re-running an experiment replaces its row."""
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = [
                r for r in csv.DictReader(f)
                if not (r.get("fine_tune_method") == summary["fine_tune_method"]
                        and r.get("loss_name") == summary["loss_name"])
            ]
    rows.append(summary)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()),
                                extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Train / validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)

        if ground_truth_masks.ndim == 3:
            ground_truth_masks = ground_truth_masks.unsqueeze(1)

        outputs = model(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=False
        )

        predicted_masks = outputs.pred_masks

        if predicted_masks.ndim == 5:
            predicted_masks = predicted_masks.squeeze(2)

        if predicted_masks.shape[-2:] != ground_truth_masks.shape[-2:]:
            predicted_masks = F.interpolate(
                predicted_masks,
                size=ground_truth_masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        loss = loss_fn(predicted_masks, ground_truth_masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate_one_epoch(model, val_loader, loss_fn, device, threshold=0.5):
    """
    Returns a dict with val loss plus Dice / IoU and the predicted vs.
    ground-truth foreground fraction. The foreground fraction shows
    immediately if the model collapses to empty (all-background) masks.
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_pred_fg = 0.0
    total_gt_fg = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            if ground_truth_masks.ndim == 3:
                ground_truth_masks = ground_truth_masks.unsqueeze(1)

            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False
            )

            predicted_masks = outputs.pred_masks

            if predicted_masks.ndim == 5:
                predicted_masks = predicted_masks.squeeze(2)

            if predicted_masks.shape[-2:] != ground_truth_masks.shape[-2:]:
                predicted_masks = F.interpolate(
                    predicted_masks,
                    size=ground_truth_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            loss = loss_fn(predicted_masks, ground_truth_masks)
            total_loss += loss.item()

            total_dice += dice_score(predicted_masks, ground_truth_masks, threshold)
            total_iou += iou_score(predicted_masks, ground_truth_masks, threshold)
            total_pred_fg += (torch.sigmoid(predicted_masks) > threshold).float().mean().item()
            total_gt_fg += ground_truth_masks.mean().item()

    n = len(val_loader)
    return {
        "val_loss": total_loss / n,
        "val_dice": total_dice / n,
        "val_iou": total_iou / n,
        "val_pred_fg": total_pred_fg / n,
        "val_gt_fg": total_gt_fg / n,
    }


# ---------------------------------------------------------------------------
# One experiment = one (fine-tune method, loss) combination
# ---------------------------------------------------------------------------

def run_experiment(config, fine_tune_method, loss_name, train_loader, val_loader,
                   n_train_patches, n_val_patches, device):

    epochs = config["epochs"]
    threshold = config.get("threshold", 0.5)

    run_dir = os.path.join(config["checkpoint_dir"], fine_tune_method, loss_name)
    best_path = os.path.join(run_dir, "best_model.pt")
    last_path = os.path.join(run_dir, "last_model.pt")
    history_csv = os.path.join(run_dir, "history.csv")
    curves_png = os.path.join(run_dir, "curves.png")
    summary_json = os.path.join(run_dir, "summary.json")

    resume = bool(config.get("resume", False)) and os.path.exists(last_path)
    log = RunLogger(os.path.join(run_dir, "training_log.txt"), fresh=not resume)

    # Same seed for every experiment -> same init / data order / dropout -> fair comparison
    set_seed(config["seed"])

    if fine_tune_method not in MODEL_BUILDERS:
        raise ValueError(f"Unknown fine_tune_method: {fine_tune_method}")

    loss_fn = get_loss_function(loss_name)
    model = MODEL_BUILDERS[fine_tune_method](config["model_name"]).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["learning_rate"])   # float(): YAML reads "1e-4" as text
    )
    n_trainable, n_total = count_parameters(model)

    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = None
    history = []

    if resume:
        ckpt = load_checkpoint(last_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt["best_val_loss"]
        best_epoch = ckpt["best_epoch"]
        history = ckpt["history"] or []
        set_rng_state(ckpt.get("rng_state"))

    # ---------------- header ----------------
    log("=" * 90)
    log(f"EXPERIMENT: {fine_tune_method} + {loss_name}")
    log("=" * 90)
    log(f"Model:              {config['model_name']}")
    log(f"Trainable params:   {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.3f}%)")
    log(f"Learning rate:      {config['learning_rate']}")
    log(f"Epochs:             {epochs} (no early stopping)")
    log(f"Batch size:         {config['batch_size']}")
    log(f"Train / val patches:{n_train_patches:>7,} / {n_val_patches:,}")
    log(f"Seed:               {config['seed']}")
    log(f"Device:             {device}")
    log(f"Output dir:         {run_dir}")
    if resume:
        log(f"RESUMING from {last_path}: {start_epoch} epochs done, "
            f"best val loss {best_val_loss:.6f} at epoch {best_epoch}")
    elif os.path.exists(last_path):
        log("NOTE: an old last_model.pt exists but resume=false -> starting from scratch and overwriting it")
    log("-" * 90)

    if start_epoch >= epochs:
        log(f"Already completed {start_epoch}/{epochs} epochs -> nothing to train.")
    else:
        run_start = time.time()

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val = validate_one_epoch(model, val_loader, loss_fn, device, threshold)
            val_loss = val["val_loss"]

            epoch_time = time.time() - epoch_start

            if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
                log(f"WARNING: non-finite loss at epoch {epoch + 1} "
                    f"(train {train_loss}, val {val_loss}) - learning rate may be too high")

            prev_best_loss, prev_best_epoch = best_val_loss, best_epoch
            is_best = val_loss < best_val_loss - 1e-5
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch + 1

            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val["val_dice"],
                "val_iou": val["val_iou"],
                "val_pred_fg": val["val_pred_fg"],
                "val_gt_fg": val["val_gt_fg"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_s": round(epoch_time, 1),
                "is_best": is_best,
            }
            history.append(row)
            write_history_csv(history_csv, history)

            # ---------------- per-epoch line ----------------
            if is_best:
                if prev_best_epoch is None:
                    best_info = "* NEW BEST (first epoch)"
                else:
                    best_info = (f"* NEW BEST (improved {prev_best_loss - val_loss:.6f} "
                                 f"from epoch {prev_best_epoch})")
            else:
                best_info = (f"best {best_val_loss:.6f} @ epoch {best_epoch} "
                             f"({epoch + 1 - best_epoch} epochs ago)")

            log(
                f"Epoch {epoch + 1:03d}/{epochs} | "
                f"train {train_loss:.6f} | val {val_loss:.6f} | "
                f"Dice {val['val_dice']:.4f} IoU {val['val_iou']:.4f} | "
                f"pred fg {100 * val['val_pred_fg']:.2f}% (GT {100 * val['val_gt_fg']:.2f}%) | "
                f"{epoch_time:.0f}s | {best_info}"
            )

            if val["val_gt_fg"] > 0 and val["val_pred_fg"] < 0.1 * val["val_gt_fg"]:
                log("   WARNING: model predicts <10% of the GT foreground -> "
                    "collapsing towards empty masks?")

            # ---------------- checkpoints ----------------
            ckpt_kwargs = dict(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                loss_name=loss_name,
                fine_tune_method=fine_tune_method,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                history=history,
                val_metrics=val,
            )
            if is_best:
                save_checkpoint(best_path, **ckpt_kwargs)
                log(f"   -> saved best_model.pt (epoch {epoch + 1}, val loss {val_loss:.6f})")
            save_checkpoint(last_path, **ckpt_kwargs)   # overwritten every epoch, used for resuming

            plot_curves(history, best_epoch, curves_png,
                        f"{fine_tune_method} + {loss_name}")

        log(f"Training time this session: {(time.time() - run_start) / 60:.1f} min")

    # ---------------- summary ----------------
    best_row = next(r for r in history if r["epoch"] == best_epoch)
    last_row = history[-1]

    summary = {
        "fine_tune_method": fine_tune_method,
        "loss_name": loss_name,
        "learning_rate": config["learning_rate"],
        "trainable_params": n_trainable,
        "epochs_completed": last_row["epoch"],
        "best_epoch": best_epoch,
        "best_val_loss": best_row["val_loss"],
        "best_val_dice": best_row["val_dice"],
        "best_val_iou": best_row["val_iou"],
        "best_train_loss": best_row["train_loss"],
        "final_train_loss": last_row["train_loss"],
        "final_val_loss": last_row["val_loss"],
        "final_val_dice": last_row["val_dice"],
        "max_val_dice": max(r["val_dice"] for r in history),
        "max_val_dice_epoch": max(history, key=lambda r: r["val_dice"])["epoch"],
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    log("-" * 90)
    log(f"SUMMARY {fine_tune_method} + {loss_name}")
    log(f"  Best checkpoint:  epoch {best_epoch}/{last_row['epoch']} -> {best_path}")
    log(f"    val loss {best_row['val_loss']:.6f} | val Dice {best_row['val_dice']:.4f} "
        f"| val IoU {best_row['val_iou']:.4f} | train loss {best_row['train_loss']:.6f}")
    log(f"  Final epoch:      val loss {last_row['val_loss']:.6f} | val Dice {last_row['val_dice']:.4f} "
        f"| train loss {last_row['train_loss']:.6f}")
    log(f"  Highest val Dice: {summary['max_val_dice']:.4f} at epoch {summary['max_val_dice_epoch']}")

    # simple convergence hints
    if best_epoch >= 0.9 * epochs:
        log("  NOTE: best epoch is near the end -> val loss was still improving; "
            "more epochs or a higher LR might help.")
    elif best_epoch <= 0.3 * epochs:
        log("  NOTE: best epoch is early -> later epochs did not improve val loss "
            "(possible overfitting or LR too high). Check curves.png.")
    if summary["max_val_dice_epoch"] != best_epoch:
        log("  NOTE: the lowest val loss and the highest val Dice are at different epochs.")
    log("=" * 90)
    log()

    update_experiments_summary(
        os.path.join(config["checkpoint_dir"], "experiments_summary.csv"), summary
    )
    return summary


def main():
    config = load_config("config.yaml")
    set_seed(config["seed"])

    device = get_device()   # prints cuda / mps / cpu

    # ---------------- device-dependent settings ----------------
    # CUDA (Linux server): the real runs.
    #   preload=True  -> all volumes in RAM; on Linux the DataLoader workers
    #                    share that memory, so this is fast.
    # Mac (mps) / CPU:  short debug runs only.
    #   preload=False -> one cached volume (old way). On macOS each worker
    #                    would otherwise get its own copy of all volumes, which is slow.
    #   epochs        -> limited to `debug_epochs`
    #   output        -> separate "<checkpoint_dir>_debug" folder, so debug
    #                    results never mix with the CUDA results.
    preload_setting = config.get("preload_volumes", "auto")
    if preload_setting == "auto":
        preload = device == "cuda"
    else:
        preload = bool(preload_setting)

    if device != "cuda":
        debug_epochs = config.get("debug_epochs", 2)
        if debug_epochs:
            config["epochs"] = min(int(config["epochs"]), int(debug_epochs))
        config["checkpoint_dir"] = config["checkpoint_dir"].rstrip("/") + "_debug"
        print(f"DEBUG MODE ({device}): epochs={config['epochs']}, "
              f"output -> {config['checkpoint_dir']}")

    print(f"preload_volumes = {preload} ({preload_setting!r} in config)")

    processor = SamProcessor.from_pretrained(config["model_name"])

    train_dataset, val_dataset, train_loader, val_loader = create_datasets_and_loaders(
        data_dir=config["data_dir"],
        processor=processor,
        val_ratio=config["val_ratio"],
        seed=config["seed"],
        patch_size=config["patch_size"],
        step=config["step"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        positive_only=config["positive_only"],
        min_mask_sum=config["min_mask_sum"],
        preload=preload,
    )

    summaries = []
    for fine_tune_method in config["fine_tune_methods"]:
        for loss_name in config["loss_names"]:
            summaries.append(
                run_experiment(
                    config, fine_tune_method, loss_name,
                    train_loader, val_loader,
                    len(train_dataset), len(val_dataset),
                    device,
                )
            )

    # final overview of all experiments
    print("\n" + "=" * 90)
    print(f"{'method':<14}{'loss':<16}{'best ep':>8}{'best val loss':>15}"
          f"{'val Dice@best':>15}{'max val Dice':>14}")
    print("-" * 90)
    for s in summaries:
        print(f"{s['fine_tune_method']:<14}{s['loss_name']:<16}{s['best_epoch']:>8}"
              f"{s['best_val_loss']:>15.6f}{s['best_val_dice']:>15.4f}{s['max_val_dice']:>14.4f}")
    print("=" * 90)
    print(f"Saved to {os.path.join(config['checkpoint_dir'], 'experiments_summary.csv')}")


if __name__ == "__main__":
    main()
