# utils.py

import os
import random

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)           # affects random operations performed by PyTorch
    torch.cuda.manual_seed_all(seed)  # seeds all CUDA GPUs
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)   # seeds the Apple GPU


def get_device():
    """
    Pick the best available device: NVIDIA GPU (cuda) -> Apple Silicon GPU (mps) -> CPU.
    Prints which one is used so it is visible in the console.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Device: cuda ({name}, {torch.cuda.device_count()} GPU(s))")
        return "cuda"

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("Device: mps (Apple Silicon GPU)")
        return "mps"

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_built():
        print("NOTE: this PyTorch supports MPS, but no Apple GPU is available "
              "(needs macOS 12.3+ on Apple Silicon).")
    print("Device: cpu (training will be slow)")
    return "cpu"


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# RNG state (so a resumed run continues with the same randomness)
# ---------------------------------------------------------------------------

def get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    loss_name,
    fine_tune_method,
    best_val_loss=None,
    best_epoch=None,
    history=None,
    val_metrics=None,
):
    """
    Save a checkpoint to a FIXED path (e.g. best_model.pt / last_model.pt),
    overwriting the previous one.

    The keys used by the test script (epoch, loss_name, fine_tune_method,
    train_loss, val_loss, model_state_dict, optimizer_state_dict) are unchanged.

    Writes to a temporary file first and then renames it, so a crash during
    saving can never leave a half-written (corrupted) checkpoint behind.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    state = {
        "epoch": epoch + 1,                  # number of completed epochs
        "loss_name": loss_name,
        "fine_tune_method": fine_tune_method,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_metrics": val_metrics,          # val Dice / IoU / foreground ratio at this epoch
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,                  # full per-epoch history up to this point
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),  # needed to resume training
        "rng_state": get_rng_state(),
    }

    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_checkpoint(path):
    """
    Load on CPU; model.load_state_dict / optimizer.load_state_dict move the
    tensors to the right device afterwards.

    weights_only=False is needed because the checkpoint also stores the
    training history and RNG state (not only tensors). Only load
    checkpoints you created yourself.
    """
    return torch.load(path, map_location="cpu", weights_only=False)
