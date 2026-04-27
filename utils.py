# utils.py

import random
import yaml
import numpy as np
import torch
import os


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    random.Random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    loss_name,
    checkpoint_dir
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    filename = (
        f"best_{loss_name}"
        f"_epoch_{epoch+1}"
        f"_valloss_{val_loss:.4f}.pt"
    )

    path = os.path.join(checkpoint_dir, filename)

    torch.save(
        {
            "epoch": epoch+1,
            "loss_name": loss_name,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        path
    )

    return path