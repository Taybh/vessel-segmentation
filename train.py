import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SamProcessor

from dataset import create_datasets_and_loaders
from model import build_sam_finetune_mask_decoder
from losses import get_loss_function
from utils import load_config, set_seed, get_device, save_checkpoint

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=False
        )

        predicted_masks = outputs.pred_masks

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


def validate_one_epoch(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False
            )

            predicted_masks = outputs.pred_masks

            if predicted_masks.shape[-2:] != ground_truth_masks.shape[-2:]:
                predicted_masks = F.interpolate(
                    predicted_masks,
                    size=ground_truth_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            loss = loss_fn(predicted_masks, ground_truth_masks)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    config = load_config("config.yaml")
    set_seed(config["seed"])

    device = get_device()
    print("Using device:", device)

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    processor = SamProcessor.from_pretrained(config["model_name"])

    train_dataset, val_dataset, train_loader, val_loader = create_datasets_and_loaders(
        root_dir=config["root_dir"],
        processor=processor,
        val_ratio=config["val_ratio"],
        seed=config["seed"],
        patch_size=config["patch_size"],
        step=config["step"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        positive_only=config["positive_only"],
        min_mask_sum=config["min_mask_sum"],
    )

    model = build_sam_finetune_mask_decoder(config["model_name"])
    model.to(device)
               
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"]
    )

    for loss_name in config["loss_names"]:
        loss_fn = get_loss_function(loss_name)

        best_val_loss = float("inf")

        checkpoint_path = os.path.join(
            config["checkpoint_dir"],
            loss_name, "best_model.pt")
        
        for epoch in range(config["epochs"]):

            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss = validate_one_epoch(model, val_loader, loss_fn, device)

            print(
                f"{loss_name} | "
                f"Epoch {epoch+1} | "
                f"Train {train_loss:.4f} | "
                f"Val {val_loss:.4f}"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                    
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    loss_name=loss_name,
                    checkpoint_dir=checkpoint_path
                )
                print(f"Saved best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()