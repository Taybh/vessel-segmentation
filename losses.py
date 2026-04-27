import torch
from monai.losses import DiceCELoss, DiceFocalLoss, TverskyLoss

def get_loss_function(loss_name):
    if loss_name == "dice_ce":
        return DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean", include_background=False)

    if loss_name == "dice_focal":
        return DiceFocalLoss(sigmoid=True, squared_pred=True, reduction="mean")

    if loss_name == "tversky":
        return TverskyLoss(
            sigmoid=True,
            alpha=0.3,
            beta=0.7,
            reduction="mean"
        )

    raise ValueError(f"Unknown loss: {loss_name}")

def dice_score(pred_logits, gt_mask, threshold=0.5, eps=1e-7):
    pred_prob = torch.sigmoid(pred_logits)
    pred_bin = (pred_prob > threshold).float()
    gt_mask = gt_mask.float()

    intersection = (pred_bin * gt_mask).sum(dim=(1, 2, 3))
    denominator = pred_bin.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (denominator + eps)
    return dice.mean().item()


def iou_score(pred_logits, gt_mask, threshold=0.5, eps=1e-7):
    pred_prob = torch.sigmoid(pred_logits)
    pred_bin = (pred_prob > threshold).float()
    gt_mask = gt_mask.float()

    intersection = (pred_bin * gt_mask).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()