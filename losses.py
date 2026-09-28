import torch
import torch.nn as nn
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, TverskyLoss
from monai.losses.cldice import soft_skel   # differentiable skeleton (MONAI 1.5.2)


# ---------------------------------------------------------------------------
# Losses that MONAI does not provide in a single-channel-friendly form
# ---------------------------------------------------------------------------

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss = (1 - Tversky index) ** gamma   (Abraham & Khan, 2019).
    The Tversky part is MONAI's TverskyLoss; only the focal exponent is added here.
    alpha weights false positives, beta weights false negatives.
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75):
        super().__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(sigmoid=True, alpha=alpha, beta=beta, reduction="none")

    def forward(self, pred_logits, target):
        tversky_loss = self.tversky(pred_logits, target.float())      # (B, 1, 1, 1) = 1 - TI
        # clamp: x ** 0.75 has an infinite gradient at exactly 0
        return tversky_loss.clamp_min(1e-7).pow(self.gamma).mean()


class SoftclDiceLoss(nn.Module):
    """
    Soft clDice loss (Shit et al., CVPR 2021) for SINGLE-CHANNEL logits.

    Uses MONAI's soft_skel for the differentiable skeleton. MONAI's own
    SoftclDiceLoss is not used directly because in released versions (<= 1.5)
    it drops channel 0 as "background" ([:, 1:, ...]), applies no sigmoid and
    takes (y_true, y_pred) - with our 1-channel vessel output it would compute
    nothing useful. Checked against the MONAI 1.5.2 source.
    """

    def __init__(self, iter_=3, smooth=1.0):
        super().__init__()
        self.iter_ = iter_
        self.smooth = smooth

    def forward(self, pred_logits, target):
        prob = torch.sigmoid(pred_logits)
        target = target.float()

        skel_pred = soft_skel(prob, self.iter_)
        skel_true = soft_skel(target, self.iter_)

        dims = tuple(range(1, prob.ndim))   # per sample
        # topology precision: how much of the predicted skeleton lies inside the GT vessel
        tprec = ((skel_pred * target).sum(dims) + self.smooth) / (skel_pred.sum(dims) + self.smooth)
        # topology sensitivity: how much of the GT skeleton is covered by the prediction
        tsens = ((skel_true * prob).sum(dims) + self.smooth) / (skel_true.sum(dims) + self.smooth)

        cl_dice = 2.0 * tprec * tsens / (tprec + tsens)
        return (1.0 - cl_dice).mean()


class DiceclDiceLoss(nn.Module):
    """(1 - alpha) * Dice + alpha * clDice, as recommended in the clDice paper."""

    def __init__(self, alpha=0.5, iter_=3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss(sigmoid=True, reduction="mean")
        self.cldice = SoftclDiceLoss(iter_=iter_, smooth=smooth)

    def forward(self, pred_logits, target):
        target = target.float()
        return ((1.0 - self.alpha) * self.dice(pred_logits, target)
                + self.alpha * self.cldice(pred_logits, target))


class BCELoss(nn.Module):
    """nn.BCEWithLogitsLoss that accepts float/uint8 targets (MONAI has no plain BCE loss)."""

    def __init__(self, pos_weight=None):
        super().__init__()
        pw = None if pos_weight is None else torch.tensor([float(pos_weight)])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, pred_logits, target):
        if self.bce.pos_weight is not None and self.bce.pos_weight.device != pred_logits.device:
            self.bce.to(pred_logits.device)
        return self.bce(pred_logits, target.float())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
# All inputs: pred_logits (B, 1, H, W) raw logits, target (B, 1, H, W) in {0, 1}.
# Channel 0 is the VESSEL channel -> keep include_background=True (MONAI default).
# squared_pred is left at the default (False) for all Dice-type losses so they
# use the same Dice formulation and the comparison between losses is fair.

def get_loss_function(loss_name):
    if loss_name == "dice":
        return DiceLoss(sigmoid=True, reduction="mean")

    elif loss_name in ("dice_ce", "dice_ce_new"):
        return DiceCELoss(sigmoid=True, reduction="mean")        # Dice + BCE (1-channel -> BCE)

    elif loss_name == "dice_focal":
        return DiceFocalLoss(sigmoid=True, reduction="mean")     # Dice + sigmoid focal (gamma=2)

    elif loss_name == "tversky":
        return TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7, reduction="mean")

    elif loss_name == "focal_tversky":
        return FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)

    elif loss_name == "ce":
        return BCELoss()

    elif loss_name == "cldice":
        return SoftclDiceLoss(iter_=3, smooth=1.0)

    elif loss_name == "dice_cldice":
        return DiceclDiceLoss(alpha=0.5, iter_=3, smooth=1.0)

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# ---------------------------------------------------------------------------
# Metrics (unchanged)
# ---------------------------------------------------------------------------

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
