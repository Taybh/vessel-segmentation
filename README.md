# SAM Vessel Segmentation

Cerebral vessel segmentation from 3D NIfTI volumes with the **original 2D Segment Anything Model (SAM, ViT-B)**, fine-tuned with parameter-efficient methods.

SAM is applied **slice-wise / patch-wise**: each 256×256 patch of an axial slice is repeated to 3 channels (RGB-like input, still 2D) and segmented with a **bounding-box prompt** derived from the ground-truth vessel mask.

```
3D NIfTI volume → 2D patch (256×256) → 3-channel copy → SamProcessor
→ SAM vision encoder → box prompt → prompt encoder → mask decoder → 2D vessel mask
```

## Current experiment

Compare **segmentation losses** for **LoRA** fine-tuning of SAM (LoRA r=8 on the `qkv` attention layers of the image encoder; mask decoder frozen):

`dice`, `ce` (BCE), `dice_ce`, `dice_focal`, `tversky`, `focal_tversky`, `cldice`, `dice_cldice`

The best 1–3 losses will later be used to compare other backbones and fine-tuning methods (e.g. adapters).

## What the training does

- **Data:** volumes are split into train/val **at volume level** (80/20, fixed seed). Only patches with ≥ 20 vessel pixels are used. Images are min-max normalized per patch to [0, 1] (`do_rescale=False` in the processor, so they are not divided by 255 again).
- **Prompts:** training boxes are randomly enlarged by 0–19 px (augmentation); validation boxes are tight and fixed, so the validation loss is comparable between epochs.
- **Fair comparison:** every loss starts from a fresh SAM model and optimizer with the **same seed**, same learning rate (1e-4, AdamW), batch size and number of epochs.
- **No early stopping:** every run trains for the full number of epochs (50). Validation runs every epoch on **all** validation batches.
- **Checkpoints** (per loss):
  - `best_model.pt` – epoch with the lowest validation loss (used for testing)
  - `last_model.pt` – saved every epoch, used to resume or extend training
- **Monitoring** (per loss): `training_log.txt`, `history.csv`, `curves.png`, `summary.json`; each epoch logs train/val loss, val Dice, val IoU and the predicted vs. ground-truth vessel fraction (warns if the model collapses to empty masks). `experiments_summary.csv` compares all losses.
- **Devices:** uses an NVIDIA GPU (CUDA) if available, otherwise an Apple GPU (MPS) or CPU. On Mac/CPU it runs a short debug run (`debug_epochs`, output in `<checkpoint_dir>_debug`).

## Files

| File | Content |
|---|---|
| `config.yaml` | all settings (paths, losses, epochs, learning rate, batch size, …) |
| `train.py` | training loop, validation, logging, checkpoints, resume |
| `dataset.py` | NIfTI loading, patch extraction, box prompts, data loaders |
| `model.py` | SAM builders: LoRA, adapter, mask-decoder fine-tuning |
| `losses.py` | loss functions (MONAI-based) and Dice/IoU metrics |
| `utils.py` | config, seed, device selection, checkpoint saving/loading |

## Setup

```bash
cd vessel_segmentation
python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**GPU server only:** the server's own cuDNN (in `LD_LIBRARY_PATH`) conflicts with the cuDNN installed with PyTorch. Add this once to the environment so it is set on every activation:

```bash
echo 'export LD_LIBRARY_PATH=/scratch/tbahador/python/lib' >> myenv/bin/activate
```

## Run the training

1. Check `config.yaml` (at least `data_dir`, `checkpoint_dir`, `loss_names`, `epochs`, `batch_size`).
2. Run:

```bash
source myenv/bin/activate
python3 train.py
```

**Resume / extend:** set `resume: true` (and optionally a higher `epochs`) in `config.yaml` and run `python train.py` again. Finished losses are skipped; unfinished ones continue from `last_model.pt`.

**Notes:** on the RTX 2080 Ti (11 GB) only `batch_size: 1` fits. 