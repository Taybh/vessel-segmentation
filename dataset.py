import nibabel as nib
from pathlib import Path
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader
import torch


def strip_nii_suffix(name: str) -> str:
    """
    Remove .nii or .nii.gz from a filename.
    """
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name

def normalize_image(slice_2d: np.ndarray) -> np.ndarray:
    """
    Simple min-max normalization per slice.
    """
    slice_2d = slice_2d.astype(np.float32)
    vmin = slice_2d.min()
    vmax = slice_2d.max()
    if vmax > vmin:
        slice_2d = (slice_2d - vmin) / (vmax - vmin)
    else:
        slice_2d = np.zeros_like(slice_2d, dtype=np.float32)
    return slice_2d

def load_nifti_image(file_path: Path, is_mask: bool = False) -> np.ndarray:
    """
    Memory-friendlier NIfTI loading.
    - images -> float32
    - masks  -> uint8
    """
    nii = nib.load(str(file_path))
    data = np.asanyarray(nii.dataobj)

    if is_mask:
        return data.astype(np.uint8)
    return data.astype(np.float32)

def build_patch_index_for_volume(
    shape: Tuple[int, int, int],
    patch_size: int = 256,
    step: int = 256
) -> List[Tuple[int, int, int]]:
    """
    Build a list of (z, x, y) patch coordinates for one volume.
    Assumes slice axis is axis=2 (z).
    """
    x_dim, y_dim, z_dim = shape
    index = []

    for z in range(z_dim):
        for x in range(0, x_dim - patch_size + 1, step):
            for y in range(0, y_dim - patch_size + 1, step):
                index.append((z, x, y))  

    return index

def find_data_mask_pairs(data_dir: str) -> List[Tuple[Path, Path]]:
    """
    Find .nii.gz data files and .nii mask files and match them by basename.
    """
    root = Path(data_dir)

    data_files = sorted(root.rglob("*.nii.gz"))
    mask_files = sorted([f for f in root.rglob("*.nii") if not str(f).endswith(".nii.gz")])

    data_map: Dict[str, Path] = {strip_nii_suffix(f.name): f for f in data_files}
    mask_map: Dict[str, Path] = {strip_nii_suffix(f.name): f for f in mask_files}

    common_keys = sorted(set(data_map.keys()) & set(mask_map.keys()))
    pairs = [(data_map[k], mask_map[k]) for k in common_keys]

    print(f"Found {len(data_files)} data files")
    print(f"Found {len(mask_files)} mask files")
    print(f"Matched {len(pairs)} data-mask pairs")

    missing_data = sorted(set(mask_map.keys()) - set(data_map.keys()))
    missing_masks = sorted(set(data_map.keys()) - set(mask_map.keys()))

    if missing_data:
        print(f"Warning: {len(missing_data)} masks without matching data")
    if missing_masks:
        print(f"Warning: {len(missing_masks)} data files without matching masks")

    return pairs

def split_train_val(
    pairs: List[Tuple[Path, Path]],
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Split at volume level.
    """
    pairs = pairs.copy()
    rng = random.Random(seed)
    rng.shuffle(pairs)

    split_idx = int(len(pairs) * (1.0 - val_ratio))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    return train_pairs, val_pairs

#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  # handle empty mask
  if len(x_indices) == 0:
    H, W = ground_truth_map.shape
    return [0, 0, W - 1, H - 1]


  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)

  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W-1, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H-1, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

class PatchDataset(Dataset):
    """
    Dataset that:
    - reads NIfTI image/mask pairs
    - extracts 2D patches on the fly
    - computes bounding box prompt from the mask
    - applies a SAM processor
    - returns inputs + ground truth mask
    """

    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        processor,
        get_bounding_box_fn,
        patch_size: int = 256,
        step: int = 256,
        normalize: bool = True,
        positive_only: bool = False,
        min_mask_sum: int = 1
    ):
        """
        Args:
            pairs: list of (image_path, mask_path)
            patch_size: 2D patch size
            step: stride between patches
            normalize: whether to min-max normalize each image patch/slice
            positive_only: if True, keep only patches with some mask foreground
            min_mask_sum: minimum number of positive pixels to keep a mask patch
        """
        self.pairs = pairs
        self.processor = processor
        self.get_bounding_box_fn = get_bounding_box_fn
        self.patch_size = patch_size
        self.step = step
        self.normalize = normalize
        self.positive_only = positive_only
        self.min_mask_sum = min_mask_sum

        # stores (volume_idx, z, x, y)
        self.patch_index: List[Tuple[int, int, int, int]] = []

        # Optional simple cache to avoid reloading the same volume repeatedly
        self._cached_volume_idx: Optional[int] = None
        self._cached_image: Optional[np.ndarray] = None
        self._cached_mask: Optional[np.ndarray] = None

        self._build_index()

    def _build_index(self):
        """
        Build patch index lazily from volume shapes only.
        This is much lighter than precomputing and storing actual patches.
        """
        for volume_idx, (img_path, mask_path) in enumerate(self.pairs):
            img = load_nifti_image(img_path, is_mask=False)
            mask = load_nifti_image(mask_path, is_mask=True)

            if img.shape != mask.shape:
                raise ValueError(
                    f"Shape mismatch for {img_path.name} and {mask_path.name}: "
                    f"{img.shape} vs {mask.shape}"
                )

            coords = build_patch_index_for_volume(
                img.shape,
                patch_size=self.patch_size,
                step=self.step
            )

            if self.positive_only:
                for z, x, y in coords:
                    mask_patch = mask[x:x+self.patch_size, y:y+self.patch_size, z]
                    if np.sum(mask_patch) >= self.min_mask_sum:
                        self.patch_index.append((volume_idx, z, x, y))
            else:
                for z, x, y in coords:
                    self.patch_index.append((volume_idx, z, x, y))

            # release immediately
            del img
            del mask

        print(f"Total patches indexed: {len(self.patch_index)}")

    def __len__(self):
        return len(self.patch_index)

    def _load_volume_if_needed(self, volume_idx: int):
        """
        Cache one volume pair at a time.
        Helpful when DataLoader accesses nearby indices.
        """
        if self._cached_volume_idx == volume_idx:
            return

        img_path, mask_path = self.pairs[volume_idx]
        self._cached_image = load_nifti_image(img_path, is_mask=False)
        self._cached_mask = load_nifti_image(mask_path, is_mask=True)
        self._cached_volume_idx = volume_idx

    def __getitem__(self, idx: int):
        volume_idx, z, x, y = self.patch_index[idx]

        self._load_volume_if_needed(volume_idx)

        image = self._cached_image
        mask = self._cached_mask

        # extract 2D patch from slice z
        img_patch = image[x:x+self.patch_size, y:y+self.patch_size, z]
        mask_patch = mask[x:x+self.patch_size, y:y+self.patch_size, z]

        if self.normalize:
            img_patch = normalize_image(img_patch)
        else:
            img_patch = img_patch.astype(np.float32)

        mask_patch = mask_patch.astype(np.uint8)

        # SAM usually expects image-like input, often HxW or HxWx3
        # If grayscale, repeat to 3 channels
        image_for_processor = img_patch
        if image_for_processor.ndim == 2:
            image_for_processor = np.stack([image_for_processor] * 3, axis=-1)

        ground_truth_mask = np.array(mask_patch)

        # get bounding box prompt
        prompt = self.get_bounding_box_fn(ground_truth_mask)

        # prepare image and prompt for the model
        inputs = self.processor(
            image_for_processor,
            input_boxes=[[prompt]],
            return_tensors="pt"
        )

        # remove batch dimension added by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask).float()

        return inputs

def create_datasets_and_loaders(
    data_dir: str,
    processor,
    val_ratio: float = 0.2,
    seed: int = 42,
    patch_size: int = 256,
    step: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    positive_only: bool = True,
    min_mask_sum: int = 1,
):
    """
    Full pipeline:
    1. find pairs
    2. split train/validation
    3. create datasets
    4. create dataloaders
    """
    pairs = find_data_mask_pairs(data_dir)
    train_pairs, val_pairs = split_train_val(pairs, val_ratio=val_ratio, seed=seed)

    print(f"Train volumes: {len(train_pairs)}")
    print(f"Val volumes:   {len(val_pairs)}")

    train_dataset = PatchDataset(
        pairs=train_pairs,
        processor=processor,
        get_bounding_box_fn=get_bounding_box,
        patch_size=patch_size,
        step=step,
        normalize=True,
        positive_only=positive_only,
        min_mask_sum=min_mask_sum,
    )

    val_dataset = PatchDataset(
        pairs=val_pairs,
        processor=processor,
        get_bounding_box_fn=get_bounding_box,
        patch_size=patch_size,
        step=step,
        normalize=True,
        positive_only=positive_only,
        min_mask_sum=min_mask_sum,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataset, val_dataset, train_loader, val_loader



if __name__ == "__main__":
    ### Test the dataset
    nii = nib.load('sub004.nii.gz')  # or .nii
    # Get image data as a NumPy array
    data = nii.get_fdata()
    print("Shape:", data.shape)
    print("Data type:", data.dtype)
    print("Orientation:", nib.aff2axcodes(nii.affine))

    # Show a middle slice 
    slice_idx = data.shape[2] // 2
    plt.imshow(data[:, :, slice_idx], cmap='gray')
    plt.title(f"Slice {slice_idx}")
    plt.axis('off')
    plt.show()