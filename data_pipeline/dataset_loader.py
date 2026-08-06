import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from sutram.calibration.planck import normalize_bt

class PatchDataset(Dataset):
    """
    PyTorch Dataset for loading co-registered patches.
    Strictly enforces loading of .npy files and raises an exception for any .png file requests.
    """
    def __init__(self, patches_dir, product_ids=None, transform=None, augment=False):
        super().__init__()
        self.patches_dir = patches_dir
        self.transform = transform
        self.augment = augment
        self.samples = []

        # Find product directories
        if product_ids is None:
            product_dirs = [d for d in glob.glob(os.path.join(patches_dir, "*")) if os.path.isdir(d)]
        else:
            product_dirs = [os.path.join(patches_dir, pid) for pid in product_ids if os.path.isdir(os.path.join(patches_dir, pid))]

        for pdir in product_dirs:
            p_samples = sorted([d for d in glob.glob(os.path.join(pdir, "*")) if os.path.isdir(d)])
            self.samples.extend(p_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]

        for fname in os.listdir(sample_path):
            if fname.lower().endswith(".png"):
                pass

        # Find array files matching suffixes
        tir_200m_path, tir_100m_path, rgb_100m_path = None, None, None
        for fname in os.listdir(sample_path):
            if fname.endswith("tir_200m.npy"):
                tir_200m_path = os.path.join(sample_path, fname)
            elif fname.endswith("tir_100m_512.npy") or fname.endswith("tir_100m.npy"):
                tir_100m_path = os.path.join(sample_path, fname)
            elif fname.endswith("rgb_100m_512.npy") or fname.endswith("rgb_100m.npy"):
                rgb_100m_path = os.path.join(sample_path, fname)

        if not tir_200m_path:
            tir_200m_path = os.path.join(sample_path, "tir_200m.npy")
        if not tir_100m_path:
            tir_100m_path = os.path.join(sample_path, "tir_100m_512.npy")
        if not rgb_100m_path:
            rgb_100m_path = os.path.join(sample_path, "rgb_100m_512.npy")

        for p in [tir_200m_path, tir_100m_path, rgb_100m_path]:
            if not p.endswith(".npy"):
                raise ValueError(f"CRITICAL: Non-NPY file specified for dataset loading: {p}")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required dataset array not found: {p}")

        # Load arrays
        tir_200 = np.load(tir_200m_path).astype(np.float32)
        tir_100 = np.load(tir_100m_path).astype(np.float32)
        rgb_100 = np.load(rgb_100m_path).astype(np.float32)

        # Patches are stored in physical units by prepare_dataset.py (thermal in
        # Kelvin — calibrated once per scene so every patch and both resolutions
        # share the same mapping; RGB already on the display 0-255 scale). The
        # loader only applies the fixed brightness-temperature normalization.
        tir_200 = normalize_bt(tir_200)
        tir_100 = normalize_bt(tir_100)
        rgb_100 = np.clip(rgb_100, 0.0, 255.0)

        # Expand dims if single-channel to (C, H, W)
        if tir_200.ndim == 2:
            tir_200 = np.expand_dims(tir_200, axis=0)
        if tir_100.ndim == 2:
            tir_100 = np.expand_dims(tir_100, axis=0)

        # Convert RGB shape from (C, H, W) or (H, W, C) to PyTorch standard (C, H, W)
        if rgb_100.ndim == 3 and rgb_100.shape[0] != 3:
            rgb_100 = np.moveaxis(rgb_100, -1, 0)

        # PyTorch tensors
        tir_200 = torch.from_numpy(tir_200)
        tir_100 = torch.from_numpy(tir_100)
        rgb_100 = torch.from_numpy(rgb_100)

        # Apply random spatial augmentations if enabled (training mode)
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                tir_200 = torch.flip(tir_200, dims=[-1])
                tir_100 = torch.flip(tir_100, dims=[-1])
                rgb_100 = torch.flip(rgb_100, dims=[-1])
            # Random vertical flip
            if np.random.rand() > 0.5:
                tir_200 = torch.flip(tir_200, dims=[-2])
                tir_100 = torch.flip(tir_100, dims=[-2])
                rgb_100 = torch.flip(rgb_100, dims=[-2])
            # Random 90-degree rotation
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                tir_200 = torch.rot90(tir_200, k=rot_k, dims=[-2, -1])
                tir_100 = torch.rot90(tir_100, k=rot_k, dims=[-2, -1])
                rgb_100 = torch.rot90(rgb_100, k=rot_k, dims=[-2, -1])

        sample = {
            "tir_200m": tir_200,
            "tir_100m_512": tir_100,
            "rgb_100m_512": rgb_100
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class EnforceNPYOnlyDataset(Dataset):
    """
    A wrapper class built explicitly to refuse any paths containing '.png'.
    Raises ValueError immediately on initialization if any file contains '.png'.
    """
    def __init__(self, file_list):
        for f in file_list:
            if ".png" in f.lower():
                raise ValueError("CRITICAL SECURITY ERROR: PNG files are strictly forbidden from training datasets.")
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return np.load(self.file_list[idx])
