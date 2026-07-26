import os
import glob
import random
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.normalization import normalize_tir, normalize_rgb

class PatchDataset(Dataset):
    """
    PyTorch Dataset for loading co-registered patches.
    Strictly enforces loading of .npy files and raises an exception for any .png file requests.
    """
    def __init__(self, patches_dir: str, product_ids: Optional[List[str]] = None, transform: Any = None, augment: bool = False):
        super().__init__()
        self.patches_dir = patches_dir
        self.transform = transform
        self.augment = augment
        self.samples: List[str] = []

        # Find product directories matching exact configured ID
        if product_ids is None:
            product_dirs = [d for d in glob.glob(os.path.join(patches_dir, "*")) if os.path.isdir(d)]
        else:
            product_dirs = []
            for pid in product_ids:
                pdir = os.path.join(patches_dir, pid)
                if not os.path.exists(pdir) or not os.path.isdir(pdir):
                    raise FileNotFoundError(f"CRITICAL: Configured scene directory '{pid}' could not be found under {patches_dir}")
                product_dirs.append(pdir)

        for pdir in product_dirs:
            p_samples = sorted(glob.glob(os.path.join(pdir, "*_patch_*")))
            # Verify PNGs on init once per dataset build (not inside __getitem__)
            for s_path in p_samples:
                for fname in os.listdir(s_path):
                    if fname.lower().endswith(".png"):
                        raise ValueError(f"PNG images are strictly prohibited in the dataset split: {fname}")
                self.samples.append(s_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: Any) -> Dict[str, torch.Tensor]:
        sample_path = self.samples[index]
        patch_name = os.path.basename(sample_path)

        tir_200m_path = os.path.join(sample_path, f"{patch_name}_tir_200m.npy")
        tir_100m_path = os.path.join(sample_path, f"{patch_name}_tir_100m.npy")
        rgb_100m_path = os.path.join(sample_path, f"{patch_name}_rgb_100m.npy")

        for p in [tir_200m_path, tir_100m_path, rgb_100m_path]:
            if not p.endswith(".npy"):
                raise ValueError(f"CRITICAL: Non-NPY file specified for dataset loading: {p}")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required dataset array not found: {p}")

        # Load arrays
        tir_200 = np.load(tir_200m_path).astype(np.float32)
        tir_100 = np.load(tir_100m_path).astype(np.float32)
        rgb_100 = np.load(rgb_100m_path).astype(np.float32)

        # Centralized normalization
        tir_200 = normalize_tir(tir_200)
        tir_100 = normalize_tir(tir_100)
        rgb_100 = normalize_rgb(rgb_100)

        # Expand dims if single-channel to (C, H, W)
        if tir_200.ndim == 2:
            tir_200 = np.expand_dims(tir_200, axis=0)
        if tir_100.ndim == 2:
            tir_100 = np.expand_dims(tir_100, axis=0)

        # Convert RGB shape from (H, W, C) to PyTorch standard (C, H, W)
        if rgb_100.ndim == 3 and rgb_100.shape[0] != 3:
            rgb_100 = np.ascontiguousarray(np.transpose(rgb_100, (2, 0, 1)))

        # PyTorch tensors
        t_tir_200 = torch.from_numpy(tir_200)
        t_tir_100 = torch.from_numpy(tir_100)
        t_rgb_100 = torch.from_numpy(rgb_100)

        # Apply random spatial augmentations if enabled (training mode)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                t_tir_200 = torch.flip(t_tir_200, dims=[-1])
                t_tir_100 = torch.flip(t_tir_100, dims=[-1])
                t_rgb_100 = torch.flip(t_rgb_100, dims=[-1])
            # Random vertical flip
            if random.random() > 0.5:
                t_tir_200 = torch.flip(t_tir_200, dims=[-2])
                t_tir_100 = torch.flip(t_tir_100, dims=[-2])
                t_rgb_100 = torch.flip(t_rgb_100, dims=[-2])
            # Random 90-degree rotation
            rot_k = random.randint(0, 3)
            if rot_k > 0:
                t_tir_200 = torch.rot90(t_tir_200, k=rot_k, dims=[-2, -1])
                t_tir_100 = torch.rot90(t_tir_100, k=rot_k, dims=[-2, -1])
                t_rgb_100 = torch.rot90(t_rgb_100, k=rot_k, dims=[-2, -1])

        sample: Dict[str, torch.Tensor] = {
            "tir_200m": t_tir_200,
            "tir_100m_512": t_tir_100,
            "rgb_100m_512": t_rgb_100
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class EnforceNPYOnlyDataset(Dataset):
    """
    A wrapper class built explicitly to refuse any paths containing '.png'.
    Raises ValueError immediately on initialization if any file contains '.png'.
    """
    def __init__(self, file_list: List[str]):
        for f in file_list:
            if ".png" in f.lower():
                raise ValueError("CRITICAL SECURITY ERROR: PNG files are strictly forbidden from training datasets.")
        self.file_list = file_list

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: Any) -> Any:
        return np.load(self.file_list[index])
