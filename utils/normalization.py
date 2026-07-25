import numpy as np
import torch

def normalize_tir(arr, dtype=None):
    """
    Centralized TIR (B10) normalization to [0.0, 1.0].
    Handles numpy arrays and torch Tensors.
    """
    is_tensor = isinstance(arr, torch.Tensor)
    
    # Determine max value dynamically or from dtype
    if dtype is None:
        if is_tensor:
            arr_max = float(arr.max().item())
        else:
            arr_max = float(arr.max())
    else:
        dt_str = str(dtype).lower()
        if "uint8" in dt_str:
            arr_max = 255.0
        elif "uint16" in dt_str:
            arr_max = 65535.0
        else:
            if is_tensor:
                arr_max = float(arr.max().item())
            else:
                arr_max = float(arr.max())
                
    # Scale based on range
    if arr_max <= 1.0:
        # Already normalized
        if is_tensor:
            return torch.clamp(arr, 0.0, 1.0)
        else:
            return np.clip(arr, 0.0, 1.0)
    elif arr_max <= 255.0:
        # 8-bit visual
        if is_tensor:
            return torch.clamp(arr / 255.0, 0.0, 1.0)
        else:
            return np.clip(arr / 255.0, 0.0, 1.0)
    else:
        # 16-bit Landsat raw
        TIR_MIN, TIR_MAX = 20000.0, 35000.0
        if is_tensor:
            return torch.clamp((arr - TIR_MIN) / (TIR_MAX - TIR_MIN), 0.0, 1.0)
        else:
            return np.clip((arr - TIR_MIN) / (TIR_MAX - TIR_MIN), 0.0, 1.0)

def normalize_rgb(arr, dtype=None):
    """
    Centralized RGB (B2/B3/B4) normalization to [0.0, 255.0].
    Handles numpy arrays and torch Tensors.
    """
    is_tensor = isinstance(arr, torch.Tensor)
    
    # Determine max value
    if dtype is None:
        if is_tensor:
            arr_max = float(arr.max().item())
        else:
            arr_max = float(arr.max())
    else:
        dt_str = str(dtype).lower()
        if "uint8" in dt_str:
            arr_max = 255.0
        elif "uint16" in dt_str:
            arr_max = 65535.0
        else:
            if is_tensor:
                arr_max = float(arr.max().item())
            else:
                arr_max = float(arr.max())
                
    if arr_max > 255.0:
        # Landsat 16-bit Surface Reflectance (target range 0-10000)
        RGB_SCALE = 10000.0
        if is_tensor:
            return torch.clamp((arr / RGB_SCALE) * 255.0, 0.0, 255.0)
        else:
            return np.clip((arr / RGB_SCALE) * 255.0, 0.0, 255.0)
    else:
        # Already in 8-bit visual range [0, 255] or float
        if is_tensor:
            return torch.clamp(arr, 0.0, 255.0)
        else:
            return np.clip(arr, 0.0, 255.0)
