import numpy as np
import torch

from sutram.calibration.planck import dn_to_brightness_temp, brightness_temp_to_dn

TB_MIN, TB_MAX = 278.30386, 314.5386

def normalize_tir(arr, dtype=None):
    """
    Centralized TIR (B10) normalization to [0.0, 1.0].
    If 16-bit raw Landsat, converts raw DN to Brightness Temperature (Kelvin)
    via Planck calibration, then normalizes to [0.0, 1.0].
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
        # Convert to Brightness Temperature first (Kelvin)
        tb = dn_to_brightness_temp(arr)
        if is_tensor:
            return torch.clamp((tb - TB_MIN) / (TB_MAX - TB_MIN), 0.0, 1.0)
        else:
            return np.clip((tb - TB_MIN) / (TB_MAX - TB_MIN), 0.0, 1.0)

def denormalize_tir(arr, scale_mode):
    """
    Centralized TIR (B10) denormalization.
    Inverts normalize_tir back to raw range (either normalized, 8bit, or 16bit raw).
    """
    is_tensor = isinstance(arr, torch.Tensor)
    if scale_mode == "normalized":
        if is_tensor:
            return torch.clamp(arr, 0.0, 1.0)
        return np.clip(arr, 0.0, 1.0)
    elif scale_mode == "8bit":
        if is_tensor:
            return torch.clamp(arr * 255.0, 0.0, 255.0)
        return np.clip(arr * 255.0, 0.0, 255.0)
    else:
        # 16-bit Landsat raw: [0, 1] -> Kelvin -> DN
        if is_tensor:
            tb = arr * (TB_MAX - TB_MIN) + TB_MIN
            dn = brightness_temp_to_dn(tb)
            return torch.clamp(dn, 20000.0, 35000.0)
        else:
            tb = arr * (TB_MAX - TB_MIN) + TB_MIN
            dn = brightness_temp_to_dn(tb)
            return np.clip(dn, 20000.0, 35000.0)


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
