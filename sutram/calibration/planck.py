import numpy as np

try:
    import torch as _torch
except ImportError:
    _torch = None

def _is_tensor(x):
    return _torch is not None and isinstance(x, _torch.Tensor)

# Standard Landsat 9 Band 10 calibration constants
ML_DEFAULT = 0.0003342  # Radiance multiplicative scaling factor
AL_DEFAULT = 0.1        # Radiance additive scaling factor
K1_DEFAULT = 774.89     # Calibration constant 1
K2_DEFAULT = 1321.07    # Calibration constant 2

def dn_to_radiance(dn, ml=ML_DEFAULT, al=AL_DEFAULT):
    """
    Converts Digital Numbers (DN) to spectral radiance.
    L_lambda = ML * DN + AL
    """
    if _is_tensor(dn):
        return ml * dn.float() + al
    return ml * np.asarray(dn, dtype=np.float32) * 1.0 + al

def brightness_temp_to_dn(bt, ml=ML_DEFAULT, al=AL_DEFAULT, k1=K1_DEFAULT, k2=K2_DEFAULT):
    """
    Inverse conversion from Brightness Temperature (Kelvin) back to raw DN.
    """
    if _is_tensor(bt):
        radiance = k1 / (_torch.exp(k2 / _torch.clamp(bt, min=1e-6)) - 1.0)
        return (radiance - al) / ml
    bt_np = np.asarray(bt, dtype=np.float32)
    radiance = k1 / (np.exp(k2 / np.clip(bt_np, 1e-6, None)) - 1.0)
    return (radiance - al) / ml

def radiance_to_brightness_temp(radiance, k1=K1_DEFAULT, k2=K2_DEFAULT):
    """
    Converts spectral radiance to Brightness Temperature (T_B) in Kelvin.
    T = K2 / ln((K1 / L_lambda) + 1)
    """
    # Avoid log of zero or negative radiance
    safe_radiance = np.clip(radiance, 1e-6, None)
    return k2 / np.log((k1 / safe_radiance) + 1.0)

def dn_to_brightness_temp(dn, ml=ML_DEFAULT, al=AL_DEFAULT, k1=K1_DEFAULT, k2=K2_DEFAULT):
    """
    Full Stage 0 conversion from raw DN to Brightness Temperature (Kelvin).
    """
    radiance = dn_to_radiance(dn, ml, al)
    return radiance_to_brightness_temp(radiance, k1, k2)

# ----------------------------------------------------------------------------
# Brightness-temperature normalization
# ----------------------------------------------------------------------------
# The network operates in physical brightness-temperature space: Band-10 DN is
# Planck-inverted to Kelvin and then linearly scaled to [0, 1] using this
# land-surface BT range before it enters the backbone / SR head. Keeping the
# range fixed and physical makes the "Planck inversion feeds super-resolution"
# path literal and lets outputs be denormalized straight back to Kelvin.
#
# The range is tuned to real daytime land-surface brightness temperatures. An
# over-wide range (e.g. 250-330 K) squeezes typical ~297-309 K scenes into a
# tiny sub-band (~15% of [0,1]), starving the network of contrast and washing
# out the super-resolution and colour outputs. 290-315 K keeps the full
# dynamic range usable while still covering realistic land scenes; widen it
# only if you expect very cold (water/ice) or very hot surfaces.
TB_MIN = 290.0  # Kelvin
TB_MAX = 315.0  # Kelvin

def normalize_bt(bt, tb_min=TB_MIN, tb_max=TB_MAX):
    """Scale brightness temperature (Kelvin) to [0, 1] for the network input."""
    return np.clip((bt - tb_min) / (tb_max - tb_min), 0.0, 1.0)

def denormalize_bt(x, tb_min=TB_MIN, tb_max=TB_MAX):
    """Inverse of normalize_bt: map a [0, 1] value back to Kelvin."""
    return x * (tb_max - tb_min) + tb_min
