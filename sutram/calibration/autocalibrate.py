"""
Auto-calibration — single source of truth for turning whatever thermal / optical
raster arrives into the physical (or, where no radiometry exists, honestly
display-calibrated) quantities the network consumes.

Used identically by the dataset loader, the inference pipeline's callers and the
web dashboard so training and serving can never drift apart. The functions here
are numpy-based and auto-detect the input encoding at runtime; the network graph
itself (backbone / SR / mixture) consumes an already-normalized [0, 1]
brightness-temperature tensor, which keeps it input-type agnostic and
ONNX-portable.

Three thermal encodings are recognised:
  * 8-bit browse thermal (<= 255)      -> no radiometry; display-calibrated into a
                                          plausible land-surface BT window. Callers
                                          must label this an estimate, not physics.
  * L1 Band-10 DN (uint16, ~20k-40k)   -> real Planck inversion (dn_to_brightness_temp)
  * L2 ST_B10 surface temp (uint16)    -> real L2 scaling 0.00341802*DN + 149

RGB encodings:
  * 8-bit browse RGB (<= 255)          -> passthrough (already display 0-255)
  * synthetic reflectance (0..~10000)  -> /10000 * 255
  * L2 SR reflectance (uint16)         -> 0.0000275*DN - 0.2, stretched to 0-255
"""

import numpy as np

from .planck import dn_to_brightness_temp, normalize_bt, TB_MIN, TB_MAX

# Display-calibration window for browse thermal (no radiometry available).
# Kept inside the training BT range [TB_MIN, TB_MAX] so normalize_bt does not clip.
BROWSE_BT_LO = 292.0  # Kelvin
BROWSE_BT_HI = 312.0  # Kelvin

# L2 Collection-2 Level-2 product scaling constants.
ST_SCALE, ST_OFFSET = 0.00341802, 149.0          # ST_B10 surface temperature (K)
SR_SCALE, SR_OFFSET = 0.0000275, -0.2            # SR_B* surface reflectance


def _classify_thermal(a):
    """Return one of 'browse' | 'l2_st' | 'l1_dn' for a thermal array."""
    amax = float(np.nanmax(a))
    if amax <= 255.0:
        return "browse"
    # L2 ST_B10 encodes ~150-330 K as 0.00341802*DN + 149, i.e. DN ~ 300-53000
    # for 150-330 K; L1 B10 DN for land temps sits ~20k-35k. They overlap, so we
    # disambiguate on whether the L2 decode lands in a physical LST band.
    st_k = a.astype(np.float32) * ST_SCALE + ST_OFFSET
    st_med = float(np.nanmedian(st_k[a > 0])) if np.any(a > 0) else float(np.nanmedian(st_k))
    if 240.0 <= st_med <= 340.0 and amax > 40000.0:
        return "l2_st"
    return "l1_dn"


def fit_thermal_calibration(reference):
    """Fit calibration parameters on a REFERENCE array (e.g. the full scene) so
    the same mapping can be applied consistently to every patch / resolution cut
    from that scene. For physical encodings (L1 DN, L2 ST) the mapping is global
    and carries no fitted state; only the browse branch needs the scene's
    percentile window."""
    a = np.asarray(reference, dtype=np.float32)
    kind = _classify_thermal(a)
    if kind != "browse":
        return {"kind": kind}
    valid = a[a > 0]
    if valid.size < 16:
        valid = a
    lo, hi = float(np.percentile(valid, 1)), float(np.percentile(valid, 99))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return {"kind": "browse", "lo": lo, "hi": hi}


def apply_thermal_calibration(arr, params):
    """Apply a fitted calibration to an array -> brightness temperature (K)."""
    a = np.asarray(arr, dtype=np.float32)
    kind = params["kind"]
    if kind == "browse":
        lo, hi = params["lo"], params["hi"]
        t = (np.clip(a, lo, hi) - lo) / (hi - lo)
        return (BROWSE_BT_LO + t * (BROWSE_BT_HI - BROWSE_BT_LO)).astype(np.float32)
    if kind == "l2_st":
        return (a * ST_SCALE + ST_OFFSET).astype(np.float32)
    return dn_to_brightness_temp(a).astype(np.float32)


def calibrate_thermal_to_bt(arr):
    """Thermal raster -> brightness temperature in Kelvin (float32), auto-detecting
    the input encoding (fit + apply on the same array; for multi-patch scenes,
    fit once on the scene with fit_thermal_calibration and apply per patch)."""
    a = np.asarray(arr, dtype=np.float32)
    return apply_thermal_calibration(a, fit_thermal_calibration(a))


def calibrate_thermal_to_norm(arr):
    """Convenience: thermal raster -> normalized BT in [0, 1] (network input)."""
    return normalize_bt(calibrate_thermal_to_bt(arr)).astype(np.float32)


def thermal_is_physical(arr):
    """True when the thermal input carries real radiometry (L1/L2), False for
    8-bit browse data. Lets the UI label temperatures honestly."""
    return _classify_thermal(np.asarray(arr, dtype=np.float32)) != "browse"


def calibrate_rgb_to_display(rgb):
    """Optical raster -> display RGB on a 0-255 scale (float32), auto-detecting
    the encoding. Accepts (3, H, W) or (H, W, 3); returns the same layout."""
    a = np.asarray(rgb, dtype=np.float32)
    amax = float(np.nanmax(a))
    if amax <= 255.0 + 1e-3:
        return np.clip(a, 0.0, 255.0)                       # browse RGB (already display)
    if amax <= 15000.0:
        return np.clip(a / 10000.0 * 255.0, 0.0, 255.0)     # synthetic reflectance 0-10000
    refl = a * SR_SCALE + SR_OFFSET                          # L2 SR uint16 reflectance
    return np.clip(refl / 0.35 * 255.0, 0.0, 255.0)
