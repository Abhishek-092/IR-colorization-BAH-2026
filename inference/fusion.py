"""
Fused ground reconstruction
===========================
Combines EVERY signal the pipeline produces into one image that is the model's
best single estimate of the ground:

  * Super-resolved brightness temperature  -> spatial detail (luminance backbone)
  * Synthesized mixture RGB                -> colour (chroma), trust-weighted
  * Mixture uncertainty (entropy / conf.)  -> per-pixel trust in that colour

Method (luma/chroma fusion — sharper and cleaner than naive alpha blending):
the sharpened SR-thermal band IS the luminance of the output everywhere, so
structural detail is never diluted; the synthesized colour contributes only its
chroma (colour offset), scaled by a scene-relative trust map with adaptive
saturation. Where the mixture is ambiguous the chroma fades out and the pixel
degrades to crisp calibrated-thermal greyscale — confident abstention without
the washed-out look of direct blending.
"""

import numpy as np


def _stretch01(a, p_lo=1.0, p_hi=99.0):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, p_lo), np.percentile(a, p_hi)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _box_blur(a, r=2):
    """Separable box blur (two passes ~ Gaussian), pure numpy."""
    if r <= 0:
        return a
    k = 2 * r + 1
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(a, pad, mode="edge")
        c = np.cumsum(p, axis=axis, dtype=np.float32)
        take_hi = [slice(None)] * 2
        take_lo = [slice(None)] * 2
        take_hi[axis] = slice(k - 1, None)
        take_lo[axis] = slice(None, -k + 1) if k > 1 else slice(None)
        hi = c[tuple(take_hi)]
        lo = np.zeros_like(hi)
        lo_src = c[tuple(take_lo)]
        idx = [slice(None)] * 2
        idx[axis] = slice(1, None)
        lo[tuple(idx)] = lo_src[tuple([slice(None, -1) if i == axis else slice(None)
                                       for i in range(2)])]
        a = (hi - lo) / k
    return a


def _unsharp(l01, radius=2, amount=0.6):
    """Local-contrast boost on the luminance backbone."""
    blurred = _box_blur(l01.copy(), r=radius)
    return np.clip(l01 + amount * (l01 - blurred), 0.0, 1.0)


def fuse_reconstruction(sr_bt, rgb_display, entropy, K=6, confidence=None,
                        chroma_target=68.0, mid_grey=0.46, chroma_denoise=2):
    """Build the fused reconstruction.

    Args:
        sr_bt:        (H, W) super-resolved brightness temperature, Kelvin.
        rgb_display:  (H, W, 3) synthesized colour on the display 0-255 scale.
        entropy:      (H, W) mixture mixing-entropy (0 .. ln K).
        K:            number of mixture components (for entropy normalization).
        confidence:   optional (H, W) dominant-component weight in [0, 1].
        chroma_target: adaptive-saturation target for the 95th-pct chroma
                       magnitude (display units); higher = more vivid.
        mid_grey:     auto-tone target — the scene's median luminance is gamma-
                      mapped to this value (order-preserving), so mostly-warm or
                      mostly-cool scenes both land in a readable tonal range.

    Returns:
        (H, W, 3) uint8 fused reconstruction.
    """
    rgb = np.asarray(rgb_display, dtype=np.float32)

    # --- 1. Luminance backbone: sharpened, auto-toned SR thermal ----------
    l01 = _stretch01(np.asarray(sr_bt, dtype=np.float32))
    l01 = _unsharp(l01, radius=2, amount=0.6)
    med = float(np.clip(np.median(l01), 0.02, 0.98))
    auto_gamma = float(np.clip(np.log(mid_grey) / np.log(med), 0.4, 2.5))
    l01 = l01 ** auto_gamma

    # --- 2. Scene-relative trust map ---------------------------------------
    # Absolute honesty lives in the uncertainty deliverable; the fused image is
    # a *best estimate*, so trust is rescaled within the scene: the most
    # confident regions get full colour, the most ambiguous fade to thermal.
    ent01 = np.clip(np.asarray(entropy, dtype=np.float32) / max(np.log(K), 1e-6), 0.0, 1.0)
    trust = 1.0 - ent01
    if confidence is not None:
        trust = 0.5 * trust + 0.5 * np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
    trust = _stretch01(trust, 5.0, 95.0) ** 0.8

    # --- 3. Chroma from the synthesized colour, denoised + adaptive sat ----
    chroma = rgb - rgb.mean(axis=2, keepdims=True)          # zero-luminance colour offset
    # Colour detail carries most of the model's per-pixel noise while the human
    # eye tolerates soft chroma; blur the colour offset (luminance stays sharp)
    # so the reconstruction reads clean instead of speckled.
    if chroma_denoise > 0:
        for c in range(3):
            chroma[..., c] = _box_blur(chroma[..., c], r=chroma_denoise)
    mag95 = float(np.percentile(np.abs(chroma), 95))
    gain = float(np.clip(chroma_target / max(mag95, 1e-3), 1.0, 6.0))

    # --- 4. Compose: thermal luminance + trust-weighted chroma ------------
    fused = l01[..., None] * 255.0 + (gain * trust[..., None]) * chroma
    return np.clip(fused, 0.0, 255.0).astype(np.uint8)
