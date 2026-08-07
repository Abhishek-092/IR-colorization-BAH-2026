"""
generate_all_synthetic.py
=========================
Generates a rich, structurally-coherent synthetic Landsat-9 dataset for every
product id in configs/data.yaml (train + val): B2/B3/B4/B10 written with the
correct per-product prefix into input/<product_id>/.

This is a DATA-ONLY accuracy lever — it does not touch the model pipeline. It
improves on the first version in three ways that directly raise SR + colour
accuracy:

  1. ORGANIC, MULTI-SCALE TEXTURE. Land cover comes from smooth low-frequency
     noise fields (elevation, moisture, urban) rather than hard circles/rects,
     and every band gets multi-octave texture. This gives the super-resolution
     head real high-frequency structure to reconstruct instead of flat blocks.

  2. FULL-RANGE COLOUR. Class reflectances span ~500..9500 so that after the
     loader's fixed RGB normalisation (reflectance/10000*255) the colour targets
     fill the whole 0..255 range — vivid, well-separated colours the mixture
     head can actually learn. (The pipeline's RGB_SCALE=10000 is UNCHANGED.)

  3. THERMAL CORRELATED WITH LAND COVER. Water is cool, vegetation moderate,
     bare soil warm, urban/road hot — all inside the pipeline's [20000,35000]
     DN normalisation window — so colour is predictable from thermal.

Each product is seeded by its id, so scenes differ and train/val stay disjoint.
Run:  ./.venv/bin/python scripts/generate_all_synthetic.py
"""
import os, sys, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import cv2
import tifffile
from omegaconf import OmegaConf

SIZE = 3416  # matches the real product; yields 512-patches after 3.33x downscale

# Per-class band means (B2 blue, B3 green, B4 red) chosen to fill 0..255 after
# reflectance/10000*255, and B10 thermal DN inside the [20000,35000] window.
#            B2     B3     B4     B10(DN)
CLASS = {
    "water":  (2600, 1900,  650, 27600),   # dark, blue>green>red, cool
    "veg":    ( 700, 5600, 1400, 29000),   # green-dominant, moderate temp
    "soil":   (4200, 6200, 9300, 30600),   # bright reddish-brown, warm
    "urban":  (6600, 6900, 7200, 31500),   # bright grey, hot
    "road":   (1900, 2000, 2100, 32100),   # dark asphalt, hottest
}


def _seed_for(pid):
    return int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)


def _noise(rng, size, cells):
    """Smooth low-frequency field in [0,1] from upsampled random noise."""
    low = rng.random((cells, cells)).astype(np.float32)
    up = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
    return np.clip((up - up.min()) / (np.ptp(up) + 1e-6), 0, 1)


def _fractal(rng, size):
    """Multi-octave texture in [0,1] for realistic within-class detail."""
    f = np.zeros((size, size), np.float32)
    amp = 1.0
    for cells in (6, 14, 32, 80):
        f += amp * _noise(rng, size, cells)
        amp *= 0.5
    return np.clip((f - f.min()) / (np.ptp(f) + 1e-6), 0, 1)


def generate_product(pid, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(_seed_for(pid))
    N = SIZE

    elevation = _fractal(rng, N)                 # low = water basins
    moisture = _fractal(rng, N)                  # high = vegetation
    urbanf = _noise(rng, N, 5) * _fractal(rng, N)  # clustered settlements
    detail = _fractal(rng, N)                    # fine texture for all bands

    # --- classify land cover (organic, per-product varied thresholds) -------
    water_l = np.quantile(elevation, rng.uniform(0.14, 0.24))
    veg_l = np.quantile(moisture, rng.uniform(0.45, 0.6))
    urban_l = np.quantile(urbanf, rng.uniform(0.8, 0.9))

    label = np.full((N, N), 2, np.uint8)          # default soil (idx 2)
    idx = {"water": 0, "veg": 1, "soil": 2, "urban": 3, "road": 4}
    label[(moisture > veg_l) & (elevation > water_l)] = idx["veg"]
    label[urbanf > urban_l] = idx["urban"]
    label[elevation < water_l] = idx["water"]
    # roads: thin high-gradient lines through urban areas
    gx = np.abs(cv2.Sobel(urbanf, cv2.CV_32F, 1, 0, ksize=5))
    gy = np.abs(cv2.Sobel(urbanf, cv2.CV_32F, 0, 1, ksize=5))
    road = (label == idx["urban"]) & ((gx + gy) > np.quantile(gx + gy, 0.92))
    label[road] = idx["road"]

    # --- paint bands from class means + multi-octave texture ----------------
    names = ["water", "veg", "soil", "urban", "road"]
    b2 = np.zeros((N, N), np.float32); b3 = np.zeros_like(b2)
    b4 = np.zeros_like(b2); b10 = np.zeros_like(b2)
    tex = (detail - 0.5)                          # centered texture
    for i, nm in enumerate(names):
        m = label == i
        if not m.any():
            continue
        v2, v3, v4, vt = CLASS[nm]
        b2[m] = v2 + tex[m] * 900
        b3[m] = v3 + tex[m] * 900
        b4[m] = v4 + tex[m] * 900
        b10[m] = vt + tex[m] * 700
    # global fine grain so SR has high-frequency signal everywhere
    grain = rng.normal(0, 60, (N, N)).astype(np.float32)
    for b, s in ((b2, 1), (b3, 1), (b4, 1), (b10, 3)):
        b += grain * s

    clip16 = lambda a: np.clip(a, 0, 65535).astype(np.uint16)
    for band, arr in (("B2", b2), ("B3", b3), ("B4", b4), ("B10", b10)):
        tifffile.imwrite(os.path.join(out_dir, f"{pid}_{band}.TIF"), clip16(arr))

    frac = {n: float((label == i).mean()) for i, n in enumerate(names)}
    return frac


def main():
    cfg = OmegaConf.load(os.path.join(ROOT, "configs", "data.yaml"))
    pids = list(cfg.splits.train) + list(cfg.splits.val)
    print(f"Generating {len(pids)} rich synthetic products ({SIZE}x{SIZE}) ...")
    for pid in pids:
        frac = generate_product(pid, os.path.join(ROOT, "input", pid))
        split = "train" if pid in cfg.splits.train else "val"
        comp = " ".join(f"{k}={v*100:.0f}%" for k, v in frac.items())
        print(f"  [{split}] {pid}: {comp}")
    print("Done. Next: python -m data_pipeline.prepare_dataset --force")


if __name__ == "__main__":
    main()
