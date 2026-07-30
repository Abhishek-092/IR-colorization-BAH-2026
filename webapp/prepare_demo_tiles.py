"""
prepare_demo_tiles.py
=====================
Carve 256x256 thermal demo tiles for the dashboard from the bundled Landsat
products. Two kinds of source are used, and the dashboard's auto-calibration
(sutram.calibration.autocalibrate) handles each correctly at inference time:

  * REAL browse scenes (ST_B10, uint8): genuine Earth imagery — rivers,
    forests, clouds — display-calibrated to brightness temperature.
  * LC09_146044 (B10, uint16 DN): the physical-calibration demo product —
    tiles run through the true Planck inversion path.

Run once:

    ./.venv/bin/python webapp/prepare_demo_tiles.py
"""
import os, json, glob
import numpy as np
import tifffile
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_tiles")
os.makedirs(OUT, exist_ok=True)

# (product glob, tiles to take, short label) — real ground scenes first.
SOURCES = [
    ("LC09_L2SP_136042_*", 2, "136042"),   # river + forest (real)
    ("LC09_L2SP_146040_*", 1, "146040"),   # real (val scene)
    ("LC09_L2SP_147046_*", 1, "147046"),   # real (val scene)
    ("LC08_L2SP_143051_*", 1, "143051"),   # real
    ("LC09_L2SP_146044_*", 1, "146044"),   # uint16 -> true Planck physics path
]
NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]

def find_b10(pattern):
    hits = (glob.glob(os.path.join(ROOT, "input", pattern, "*_ST_B10.TIF"))
            or glob.glob(os.path.join(ROOT, "input", pattern, "*_B10.TIF")))
    return hits[0] if hits else None

manifest, n = [], 0
for pattern, want, label in SOURCES:
    src = find_b10(pattern)
    if src is None:
        continue
    prod = os.path.basename(os.path.dirname(src))
    img = tifffile.imread(src).astype(np.float32)
    if img.ndim == 3:                             # RGBA/paletted browse TIFs
        img = img[..., 0] if img.shape[-1] <= 4 else img[0]
    H, W = img.shape
    is_browse = img.max() <= 255
    min_contrast = 15 if is_browse else 100      # scale-appropriate flatness gate
    crop = min(900, H // 3, W // 3)
    got = 0
    for fy, fx in [(0.40, 0.40), (0.55, 0.55), (0.35, 0.60), (0.60, 0.35), (0.48, 0.48)]:
        if got >= want or n >= len(NAMES):
            break
        y, x = int(fy * (H - crop)), int(fx * (W - crop))
        tile = img[y:y + crop, x:x + crop]
        if (tile > 0).mean() < 0.98:              # skip zero-filled scene corners
            continue
        tile256 = np.array(Image.fromarray(tile, mode="F")
                           .resize((256, 256), Image.BILINEAR), dtype=np.float32)
        if tile256.max() - tile256.min() < min_contrast:
            continue
        tid = f"SUTRAM-{label}-{NAMES[n]}"
        np.save(os.path.join(OUT, tid + ".npy"), tile256)
        manifest.append({
            "id": tid, "tile": NAMES[n], "product_id": prod,
            "dn_min": round(float(tile256.min()), 1),
            "dn_max": round(float(tile256.max()), 1),
        })
        print(f"  {tid}  ({'browse' if is_browse else 'uint16 physical'})  "
              f"DN {tile256.min():.0f}..{tile256.max():.0f}")
        got += 1
        n += 1

# Remove stale tiles from previous runs, keep only the fresh manifest set.
keep = {m["id"] + ".npy" for m in manifest} | {"manifest.json"}
for f in os.listdir(OUT):
    if f not in keep:
        os.remove(os.path.join(OUT, f))

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump({"tiles": manifest}, f, indent=2)
print(f"wrote {len(manifest)} demo tiles -> {OUT}")
