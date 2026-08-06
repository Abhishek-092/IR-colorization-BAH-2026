"""
SUTRAM Dashboard — backend server
=================================
Serves the SUTRAM mission-control dashboard and runs REAL inference with the
trained model (checkpoints/sutram_final.pth). No simulation: every image and
number the dashboard shows comes from an actual forward pass of the loaded
backbone + SR head + mixture head.

Run from the repository root:

    ./.venv/bin/python webapp/server.py
    # then open http://127.0.0.1:8000

Endpoints
---------
GET  /                served dashboard (webapp/index.html + static files)
GET  /api/health      model status, version, params, device, K
GET  /api/scenes      list of bundled Landsat demo products
POST /api/infer       run the model on an uploaded file or a named demo scene;
                      returns rendered PNGs (data URIs) + per-pixel data grids
                      (base64 typed arrays) + real metrics + timing.
"""

import os
import sys
import io
import time
import glob
import json
import base64

# Make the repo importable no matter where the process is started from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, send_file

from training.backbone import ResNetBackbone
from training.sr_head import SRHead
from training.mixture_head import MixtureHead
from inference.fusion import fuse_reconstruction
from sutram.calibration.planck import TB_MIN, TB_MAX
from sutram.calibration.autocalibrate import (
    calibrate_thermal_to_bt, calibrate_thermal_to_norm, thermal_is_physical,
)

WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(ROOT, "checkpoints", "sutram_final.pth")
INPUT_DIR = os.path.join(ROOT, "input")
TILES_DIR = os.path.join(WEBAPP_DIR, "demo_tiles")

# Derived land-cover classes (the model itself outputs colour, not a label map;
# we derive a material map from the model's synthesized RGB + brightness
# temperature — see derive_landcover()). Colours are segmentation-style.
CLASS_NAMES = ["Water", "Vegetation", "Bare Soil", "Urban", "Road"]
CLASS_COLORS = np.array([
    [47, 111, 237],    # Water        blue
    [63, 163, 77],     # Vegetation   green
    [216, 198, 156],   # Bare Soil    tan
    [179, 84, 30],     # Urban        brown/terracotta
    [139, 149, 164],   # Road         grey
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Model: load once at startup and keep resident.
# ---------------------------------------------------------------------------
class SutramModel:
    def __init__(self, ckpt_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu"))
        ck = torch.load(ckpt_path, map_location="cpu")
        self.config = ck.get("config", {})
        self.stored_metrics = ck.get("metrics", {})
        self.version = self.config.get("version", ck.get("version", "1.0.0"))
        self.K = self.config.get("K_components", 6)

        self.backbone = ResNetBackbone()
        self.sr_head = SRHead()
        self.mixture_head = MixtureHead(K=self.K)
        self.backbone.load_state_dict(ck["backbone_state_dict"])
        self.sr_head.load_state_dict(ck["sr_head_state_dict"])
        self.mixture_head.load_state_dict(ck["mixture_head_state_dict"])
        for m in (self.backbone, self.sr_head, self.mixture_head):
            m.eval().to(self.device)

        self.n_params = sum(
            p.numel() for m in (self.backbone, self.sr_head, self.mixture_head)
            for p in m.parameters())

        # Warm the graph so the first real request isn't billed for lazy
        # device/kernel initialisation (matters a lot on MPS/CUDA).
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 256, 256, device=self.device)
            for _ in range(3):
                fdum = self.backbone(dummy)
                sdum = self.sr_head(fdum, dummy)
                self.mixture_head(fdum, sdum)
        self._sync()

    def _sync(self):
        """Block until queued device work finishes, for honest timing."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @staticmethod
    def _denorm_sr(sr):
        # Super-resolved output is denormalized straight back to Kelvin.
        return torch.clamp(sr * (TB_MAX - TB_MIN) + TB_MIN, TB_MIN, TB_MAX)

    @torch.no_grad()
    def infer(self, lr_norm):
        """Run the real forward pass. `lr_norm` is a float32 tensor (1,1,H,W) of
        already-calibrated, normalized brightness temperature in [0, 1]
        (calibration is done host-side via sutram.calibration.autocalibrate, so
        this matches the training input and the inference pipeline exactly).
        Returns a dict of numpy arrays with full uncertainty decomposition."""
        lr_norm = lr_norm.to(self.device)

        self._sync()
        t0 = time.perf_counter()
        feats = self.backbone(lr_norm)
        sr_tir = self.sr_head(feats, lr_norm)
        logit_w, means, log_scales = self.mixture_head(feats, sr_tir)

        # --- mixture decode (mirrors DecodeSubmoduleFP32, plus pi_max) --------
        logit_w = logit_w.float(); means = means.float(); log_scales = log_scales.float()
        pi = F.softmax(logit_w, dim=1)                       # (B,K,H,W)
        scales = F.softplus(log_scales) + 1.0                # (B,K,3,H,W)

        top_w, top_i = torch.topk(pi, k=min(2, self.K), dim=1)
        pi_max = top_w[:, 0]                                  # (B,H,W) confidence
        k_star = top_i[:, 0:1]
        dom = torch.gather(means, 1, k_star.unsqueeze(2).expand(-1, -1, 3, -1, -1)).squeeze(1)

        log_pi = torch.log(torch.clamp(pi, min=1e-7))
        entropy = -(pi * log_pi).sum(dim=1)                  # (B,H,W)
        pi_u = pi.unsqueeze(2)
        within = (pi_u * scales**2 * (np.pi**2 / 3.0)).sum(1).mean(1)
        bar_mu = (pi_u * means).sum(1, keepdim=True)
        between = (pi_u * (means - bar_mu)**2).sum(1).mean(1)
        self._sync()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        sr_bt = self._denorm_sr(sr_tir)                      # (1,1,512,512) Kelvin

        return {
            "sr_bt": sr_bt.squeeze().cpu().numpy(),          # (512,512) brightness temp (K)
            "rgb_raw": dom.squeeze().cpu().numpy(),          # (3,512,512)
            "confidence": pi_max.squeeze().cpu().numpy(),    # (512,512) 0..1
            "entropy": entropy.squeeze().cpu().numpy(),      # (512,512)
            "within_var": within.squeeze().cpu().numpy(),
            "between_var": between.squeeze().cpu().numpy(),
            "time_ms": dt_ms,
        }


MODEL = SutramModel(CKPT_PATH)
MAX_ENTROPY = float(np.log(MODEL.K))
print(f"[sutram] model loaded: {MODEL.n_params:,} params, K={MODEL.K}, "
      f"device={MODEL.device}, v{MODEL.version}")


# ---------------------------------------------------------------------------
# Image / data helpers
# ---------------------------------------------------------------------------
def greyscale_thermal(arr):
    """Non colour-coded super-resolution: percentile-stretch the thermal field to
    a plain 0-255 greyscale (no LUT). This is the raw super-resolved structure —
    hot = bright, cold = dark — the honest scientific view without false colour."""
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)  # 2-D -> L PNG


def normalize_rgb(rgb_raw):
    """Render the model's dominant RGB means as a viewable uint8 image.

    The mixture means are trained directly against 0-255 RGB targets, so they
    are already on a physical colour scale. We display them on that scale with
    only a light bright-tail clip. A per-channel min/max (or percentile-2-98)
    stretch is deliberately NOT used: when the colorization output is
    near-uniform (its common failure mode on low-variety data), the bulk sits at
    the low percentile and such a stretch maps almost every pixel to black. This
    keeps the true colour visible and just tames sparse over-bright pixels."""
    rgb = np.transpose(rgb_raw, (1, 2, 0)).astype(np.float32)  # H,W,3
    # Tame only the sparse over-bright tail (per channel), then clip to 0-255.
    for c in range(3):
        hi = np.percentile(rgb[..., c], 99.5)
        if hi > 255.0:
            rgb[..., c] = np.minimum(rgb[..., c], hi)
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def display_stretch(rgb_u8):
    """Hue-preserving display stretch for VIEWING only. One global affine gain
    (from the 2-98 luminance percentiles) applied identically to all three
    channels: dark-but-correct terrain becomes readable without the per-channel
    stretch failure mode that maps near-uniform output to black, and without
    changing the colour ratios the model predicted. Exports/grids stay physical."""
    rgb = rgb_u8.astype(np.float32)
    lum = rgb.mean(axis=2)
    lo, hi = np.percentile(lum, 0.5), np.percentile(lum, 99.5)
    if hi - lo < 5.0:
        # Near-uniform image: modest fixed gain toward mid-grey, never black.
        gain = 128.0 / max(lum.mean(), 1.0)
        return np.clip(rgb * min(gain, 4.0), 0, 255).astype(np.uint8)
    return np.clip((rgb - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def derive_landcover(rgb_view, temp_k):
    """Derive a 5-class material map from the model's synthesized RGB (display-
    stretched view) and the brightness temperature. This is a transparent
    downstream heuristic on the REAL model output — not a separate trained
    classifier — so it's labelled as 'derived' in the UI. All thresholds are
    percentile-relative, so the map adapts to each scene's brightness range
    instead of assuming one global scale.
    Returns (label_idx HxW uint8, seg_rgb HxW3 uint8)."""
    r = rgb_view[..., 0].astype(np.float32)
    g = rgb_view[..., 1].astype(np.float32)
    b = rgb_view[..., 2].astype(np.float32)
    bright = (r + g + b) / 3.0
    b30, b60, b85 = np.percentile(bright, [30, 60, 85])
    tk = temp_k
    t_lo, t_hi = np.percentile(tk, 5), np.percentile(tk, 95)
    t_norm = np.clip((tk - t_lo) / max(t_hi - t_lo, 1e-6), 0, 1)

    H, W = r.shape
    label = np.full((H, W), 2, dtype=np.uint8)   # default: Bare Soil
    # Vegetation: green channel leads and not hot.
    veg = (g > r + 4) & (g > b + 2) & (t_norm < 0.75)
    # Urban: warm and bright for the scene, without a green lead.
    urban = (t_norm > 0.60) & (bright > b60) & (g <= r + 4)
    # Road: warm, dark for the scene, low saturation.
    road = (t_norm > 0.55) & (bright < b30) & (np.abs(r - g) < 22) & (np.abs(g - b) < 22)
    # Water: cool and dark / blue-leaning (clouds are cool but BRIGHT — excluded).
    water = (t_norm < 0.30) & (bright < b60) & (b >= g - 8)
    # Cloud/snow: very bright for the scene and cool -> shown as bare-soil-neutral?
    # No — keep the 5-class taxonomy; very bright & cool pixels default to class 2.

    label[urban] = 3
    label[veg] = 1
    label[road] = 4
    label[water] = 0
    seg = CLASS_COLORS[label]
    return label, seg


def to_png_datauri(arr_u8):
    """Encode an (H,W,3) or (H,W) uint8 array as a data: URI PNG."""
    if arr_u8.ndim == 2:
        img = Image.fromarray(arr_u8, mode="L")
    else:
        img = Image.fromarray(arr_u8, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def b64_array(arr):
    """Base64 of a raw little-endian array (for compact transfer to JS)."""
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


# ---------------------------------------------------------------------------
# Input reading
# ---------------------------------------------------------------------------
def read_thermal_any(path_or_bytes, is_bytes=False):
    """Read a thermal raster from a GeoTIFF/TIFF/PNG (path or bytes) into a
    float32 2-D array of DN values. Multi-band inputs use the first band."""
    if is_bytes:
        bio = io.BytesIO(path_or_bytes)
        try:
            import tifffile
            arr = tifffile.imread(bio)
        except Exception:
            bio.seek(0)
            arr = np.array(Image.open(bio))
    else:
        try:
            import tifffile
            arr = tifffile.imread(path_or_bytes)
        except Exception:
            arr = np.array(Image.open(path_or_bytes))
    arr = np.asarray(arr).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[2] <= 4 else arr[0]
    return arr


def resize_to(arr, size=256):
    """Resize a 2-D float array to size×size using PIL (bilinear)."""
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


def load_demo_manifest():
    """Load the real-thermal demo tiles produced by prepare_demo_tiles.py."""
    mpath = os.path.join(TILES_DIR, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            return json.load(f)
    return {"tiles": []}


def demo_scene_meta(tile):
    """Scene-info card fields for a demo tile (real product 146044)."""
    meta = parse_product_meta(tile["product_id"])
    meta.update({
        "id": tile["id"],
        "tile": tile.get("tile", ""),
        "display_name": tile.get("tile", ""),
    })
    return meta



def load_scene_array(scene_id):
    """Return the real thermal DN array for a demo scene id, or None."""
    p = os.path.join(TILES_DIR, scene_id + ".npy")
    if os.path.exists(p):
        return np.load(p).astype(np.float32)
    return None


def parse_product_meta(prod_id):
    """Parse a Landsat product id into scene metadata for the info card."""
    # e.g. LC09_L2SP_146040_20260712_20260713_02_T1
    parts = prod_id.split("_")
    sat = "Landsat-9" if prod_id.startswith("LC09") else \
          ("Landsat-8" if prod_id.startswith("LC08") else "Landsat")
    path = row = date = "—"
    if len(parts) >= 4:
        pr = parts[2]
        if len(pr) == 6:
            path, row = pr[:3], pr[3:]
        d = parts[3]
        if len(d) == 8:
            date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return {
        "product_id": prod_id,
        "satellite": sat,
        "sensor": "TIRS Band 10",
        "wrs_path": path,
        "wrs_row": row,
        "acquisition": date,
        "resolution_in": "200 m",
        "resolution_out": "100 m",
        "image_size": "256 × 256",
        "processing": "Level-2 (L2SP)",
    }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(WEBAPP_DIR, "index.html")


@app.get("/static/<path:fname>")
def static_files(fname):
    return send_from_directory(os.path.join(WEBAPP_DIR, "static"), fname)


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ready",
        "model_name": MODEL.config.get("model_name", "Project SUTRAM"),
        "version": MODEL.version,
        "parameters": MODEL.n_params,
        "parameters_m": round(MODEL.n_params / 1e6, 2),
        "K": MODEL.K,
        "precision": MODEL.config.get("precision", "fp32"),
        "device": str(MODEL.device).upper(),
        "encoder": "ResNet",
    })


@app.get("/api/scenes")
def scenes():
    manifest = load_demo_manifest()
    out = [demo_scene_meta(t) for t in manifest.get("tiles", [])]
    return jsonify({"scenes": out})


def run_inference_payload(lr_np, meta):
    """Shared inference + rendering for both upload and demo-scene requests."""
    lr256 = resize_to(lr_np, 256) if lr_np.shape != (256, 256) else lr_np

    # Auto-calibrate the raw raster to brightness temperature (real physics for
    # L1/L2 inputs; display-calibrated for 8-bit browse), then normalize for the
    # network — the exact same path training uses.
    bt_in = calibrate_thermal_to_bt(lr256)                   # 256×256 K
    physical = thermal_is_physical(lr256)
    lr_t = torch.from_numpy(calibrate_thermal_to_norm(lr256)).unsqueeze(0).unsqueeze(0).float()

    res = MODEL.infer(lr_t)
    bt_sr = res["sr_bt"]                                     # 512×512 K (already Kelvin)

    rgb_u8 = normalize_rgb(res["rgb_raw"])                   # 512×512×3 physical display
    rgb_view = display_stretch(rgb_u8)                       # viewing-only stretch (hue-preserving)
    label, seg = derive_landcover(rgb_view, bt_sr)          # 512×512

    # Fused ground reconstruction: SR thermal detail + synthesized colour +
    # uncertainty-driven abstention, all in one best-estimate image.
    fused = fuse_reconstruction(bt_sr, rgb_view, res["entropy"],
                                K=MODEL.K, confidence=res["confidence"])

    # --- rendered images (data URIs) — all non colour-coded (greyscale) ---
    thermal_in_img = greyscale_thermal(lr256)               # coarse input
    sr_img = greyscale_thermal(res["sr_bt"])                # super-resolved output
    ent = res["entropy"]
    ent_norm = np.clip((ent - ent.min()) / max(ent.max() - ent.min(), 1e-6), 0, 1)
    unc_img = (ent_norm * 255).astype(np.uint8)             # uncertainty: bright = less certain

    # --- per-pixel data grids for the inspector, at 256×256 to keep JSON small ---
    def to256(a):
        return np.array(Image.fromarray(a.astype(np.float32), "F")
                        .resize((256, 256), Image.NEAREST), dtype=np.float32)
    temp_grid = to256(bt_sr)                                 # Kelvin
    conf_grid = to256(res["confidence"])                    # 0..1
    ent_grid = to256(ent)
    rgb_grid = np.stack([to256(rgb_u8[..., c]) for c in range(3)], -1).astype(np.uint8)
    label_grid = np.array(Image.fromarray(label, "L")
                          .resize((256, 256), Image.NEAREST), dtype=np.uint8)

    # --- prediction summary (distribution of derived classes) ---
    counts = np.bincount(label_grid.ravel(), minlength=len(CLASS_NAMES))
    total = int(counts.sum())
    dist = [{"label": CLASS_NAMES[i], "pct": round(100.0 * counts[i] / total, 1)}
            for i in np.argsort(counts)[::-1]]
    dominant = dist[0]["label"]
    dom_conf = round(float(conf_grid[label_grid == np.argmax(counts)].mean()) * 100, 1) \
        if total else 0.0
    mean_entropy = round(float(ent_grid.mean()), 3)

    return {
        "scene": meta,
        "images": {
            "thermal_input": to_png_datauri(thermal_in_img),
            "super_resolution": to_png_datauri(sr_img),
            "synth_rgb": to_png_datauri(rgb_view),
            "final_reconstruction": to_png_datauri(fused),
            "land_cover_map": to_png_datauri(seg),
            "uncertainty": to_png_datauri(unc_img),
        },
        "grids": {
            "size": 256,
            "temperature": b64_array(temp_grid.astype(np.float32)),
            "confidence": b64_array(conf_grid.astype(np.float32)),
            "entropy": b64_array(ent_grid.astype(np.float32)),
            "rgb": b64_array(rgb_grid.astype(np.uint8)),
            "label": b64_array(label_grid.astype(np.uint8)),
        },
        "classes": [
            {"name": n, "color": CLASS_COLORS[i].tolist()}
            for i, n in enumerate(CLASS_NAMES)
        ],
        "summary": {
            "dominant": dominant,
            "confidence": dom_conf,
            "mean_entropy": mean_entropy,
            "distribution": dist,
        },
        "metrics": {
            "psnr": round(float(MODEL.stored_metrics.get("psnr", 0)), 2),
            "ssim": round(float(MODEL.stored_metrics.get("ssim", 0)), 3),
            "bt_rmse": round(float(MODEL.stored_metrics.get("bt_rmse", 0)), 3),
            "ece": round(float(MODEL.stored_metrics.get("ece", 0)), 3),
            "sparsification_auc": round(float(MODEL.stored_metrics.get("sparsification_auc", 0)), 3),
            "inference_ms": round(float(res["time_ms"]), 1),
        },
        "model": {
            "name": MODEL.config.get("model_name", "Project SUTRAM"),
            "version": MODEL.version,
            "encoder": "ResNet",
            "mixture_components": MODEL.K,
            "precision": MODEL.config.get("precision", "fp32"),
            "parameters": MODEL.n_params,
            "parameters_m": round(MODEL.n_params / 1e6, 2),
            "device": str(MODEL.device).upper(),
            "scale_mode": "brightness_temp",
        },
        "temp_stats": {
            "min": round(float(np.nanmin(bt_in)), 1),
            "max": round(float(np.nanmax(bt_in)), 1),
            "mean": round(float(np.nanmean(bt_in)), 1),
            "physical": bool(physical),  # True = real radiometry; False = display-calibrated browse
        },
    }


@app.post("/api/infer")
def infer():
    try:
        if "file" in request.files and request.files["file"].filename:
            f = request.files["file"]
            file_bytes = f.read()
            # Save it temporarily as reference
            temp_path = os.path.join(ROOT, "output", "temp_upload.tif")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as temp_file:
                temp_file.write(file_bytes)
            lr_np = read_thermal_any(file_bytes, is_bytes=True)
            meta = parse_product_meta(os.path.splitext(f.filename)[0])
            meta["satellite"] = meta.get("satellite", "User Upload")
            if meta["wrs_path"] == "—":
                meta.update({"satellite": "User Upload", "sensor": "Thermal (uploaded)",
                             "acquisition": "—", "processing": "—"})
        else:
            body = request.get_json(silent=True) or {}
            scene_id = body.get("scene")
            if not scene_id:
                return jsonify({"error": "Provide a 'file' upload or a 'scene' id."}), 400
            lr_np = load_scene_array(scene_id)
            if lr_np is None:
                return jsonify({"error": f"Scene '{scene_id}' not found."}), 404
            tile = next((t for t in load_demo_manifest().get("tiles", [])
                         if t["id"] == scene_id), {"product_id": scene_id})
            meta = demo_scene_meta(tile)

        payload = run_inference_payload(lr_np, meta)
        return jsonify(payload)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


def find_scene_b10(product_id):
    p = os.path.join(INPUT_DIR, product_id)
    hits = (glob.glob(os.path.join(p, "*_B10.TIF")) +
            glob.glob(os.path.join(p, "*_B10.tif")) +
            glob.glob(os.path.join(p, "*_ST_B10.TIF")) +
            glob.glob(os.path.join(p, "*_ST_B10.tif")))
    if hits:
        return hits[0]
    any_hits = (glob.glob(os.path.join(INPUT_DIR, "*", "*_B10.TIF")) +
                glob.glob(os.path.join(INPUT_DIR, "*", "*_B10.tif")) +
                glob.glob(os.path.join(INPUT_DIR, "*", "*_ST_B10.TIF")) +
                glob.glob(os.path.join(INPUT_DIR, "*", "*_ST_B10.tif")))
    if any_hits:
        return any_hits[0]
    return None


@app.post("/api/download_geotiff")
def download_geotiff():
    from inference.geotiff_export import export_sr_geotiff, export_colorized_geotiff
    try:
        body = request.get_json(silent=True) or {}
        scene_id = body.get("scene_id")
        output_type = body.get("output_type")
        
        if not output_type:
            return jsonify({"error": "output_type is required"}), 400
            
        if not scene_id or scene_id == "null":
            ref_path = os.path.join(ROOT, "output", "temp_upload.tif")
            if not os.path.exists(ref_path):
                return jsonify({"error": "No uploaded reference file found."}), 404
            lr_np = read_thermal_any(ref_path)
        else:
            tile = next((t for t in load_demo_manifest().get("tiles", [])
                         if t["id"] == scene_id), None)
            if not tile:
                return jsonify({"error": f"Scene '{scene_id}' not found."}), 404
            ref_path = find_scene_b10(tile["product_id"])
            lr_np = load_scene_array(scene_id)
            
        if lr_np is None:
            return jsonify({"error": "Failed to load input thermal array."}), 404
            
        if not ref_path or not os.path.exists(ref_path):
            return jsonify({"error": "No valid reference GeoTIFF file available on server."}), 404

        lr256 = resize_to(lr_np, 256) if lr_np.shape != (256, 256) else lr_np
        lr_t = torch.from_numpy(calibrate_thermal_to_norm(lr256)).unsqueeze(0).unsqueeze(0).float()
        res = MODEL.infer(lr_t)
        
        os.makedirs(os.path.join(ROOT, "output", "temp_exports"), exist_ok=True)
        temp_out_path = os.path.join(ROOT, "output", "temp_exports", f"export_{output_type}.tif")
        
        if output_type == "super_resolution":
            bt_sr = res["sr_bt"]
            # Stretch [TB_MIN, TB_MAX] range to [0, 255] uint8 for correct display in photo viewers
            bt_sr_norm = np.clip((bt_sr - TB_MIN) / (TB_MAX - TB_MIN), 0.0, 1.0)
            bt_sr_u8 = (bt_sr_norm * 255.0).astype(np.uint8)
            export_sr_geotiff(bt_sr_u8, ref_path, temp_out_path)
        elif output_type == "final_reconstruction":
            bt_sr = res["sr_bt"]
            rgb_u8 = normalize_rgb(res["rgb_raw"])
            rgb_view = display_stretch(rgb_u8)
            fused = fuse_reconstruction(bt_sr, rgb_view, res["entropy"],
                                        K=MODEL.K, confidence=res["confidence"])
            export_colorized_geotiff(np.transpose(fused, (2, 0, 1)).astype(np.uint8), ref_path, temp_out_path)
        elif output_type == "synth_rgb":
            rgb_u8 = normalize_rgb(res["rgb_raw"])
            rgb_view = display_stretch(rgb_u8)
            export_colorized_geotiff(np.transpose(rgb_view, (2, 0, 1)).astype(np.uint8), ref_path, temp_out_path)
        elif output_type == "land_cover_map":
            rgb_u8 = normalize_rgb(res["rgb_raw"])
            rgb_view = display_stretch(rgb_u8)
            bt_sr = res["sr_bt"]
            label, seg = derive_landcover(rgb_view, bt_sr)
            export_colorized_geotiff(np.transpose(seg, (2, 0, 1)).astype(np.uint8), ref_path, temp_out_path)
        elif output_type == "uncertainty":
            ent = res["entropy"]
            ent_norm = np.clip((ent - ent.min()) / max(ent.max() - ent.min(), 1e-6), 0, 1)
            unc_img = (ent_norm * 255).astype(np.uint8)
            export_sr_geotiff(unc_img, ref_path, temp_out_path)
        else:
            return jsonify({"error": f"Invalid output_type: {output_type}"}), 400
            
        return send_file(temp_out_path, as_attachment=True, download_name=f"{output_type}.tif")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Export failed: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"[sutram] serving dashboard at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
