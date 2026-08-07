# SUTRAM Dashboard (real-model web app)

A mission-control web dashboard for **Project SUTRAM** that runs the **real
trained model** (`checkpoints/sutram_final.pth`) end to end: thermal
super-resolution, probabilistic RGB synthesis, uncertainty estimation and a
derived land-cover map — with an animated physics pipeline, an interactive
comparison slider, a pixel inspector, live metrics and a clickable Pipeline
Explorer.

This is **not** a mock. Every image and number comes from an actual forward
pass of the backbone + SR head + mixture head, served by a small Flask backend.

## Run it

From the repository root (the folder that contains `checkpoints/`):

```bash
./.venv/bin/python webapp/server.py
# then open http://127.0.0.1:8000
```

If you haven't already, the app needs Flask and imagecodecs in the venv:

```bash
./.venv/bin/python -m pip install flask imagecodecs
```

The six demo scenes are real 256×256 thermal tiles carved from the one bundled
Landsat-9 Band-10 product (`input/LC09_L2SP_146044_...`). Regenerate them with:

```bash
./.venv/bin/python webapp/prepare_demo_tiles.py
```

## How to use

1. **Pick a demo tile** (chips under "Upload Thermal Image") **or** drag in your
   own single-band thermal GeoTIFF/TIFF/PNG. This runs a real inference and
   shows the Original Thermal (hover it for per-pixel temperature).
2. Press **RUN SUTRAM** — the pipeline animates stage-by-stage, then the three
   outputs, comparison slider, inspector, metrics and explorer populate.
3. **Click the material map** in the Pixel Inspector for per-pixel temperature,
   RGB, predicted class, mixture confidence and mixing entropy.
4. **Export** any output as PNG or download an HTML report.

## Architecture

```
Browser (webapp/index.html)
    │  fetch /api/health, /api/scenes, POST /api/infer
    ▼
Flask (webapp/server.py)
    │  loads sutram_final.pth once (backbone + sr_head + mixture_head)
    ▼
Real forward pass  →  SR thermal, mixture RGB, entropy, uncertainty
    │  + Planck brightness temperature (sutram/calibration/planck.py)
    ▼
JSON: rendered PNGs (data URIs) + per-pixel grids (base64 typed arrays)
      + real checkpoint metrics + live latency
```

## Endpoints

| Method | Path          | Purpose                                            |
|--------|---------------|----------------------------------------------------|
| GET    | `/`           | the dashboard                                      |
| GET    | `/api/health` | model status, version, params, device, K           |
| GET    | `/api/scenes` | the real demo thermal tiles                         |
| POST   | `/api/infer`  | run the model on an uploaded file or a `scene` id   |

## Notes on the numbers

- **Temperatures** are real brightness temperatures via Planck's law on the L1
  thermal DN (≈300–307 K for these scenes).
- **Confidence / mixing entropy** are the real mixture-density outputs. This
  colorization model is genuinely uncertain (entropy near the K=6 maximum), so
  per-pixel confidence is modest — that is the model's true behaviour, not a bug.
- **PSNR / SSIM / BT-RMSE / ECE / Sparsification** are read from the
  checkpoint's stored validation metrics; **inference time** is measured live.
  The stored SSIM (0.007) looks off — likely a scale mismatch in the offline
  evaluation script — and is worth re-checking in `evaluation/metrics.py`.
- The **Final Predicted Output** is a transparent land-cover map *derived* from
  the model's synthesized RGB + brightness temperature (labelled as such in the
  UI); the model itself does not have a segmentation head.
```
