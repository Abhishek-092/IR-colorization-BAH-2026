# 🛰️ Project SUTRAM — Run Anywhere Guide

Everything needed is inside this zip: **code + trained model weights (v3.0.0) + demo scene**.
Works on **Windows / macOS / Linux**, with or without a GPU (auto-detects CUDA → Apple-MPS → CPU).

## 1. Requirements
- Python **3.9 – 3.12** (`python --version`)
- ~2 GB free disk, ~4 GB RAM

## 2. Install (one time)

```bash
# from the unzipped folder:
python -m venv .venv

# activate:
#   macOS / Linux:
source .venv/bin/activate
#   Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## 3. Launch the dashboard

```bash
python webapp/server.py
```
Then open **http://127.0.0.1:8000** in a browser.

Or use the one-click launchers: `run_dashboard.sh` (macOS/Linux) · `run_dashboard.bat` (Windows).

## 4. Using it
- **Demo scene** — click the demo tile chip: the full pipeline runs and a
  *Reconstruction vs Ground Truth* slider appears.
- **Your own data** — upload any Landsat Band-10 thermal raster
  (`*_ST_B10.TIF` GeoTIFF, or an 8-bit browse image). Calibration to
  brightness temperature is automatic.
- **Export** — panel 09 downloads every output (SR thermal, RGB, material
  map, reconstruction).

## 5. Command-line (optional)

```bash
python cli.py infer --weights checkpoints/sutram_final.pth --input <landsat_product_dir>
python cli.py benchmark
python -m pytest tests/          # 7 tests should pass
```

## 6. What's inside

| Path | Contents |
|---|---|
| `checkpoints/sutram_final.pth` | v3.0.0 weights — backbone + multi-scale SR head + K=6 colour mixture head (trained on 57 Landsat scenes, Kaggle P100) |
| `webapp/` | Dashboard (Flask) + bundled demo scene + ground-truth slider |
| `training/`, `data_pipeline/`, `configs/` | Full training stack (57-scene splits in `configs/data.yaml`) |
| `inference/`, `evaluation/`, `sutram/` | Fused reconstruction, GeoTIFF export (BGR submission order), metrics, Planck calibration |
| `kaggle/` | Cloud (Kaggle GPU) training kernel + instructions |

## Troubleshooting
- **`imagecodecs` wheel fails** (rare, old pip): `pip install --upgrade pip` first.
- **Port 8000 busy**: `PORT=8080 python webapp/server.py`.
- **No GPU**: fine — it runs on CPU (inference a few seconds per tile).
