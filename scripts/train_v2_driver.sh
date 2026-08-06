#!/bin/bash
# Unattended driver: Stage 1 (SR) -> Stage 2 (colour) -> package sutram_final.pth
# on the 37-scene DATA RELIC dataset (experiment sutram_v2). Emits clear markers.
set -o pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== DRIVER START $(date) ==="

echo ">>> STAGE1_BEGIN"
$PY -u cli.py train-stage1 --force 2>&1 || { echo "STAGE1_FAILED"; exit 1; }
echo ">>> STAGE1_DONE"

echo ">>> STAGE2_BEGIN"
$PY -u cli.py train-stage2 --force 2>&1 || { echo "STAGE2_FAILED"; exit 1; }
echo ">>> STAGE2_DONE"

echo ">>> EXPORT_BEGIN"
$PY -u - <<'PYEOF' || { echo "EXPORT_FAILED"; exit 1; }
import torch, json, os, datetime
src = "experiments/sutram_v2/checkpoints"
bb  = torch.load(f"{src}/backbone_stage1.pth", map_location="cpu")
sr  = torch.load(f"{src}/sr_head_stage1.pth", map_location="cpu")
mix = torch.load(f"{src}/mixture_head_stage2.pth", map_location="cpu")
metrics = {}
mp = "experiments/sutram_v2/metrics.json"
if os.path.exists(mp):
    metrics = json.load(open(mp))
cfg = {
    "model_name": "Project SUTRAM",
    "version": "2.0.0",
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "K_components": 6,
    "precision": "fp32",
    "dataset": "DATA RELIC 37-scene (32 train / 5 val)",
    "sr_head": "multi-scale RCAB (0.6M)",
}
final = {
    "model_name": "Project SUTRAM End-to-End Release",
    "version": "2.0.0",
    "backbone_state_dict": bb,
    "sr_head_state_dict": sr,
    "mixture_head_state_dict": mix,
    "metrics": metrics,
    "config": cfg,
}
os.makedirs("checkpoints", exist_ok=True)
torch.save(final, "checkpoints/sutram_final.pth")
n = sum(v.numel() for v in bb.values()) + sum(v.numel() for v in sr.values()) + sum(v.numel() for v in mix.values())
print(f"packaged checkpoints/sutram_final.pth  ({n:,} params)")
PYEOF
echo ">>> EXPORT_DONE"
echo "=== DRIVER ALL_DONE $(date) ==="
