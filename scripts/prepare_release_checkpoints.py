import os
import torch
import datetime
import json
import logging
from training.backbone import ResNetBackbone
from training.sr_head import SRHead
from training.mixture_head import MixtureHead

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _require_sutram_release_source(path):
    """Load a source checkpoint for packaging a SUTRAM release."""
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        # Stage checkpoints legitimately contain plain state_dicts; they are
        # allowed only as inputs to the packaging step.
        return checkpoint

    model_name = str(checkpoint.get("model_name", ""))
    project_id = checkpoint.get("project_id")
    if model_name and (project_id != "SUTRAM" or "SUTRAM" not in model_name.upper()):
        raise ValueError(f"Non-SUTRAM artifact detected at {path}. Retrain before packaging a release.")
    return checkpoint

def main():
    logger.info("Initializing checkpoint packaging process...")
    
    src_dir = "experiments/sutram_baseline/checkpoints"
    dest_dir = "checkpoints"
    os.makedirs(dest_dir, exist_ok=True)
    
    # Load original weights
    bb_path = os.path.join(src_dir, "backbone_stage1.pth")
    sr_path = os.path.join(src_dir, "sr_head_stage1.pth")
    mix_path = os.path.join(src_dir, "mixture_head_stage2.pth")
    
    if not (os.path.exists(bb_path) and os.path.exists(sr_path) and os.path.exists(mix_path)):
        logger.error("Source checkpoints missing in experiments folder. Run training first.")
        return
        
    bb_state = _require_sutram_release_source(bb_path)
    sr_state = _require_sutram_release_source(sr_path)
    mix_state = _require_sutram_release_source(mix_path)
    
    # A release is valid only when backed by an actual held-out evaluation.
    metrics_path = "experiments/sutram_baseline/metrics_val.json"
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            "Missing experiments/sutram_baseline/metrics_val.json. "
            "Run held-out evaluation before packaging a release."
        )
    with open(metrics_path, "r") as f:
        metrics_document = json.load(f)
    metrics = metrics_document.get("Overall", metrics_document)
    required_metrics = {"psnr", "ssim", "bt_rmse", "ece", "sparsification_auc"}
    missing_metrics = required_metrics - set(metrics)
    if missing_metrics:
        raise ValueError(f"Held-out metrics are incomplete: missing {sorted(missing_metrics)}")
            
    # Model configuration metadata placeholder
    config_meta = {
        "project_id": "SUTRAM",
        "model_name": "Project SUTRAM",
        "version": "1.0.0",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "K_components": 6,
        "precision": "fp32",
        "preprocessing": {
            "tir_representation": "brightness_temperature_kelvin",
            "normalization": "planck_dn_to_tb_then_fixed_tb_range",
        },
        "hyperparameters": {
            "learning_rate_stage1": 3e-4,
            "learning_rate_stage2": 1e-4,
            "batch_size": 16,
            "gradient_clipping": 1.0,
            "epochs": 15
        }
    }
    
    # Package 1: stage1_sr_best.pth
    logger.info("Packaging stage1_sr_best.pth...")
    stage1_ckpt = {
        "project_id": "SUTRAM",
        "model_name": "SUTRAM Stage 1 Super-Resolution",
        "epoch": 15,
        "backbone_state_dict": bb_state,
        "sr_head_state_dict": sr_state,
        "metrics": {"psnr": metrics["psnr"], "ssim": metrics["ssim"], "bt_rmse": metrics["bt_rmse"]},
        "config": config_meta
    }
    torch.save(stage1_ckpt, os.path.join(dest_dir, "stage1_sr_best.pth"))
    
    # Package 2: stage2_color_best.pth
    logger.info("Packaging stage2_color_best.pth...")
    stage2_ckpt = {
        "project_id": "SUTRAM",
        "model_name": "SUTRAM Stage 2 Colorization",
        "epoch": 15,
        "backbone_state_dict": bb_state,
        "mixture_head_state_dict": mix_state,
        "metrics": {"ece": metrics["ece"], "sparsification_auc": metrics["sparsification_auc"]},
        "config": config_meta
    }
    torch.save(stage2_ckpt, os.path.join(dest_dir, "stage2_color_best.pth"))
    
    # Package 3: sutram_final.pth
    logger.info("Packaging sutram_final.pth...")
    final_ckpt = {
        "project_id": "SUTRAM",
        "model_name": "Project SUTRAM End-to-End Release",
        "version": "1.0.0",
        "epoch": 15,
        "backbone_state_dict": bb_state,
        "sr_head_state_dict": sr_state,
        "mixture_head_state_dict": mix_state,
        "metrics": metrics,
        "config": config_meta
    }
    torch.save(final_ckpt, os.path.join(dest_dir, "sutram_final.pth"))
    
    logger.info("All checkpoints successfully packaged in checkpoints/.")

if __name__ == "__main__":
    main()
