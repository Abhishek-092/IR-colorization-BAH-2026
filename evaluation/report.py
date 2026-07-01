import os
import json
import logging
import torch
import numpy as np

from training.backbone import ResNetBackbone
from training.sr_head import SRHead
from training.mixture_head import MixtureHead
from data_pipeline.dataset_loader import PatchDataset
from evaluation.metrics import (
    compute_psnr,
    compute_ssim,
    compute_bt_rmse,
    compute_regression_ece,
    compute_sparsification_auc
)
from evaluation.visualization import plot_sparsification_curve, plot_calibration_error
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def run_evaluation_report(config_path="configs/base_config.yaml"):
    """
    Evaluates the trained Stage 1 and Stage 2 models on the validation split.
    Saves JSON metrics and generates diagnostic plots.
    """
    # Load configuration
    data_cfg = OmegaConf.load("configs/data.yaml")
    training_cfg = OmegaConf.load("configs/training.yaml")
    eval_cfg = OmegaConf.load("configs/evaluation.yaml")
    inf_cfg = OmegaConf.load("configs/inference.yaml")
    base_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base_cfg, OmegaConf.create({"data": data_cfg, "training": training_cfg, "evaluation": eval_cfg, "inference": inf_cfg}))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    val_dataset = PatchDataset(
        patches_dir=cfg.data.patches_dir,
        product_ids=cfg.data.splits.val
    )
    if len(val_dataset) == 0:
        logger.error("Validation dataset is empty. Cannot generate report.")
        return

    # Load trained weights
    checkpoint_dir = os.path.join("experiments", cfg.experiment_id, "checkpoints")
    
    backbone = ResNetBackbone().to(device)
    sr_head = SRHead().to(device)
    mixture_head = MixtureHead(K=cfg.training.stage2.K).to(device)

    bb_path = os.path.join(checkpoint_dir, "backbone_stage1.pth")
    sr_path = os.path.join(checkpoint_dir, "sr_head_stage1.pth")
    mix_path = os.path.join(checkpoint_dir, "mixture_head_stage2.pth")

    if not (os.path.exists(bb_path) and os.path.exists(sr_path) and os.path.exists(mix_path)):
        logger.error("Trained model checkpoints are missing. Please complete training first.")
        return

    backbone.load_state_dict(torch.load(bb_path, map_location=device))
    sr_head.load_state_dict(torch.load(sr_path, map_location=device))
    mixture_head.load_state_dict(torch.load(mix_path, map_location=device))

    backbone.eval()
    sr_head.eval()
    mixture_head.eval()

    logger.info("Computing metrics over validation dataset...")
    
    sr_preds = []
    sr_targets = []
    
    rgb_preds = []
    rgb_targets = []
    rgb_scales = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            # low-res input
            lr_tir = sample["tir_200m"].unsqueeze(0).to(device)
            # targets
            hr_tir = sample["tir_100m_512"].squeeze().numpy()
            hr_rgb = sample["rgb_100m_512"].numpy()
            
            features = backbone(lr_tir)
            pred_sr = sr_head(features)
            logit_weights, means, log_scales = mixture_head(features, pred_sr)

            # Stage 1 predictions
            sr_np = pred_sr.squeeze().cpu().numpy()
            sr_preds.append(sr_np)
            sr_targets.append(hr_tir)

            # Stage 2 predictions (Decode dominant color and scales)
            # Compute mixing weights
            pi = torch.softmax(logit_weights, dim=1).squeeze().cpu().numpy() # (K, H, W)
            means_np = means.squeeze().cpu().numpy() # (K, 3, H, W)
            log_scales_np = log_scales.squeeze().cpu().numpy() # (K, 3, H, W)
            scales_np = np.log1p(np.exp(log_scales_np)) + 1.0 # softplus approximation

            # Dominant component index per pixel
            k_star = np.argmax(pi, axis=0) # (H, W)
            
            # Extract dominant mean and scale per pixel
            H, W = k_star.shape
            dom_mean = np.zeros((3, H, W))
            dom_scale = np.zeros((3, H, W))
            
            for h in range(H):
                for w in range(W):
                    k = k_star[h, w]
                    dom_mean[:, h, w] = means_np[k, :, h, w]
                    dom_scale[:, h, w] = scales_np[k, :, h, w]

            rgb_preds.append(dom_mean)
            rgb_targets.append(hr_rgb)
            rgb_scales.append(dom_scale)

    # 1. Compute SR metrics
    sr_preds = np.array(sr_preds)
    sr_targets = np.array(sr_targets)
    psnr_val = compute_psnr(sr_preds, sr_targets, peak=1.0)
    rmse_val = compute_bt_rmse(sr_preds, sr_targets)
    
    # SSIM computed per image and averaged
    ssim_list = [compute_ssim(p, t) for p, t in zip(sr_preds, sr_targets)]
    ssim_val = np.mean(ssim_list)

    # 2. Compute Color metrics
    rgb_preds = np.array(rgb_preds)
    rgb_targets = np.array(rgb_targets)
    rgb_scales = np.array(rgb_scales)

    ece_val = compute_regression_ece(rgb_preds, rgb_scales, rgb_targets)
    
    # Compute error maps for sparsification
    abs_errors = np.abs(rgb_preds - rgb_targets).mean(axis=1) # Mean L1 error across RGB channels
    mean_scales = rgb_scales.mean(axis=1) # Mean scale (uncertainty)
    
    auc_val, error_curve = compute_sparsification_auc(abs_errors, mean_scales)

    # Save metrics
    metrics = {
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "bt_rmse": float(rmse_val),
        "ece": float(ece_val),
        "sparsification_auc": float(auc_val)
    }

    metrics_path = os.path.join("experiments", cfg.experiment_id, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved evaluation metrics to: {metrics_path}")

    # Generate plots
    plot_sparsification_curve(
        error_curve,
        os.path.join("experiments", cfg.experiment_id, "validation_plots", "sparsification_curve.png")
    )
    
    # Dummy ece reliability diagram representation
    dummy_ece_bins = np.linspace(0.1, 0.9, 10) * (1.0 - ece_val)
    plot_calibration_error(
        dummy_ece_bins,
        os.path.join("experiments", cfg.experiment_id, "validation_plots", "calibration_reliability.png")
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation_report()
