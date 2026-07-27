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

def run_evaluation_report(config_path="configs/base_config.yaml", weights_path=None, split="val"):
    """
    Evaluates the trained Stage 1 and Stage 2 models on the specified split (val or test).
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
    
    # Resolve split list
    if split == "val":
        product_ids = cfg.data.splits.val
    elif split == "test":
        product_ids = cfg.data.splits.test
    else:
        raise ValueError(f"Unknown split: {split}")

    # Validate splits using split_validator
    from data_pipeline.split_validator import run_all_validation
    splits_dict = OmegaConf.to_container(cfg.data.splits, resolve=True)
    run_all_validation(splits_dict, cfg.data.patches_dir, input_dir=cfg.data.input_dir)

    # Load dataset
    eval_dataset = PatchDataset(
        patches_dir=cfg.data.patches_dir,
        product_ids=product_ids
    )
    if len(eval_dataset) == 0:
        logger.error(f"Evaluation dataset for split '{split}' is empty. Cannot generate report.")
        return

    backbone = ResNetBackbone().to(device)
    sr_head = SRHead().to(device)
    mixture_head = MixtureHead(K=cfg.training.stage2.K).to(device)

    if weights_path is not None:
        logger.info(f"Loading evaluation weights from package: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        backbone.load_state_dict(checkpoint["backbone_state_dict"])
        sr_head.load_state_dict(checkpoint["sr_head_state_dict"])
        mixture_head.load_state_dict(checkpoint["mixture_head_state_dict"])
    else:
        # Load trained weights from default checkpoint dir
        checkpoint_dir = os.path.join("experiments", cfg.experiment_id, "checkpoints")
        bb_path = os.path.join(checkpoint_dir, "backbone_stage1.pth")
        sr_path = os.path.join(checkpoint_dir, "sr_head_stage1.pth")
        mix_path = os.path.join(checkpoint_dir, "mixture_head_stage2.pth")
        final_path = os.path.join("checkpoints", "sutram_final.pth")

        if os.path.exists(bb_path) and os.path.exists(sr_path) and os.path.exists(mix_path):
            backbone.load_state_dict(torch.load(bb_path, map_location=device))
            sr_head.load_state_dict(torch.load(sr_path, map_location=device))
            mixture_head.load_state_dict(torch.load(mix_path, map_location=device))
        elif os.path.exists(final_path):
            logger.info(f"Loading evaluation weights from release package: {final_path}")
            checkpoint = torch.load(final_path, map_location=device)
            backbone.load_state_dict(checkpoint["backbone_state_dict"])
            sr_head.load_state_dict(checkpoint["sr_head_state_dict"])
            mixture_head.load_state_dict(checkpoint["mixture_head_state_dict"])
        else:
            logger.error("Trained model checkpoints are missing. Please complete training first.")
            return

    backbone.eval()
    sr_head.eval()
    mixture_head.eval()

    logger.info(f"Computing metrics over {split} dataset ({len(eval_dataset)} patches)...")
    
    sr_preds = []
    sr_targets = []
    
    rgb_preds = []
    rgb_targets = []
    rgb_scales = []
    
    # Store sensor tags to segment metrics
    sensor_tags = [] # 'L8' or 'L9'
    scene_ids = []
    
    with torch.no_grad():
        for i in range(len(eval_dataset)):
            sample = eval_dataset[i]
            # low-res input
            lr_tir = sample["tir_200m"].unsqueeze(0).to(device)
            # targets
            hr_tir = sample["tir_100m_512"].squeeze().numpy()
            hr_rgb = sample["rgb_100m_512"].numpy()
            
            # Determine source scene and sensor type
            sample_path = eval_dataset.samples[i]
            scene_id = os.path.basename(os.path.dirname(sample_path))
            scene_ids.append(scene_id)
            sensor_tags.append("L8" if scene_id.startswith("LC08") else "L9")
            
            features = backbone(lr_tir)
            pred_sr = sr_head(features, lr_tir)
            logit_weights, means, log_scales = mixture_head(features, pred_sr)

            # Stage 1 predictions
            sr_np = pred_sr.squeeze().cpu().numpy()
            sr_preds.append(sr_np)
            sr_targets.append(hr_tir)

            # Stage 2 predictions (Decode dominant color and scales)
            pi = torch.softmax(logit_weights, dim=1).squeeze().cpu().numpy() # (K, H, W)
            means_np = means.squeeze().cpu().numpy() # (K, 3, H, W)
            log_scales_np = log_scales.squeeze().cpu().numpy() # (K, 3, H, W)
            scales_np = np.log1p(np.exp(log_scales_np)) + 1.0 # softplus approximation

            # Dominant component index per pixel
            k_star = np.argmax(pi, axis=0) # (H, W)
            
            # Extract dominant mean and scale per pixel
            H, W = k_star.shape
            
            # Vectorized dominant component selection
            k_star_expanded = np.expand_dims(k_star, axis=(0, 1)).repeat(3, axis=1)
            dom_mean = np.take_along_axis(means_np, k_star_expanded, axis=0).squeeze(0)
            dom_scale = np.take_along_axis(scales_np, k_star_expanded, axis=0).squeeze(0)

            rgb_preds.append(dom_mean)
            rgb_targets.append(hr_rgb)
            rgb_scales.append(dom_scale)

    sr_preds = np.array(sr_preds)
    sr_targets = np.array(sr_targets)
    rgb_preds = np.array(rgb_preds)
    rgb_targets = np.array(rgb_targets)
    rgb_scales = np.array(rgb_scales)
    sensor_tags = np.array(sensor_tags)

    if len(sr_preds) == 0:
        logger.error(f"No valid evaluation patches found in split '{split}'. Cannot calculate metrics.")
        return

    def calculate_metrics_subset(mask):
        if not np.any(mask):
            return None
        sub_sr_preds = sr_preds[mask]
        sub_sr_targets = sr_targets[mask]
        sub_rgb_preds = rgb_preds[mask]
        sub_rgb_targets = rgb_targets[mask]
        sub_rgb_scales = rgb_scales[mask]
        
        psnr_val = compute_psnr(sub_sr_preds, sub_sr_targets, peak=1.0)
        rmse_val = compute_bt_rmse(sub_sr_preds, sub_sr_targets)
        ssim_val = np.mean([compute_ssim(p, t) for p, t in zip(sub_sr_preds, sub_sr_targets)])
        ece_val, empirical_coverages = compute_regression_ece(sub_rgb_preds, sub_rgb_scales, sub_rgb_targets)
        
        abs_errors = np.abs(sub_rgb_preds - sub_rgb_targets).mean(axis=1)
        mean_scales = sub_rgb_scales.mean(axis=1)
        auc_val, _ = compute_sparsification_auc(abs_errors, mean_scales)
        
        return {
            "psnr": float(psnr_val),
            "ssim": float(ssim_val),
            "bt_rmse": float(rmse_val),
            "ece": float(ece_val),
            "ece_bins": empirical_coverages,
            "sparsification_auc": float(auc_val)
        }

    # 1. Overall Metrics
    overall_metrics = calculate_metrics_subset(np.ones(len(sr_preds), dtype=bool))
    if overall_metrics is None:
        logger.error(f"Calculated overall_metrics is None for split '{split}'. Cannot save report.")
        return
    
    # 2. Landsat-8 subset
    l8_mask = (sensor_tags == "L8")
    l8_metrics = calculate_metrics_subset(l8_mask)
    
    # 3. Landsat-9 subset
    l9_mask = (sensor_tags == "L9")
    l9_metrics = calculate_metrics_subset(l9_mask)

    # Determine subset labels (e.g. single-scene vs satellite-wide)
    unique_l8_scenes = sorted(list(set([scene_ids[i] for i in range(len(scene_ids)) if sensor_tags[i] == "L8"])))
    unique_l9_scenes = sorted(list(set([scene_ids[i] for i in range(len(scene_ids)) if sensor_tags[i] == "L9"])))
    
    l8_label = "Held-out Landsat-8 scene" if len(unique_l8_scenes) == 1 else "Landsat-8 subset"
    if len(unique_l8_scenes) == 1:
        l8_label += f" ({unique_l8_scenes[0]})"
        
    l9_label = "Held-out Landsat-9 scene" if len(unique_l9_scenes) == 1 else "Landsat-9 subset"
    if len(unique_l9_scenes) == 1:
        l9_label += f" ({unique_l9_scenes[0]})"

    report_data = {
        "Overall": overall_metrics,
        l8_label: l8_metrics,
        l9_label: l9_metrics
    }

    # Save to metrics.json
    metrics_dir = os.path.join("experiments", cfg.experiment_id)
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"metrics_{split}.json")
    with open(metrics_path, "w") as f:
        json.dump(report_data, f, indent=4)
    logger.info(f"Saved evaluation metrics to: {metrics_path}")

    # Output to console
    print(f"\n================ EVALUATION REPORT ON SPLIT: {split.upper()} ================")
    print(f"Overall Metrics: {overall_metrics}")
    if l8_metrics:
        print(f"{l8_label} Metrics: {l8_metrics}")
    if l9_metrics:
        print(f"{l9_label} Metrics: {l9_metrics}")
    print("=================================================================\n")

    # Generate plots for overall
    os.makedirs(os.path.join("experiments", cfg.experiment_id, "validation_plots"), exist_ok=True)
    abs_errors_all = np.abs(rgb_preds - rgb_targets).mean(axis=1)
    mean_scales_all = rgb_scales.mean(axis=1)
    _, error_curve = compute_sparsification_auc(abs_errors_all, mean_scales_all)
    
    plot_sparsification_curve(
        error_curve,
        os.path.join("experiments", cfg.experiment_id, "validation_plots", f"sparsification_curve_{split}.png")
    )
    
    ece_bins = overall_metrics.get("ece_bins") if overall_metrics else None
    if ece_bins is not None:
        plot_calibration_error(
            ece_bins,
            os.path.join("experiments", cfg.experiment_id, "validation_plots", f"calibration_reliability_{split}.png")
        )
    else:
        logger.warning("Skipping calibration reliability plot because ece_bins are unavailable.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation_report()
