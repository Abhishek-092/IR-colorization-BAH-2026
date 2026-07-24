import os
import matplotlib.pyplot as plt
import numpy as np

def percentile_stretch(img, p_min=2, p_max=98):
    """
    Linearly stretches the image values between p_min and p_max percentiles.
    Stretches RGB channels independently and excludes cloud-level saturation
    when computing the max percentile to keep land details from being crushed.
    """
    if img.ndim == 3 and img.shape[-1] == 3:
        stretched = np.zeros_like(img)
        for c in range(3):
            ch = img[..., c]
            is_raw_scale = ch.max() > 255.0
            cloud_thresh = 6000.0 if is_raw_scale else 153.0
            land_pixels = ch[ch < cloud_thresh]
            
            ch_min = np.percentile(ch, p_min)
            if len(land_pixels) > 100:
                ch_max = np.percentile(land_pixels, p_max)
            else:
                ch_max = np.percentile(ch, p_max)
                
            if ch_max - ch_min > 0:
                s = (ch - ch_min) / (ch_max - ch_min)
                stretched[..., c] = np.clip(s, 0.0, 1.0)
        return (stretched * 255).astype(np.uint8)
        
    img_min = np.percentile(img, p_min)
    img_max = np.percentile(img, p_max)
    if img_max - img_min > 0:
        stretched = (img - img_min) / (img_max - img_min)
        stretched = np.clip(stretched, 0.0, 1.0)
    else:
        stretched = np.zeros_like(img)
    return (stretched * 255).astype(np.uint8)

def plot_sparsification_curve(error_curve, save_path):
    """
    Plots the error curve as a function of the fraction of rejected pixels.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    rejection_rates = np.linspace(0.0, 0.99, len(error_curve))
    
    plt.figure(figsize=(8, 6))
    plt.plot(rejection_rates * 100, error_curve, label="SUTRAM", color="blue", linewidth=2)
    
    # Oracle / ideal rejection curve (for reference)
    plt.title("Sparsification Plot (Error vs. Discarded Uncertainty %)")
    plt.xlabel("Percentage of high-uncertainty pixels discarded (%)")
    plt.ylabel("Mean absolute error (MAE)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Sparsification curve saved to: {save_path}")

def plot_calibration_error(ece_by_bin, save_path):
    """
    Plots confidence level vs. empirical coverage to show calibration.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    bins = np.linspace(0.1, 0.9, len(ece_by_bin))
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.bar(bins, ece_by_bin, width=0.08, alpha=0.7, color="cyan", edgecolor="blue", label="Empirical Coverage")
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("Calibration Reliability Diagram")
    plt.xlabel("Nominal Confidence Interval Level")
    plt.ylabel("Empirical In-Interval Coverage")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved to: {save_path}")
