import numpy as np
import cv2

def compute_psnr(pred, target, peak=350.0):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR).
    Peak temperature is set to 350.0 Kelvin by default.
    """
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((peak ** 2) / mse)

def compute_ssim(pred, target):
    """
    Computes Structural Similarity Index (SSIM) using OpenCV.
    Supports single channel or multi-channel arrays.
    """
    # Simply use cv2 quality metrics or manual simplified SSIM to avoid dependencies
    # Standard formula implementation
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    mu_x = cv2.GaussianBlur(pred, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(target, (11, 11), 1.5)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = cv2.GaussianBlur(pred ** 2, (11, 11), 1.5) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(target ** 2, (11, 11), 1.5) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(pred * target, (11, 11), 1.5) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    
    ssim_map = num / (den + 1e-5)
    return np.mean(ssim_map)

def compute_bt_rmse(pred, target):
    """Computes Brightness-Temperature Root Mean Squared Error."""
    return np.sqrt(np.mean((pred - target) ** 2))

def compute_per_class_color_accuracy(pred_rgb, target_rgb, class_mask):
    """
    Computes average L1 color error per land-cover class.
    class_mask should be of shape (H, W) with class labels.
    """
    unique_classes = np.unique(class_mask)
    class_errors = {}
    for c in unique_classes:
        mask = (class_mask == c)
        if not np.any(mask):
            continue
        err = np.abs(pred_rgb[:, mask] - target_rgb[:, mask]).mean()
        class_errors[int(c)] = float(err)
    return class_errors

def compute_regression_ece(pred_means, pred_scales, targets, num_bins=10):
    """
    Computes Expected Calibration Error (ECE) for regression.
    For each nominal confidence level p, computes the empirical fraction of target values
    that fall within the predicted p-probability confidence intervals.
    """
    # Flatten arrays
    means = pred_means.flatten()
    scales = pred_scales.flatten()
    y_true = targets.flatten()
    
    ece = 0.0
    p_levels = np.linspace(0.1, 0.9, num_bins)
    
    for p in p_levels:
        # Logistic distribution percentiles corresponding to confidence level p
        # For a logistic distribution, CDF(x) = sigmoid((x - mean) / scale)
        # The interval containing probability p is centered around the mean:
        # [sigmoid_inverse((1-p)/2), sigmoid_inverse((1+p)/2)]
        half_alpha = (1.0 - p) / 2.0
        z_p = np.log((1.0 - half_alpha) / half_alpha) # inverse sigmoid / logit
        
        lower_bound = means - z_p * scales
        upper_bound = means + z_p * scales
        
        # Count empirical coverage
        inside = (y_true >= lower_bound) & (y_true <= upper_bound)
        empirical_coverage = np.mean(inside)
        
        ece += np.abs(empirical_coverage - p)
        
    return float(ece / num_bins)

def compute_sparsification_auc(pred_errors, pred_uncertainties):
    """
    Computes Sparsification Curve Area Under Curve (AUC).
    Progressively discards high-uncertainty pixels and tracks the remaining average error.
    """
    errors = pred_errors.flatten()
    uncertainties = pred_uncertainties.flatten()
    
    # Sort pixels by uncertainty descending
    sort_idx = np.argsort(uncertainties)[::-1]
    sorted_errors = errors[sort_idx]
    
    # Calculate error at different rejection rates
    rejection_rates = np.linspace(0.0, 0.99, 100)
    error_curve = []
    total_pixels = len(sorted_errors)
    
    for r in rejection_rates:
        discard_count = int(r * total_pixels)
        remaining_errors = sorted_errors[discard_count:]
        error_curve.append(np.mean(remaining_errors) if len(remaining_errors) > 0 else 0.0)
        
    # Standardize AUC relative to baseline (first point)
    auc = np.mean(error_curve) / (error_curve[0] + 1e-5)
    return float(auc), error_curve
