import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def validate_sutram_config(cfg: DictConfig) -> bool:
    """
    Validates configuration values and bounds.
    Returns True if valid, raises ValueError otherwise.
    """
    try:
        # Check scale factors
        scale_rgb = cfg.data.scale_factors.rgb_downscale
        scale_tir_100 = cfg.data.scale_factors.tir_100m_downscale
        scale_tir_200 = cfg.data.scale_factors.tir_200m_downscale

        if not (3.32 < scale_rgb < 3.34):
            raise ValueError(f"Invalid RGB scale factor: {scale_rgb}. Expected: 3.33")
        if not (3.32 < scale_tir_100 < 3.34):
            raise ValueError(f"Invalid TIR 100m scale factor: {scale_tir_100}. Expected: 3.33")
        if not (6.66 < scale_tir_200 < 6.68):
            raise ValueError(f"Invalid TIR 200m scale factor: {scale_tir_200}. Expected: 6.67")

        # Check patch sizes
        if cfg.data.patch_sizes.tir_200m != 256:
            raise ValueError(f"Invalid 200m TIR patch size: {cfg.data.patch_sizes.tir_200m}. Expected: 256")
        if cfg.data.patch_sizes.tir_100m != 512:
            raise ValueError(f"Invalid 100m TIR patch size: {cfg.data.patch_sizes.tir_100m}. Expected: 512")

        # Check Stage 2 Mixture Components (K)
        K = cfg.training.stage2.K
        if K not in [6, 7, 8]:
            raise ValueError(f"Mixture components K must be in [6, 7, 8], got: {K}")

        # Check learning rates
        if cfg.training.stage1.lr <= 0 or cfg.training.stage2.lr <= 0:
            raise ValueError("Learning rates must be positive values.")

        logger.info("Configuration validation: PASSED.")
        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise e
