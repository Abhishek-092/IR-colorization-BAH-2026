"""
scripts/download_models.py — Download SD-Turbo weights and verify SUTRAM checkpoint
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def verify_sutram_checkpoint():
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "sutram_final.pth")
    if os.path.exists(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        logger.info(f"SUTRAM model checkpoint verified at: {ckpt_path} ({size_mb:.2f} MB)")
        return True
    else:
        logger.warning(f"SUTRAM checkpoint not found at: {ckpt_path}")
        return False

def predownload_sd_turbo():
    logger.info("Initializing SD-Turbo download from HuggingFace (stabilityai/sd-turbo)...")
    try:
        import torch
        from diffusers import AutoPipelineForImage2Image
        
        model_id = os.environ.get("SUTRAM_REFINER_MODEL", "stabilityai/sd-turbo")
        device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info(f"Downloading pipeline '{model_id}' for device '{device}'...")
        pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=dtype)
        logger.info("SD-Turbo model weights successfully downloaded.")
        return True
    except Exception as e:
        logger.error(f"Failed to download SD-Turbo model: {e}")
        return False

if __name__ == "__main__":
    logger.info("=== SUTRAM Setup Verification ===")
    sutram_ok = verify_sutram_checkpoint()
    sd_ok = predownload_sd_turbo()
    
    if sutram_ok and sd_ok:
        logger.info("All required model weights verified and ready.")
    else:
        logger.info("Setup check finished with warnings.")

