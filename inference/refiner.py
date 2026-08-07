"""
inference/refiner.py — Realism refiner (Track 2)
================================================
A pretrained latent-diffusion prior (SD-Turbo) used as a LOW-STRENGTH img2img
pass on top of the model's predicted RGB. It does NOT generate colour from
scratch — the predicted RGB is the init image and the diffusion only runs for a
short denoising span (strength ~0.3), so it *keeps* the model's colours and
layout and adds realistic satellite texture/tone on top.

Design goals:
  * accuracy-preserving — low strength + optional blend keep the physics-driven
    colours; the diffusion is a renderer, not a hallucinator let loose.
  * fully optional & graceful — if `diffusers` isn't installed or the model
    can't load, refine() returns the input unchanged, so the core pipeline and
    the portable zip keep working with zero extra dependencies.
  * lazy & cached — the pipeline loads once on first use.

Toggle at runtime with SUTRAM_REFINER=1 (off by default).
"""
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

_PIPE = None
_TRIED = False
_MODEL = os.environ.get("SUTRAM_REFINER_MODEL", "stabilityai/sd-turbo")
_PROMPT = ("high resolution aerial satellite photograph, natural true color, "
           "sharp land cover detail, fields forests water, realistic")


def available():
    """True if the refiner can (probably) run — diffusers importable."""
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _load():
    """Lazily build the img2img pipeline. Returns None on any failure."""
    global _PIPE, _TRIED
    if _PIPE is not None or _TRIED:
        return _PIPE
    _TRIED = True
    try:
        import torch
        from diffusers import AutoPipelineForImage2Image
        device = ("cuda" if torch.cuda.is_available()
                  else ("mps" if torch.backends.mps.is_available() else "cpu"))
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = AutoPipelineForImage2Image.from_pretrained(_MODEL, torch_dtype=dtype)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        try:
            pipe.safety_checker = None
        except Exception:
            pass
        _PIPE = pipe
        logger.info(f"[refiner] loaded {_MODEL} on {device}")
    except Exception as e:
        logger.warning(f"[refiner] unavailable ({type(e).__name__}: {e}); "
                       f"returning predicted RGB unchanged")
        _PIPE = None
    return _PIPE


def refine(rgb_u8, strength=0.30, steps=4, blend=0.35, seed=42):
    """Refine a predicted RGB image (H,W,3 uint8) toward photorealism.

    strength : diffusion denoising span (0..1). Low = stay close to input.
    steps    : sampler steps (turbo needs only a few).
    blend    : final = (1-blend)*refined + blend*original — a safety anchor that
               pulls colour back toward the physics-driven prediction so the
               diffusion can sharpen texture without shifting the palette.
    Returns a (H,W,3) uint8 array. On any failure, returns rgb_u8 unchanged.
    """
    pipe = _load()
    if pipe is None:
        return rgb_u8
    try:
        import torch
        from PIL import Image
        H, W = rgb_u8.shape[:2]
        init = Image.fromarray(rgb_u8, "RGB").resize((512, 512), Image.BICUBIC)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        out = pipe(prompt=_PROMPT, image=init, strength=float(strength),
                   num_inference_steps=int(steps), guidance_scale=0.0,
                   generator=g).images[0]
        ref = np.asarray(out.resize((W, H), Image.BICUBIC), dtype=np.float32)
        base = rgb_u8.astype(np.float32)
        mixed = (1.0 - blend) * ref + blend * base
        return np.clip(mixed, 0, 255).astype(np.uint8)
    except Exception as e:
        logger.warning(f"[refiner] refine failed ({type(e).__name__}: {e}); "
                       f"returning predicted RGB unchanged")
        return rgb_u8
