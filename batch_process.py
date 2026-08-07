import os
import sys
import glob
import torch
import numpy as np
from PIL import Image

# Add root directory to sys.path to allow running from anywhere
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import model and helpers from webapp server
from webapp.server import MODEL, read_thermal_any, resize_to, normalize_rgb, display_stretch
from sutram.calibration.autocalibrate import calibrate_thermal_to_norm
from sutram.calibration.planck import TB_MIN, TB_MAX
from inference.fusion import fuse_reconstruction
from inference import refiner  # Track-2 SD-Turbo photorealistic refiner (on by default)

# Refiner on unless SUTRAM_REFINER=0; graceful no-op if diffusers/weights absent.
REFINER_ON = os.environ.get("SUTRAM_REFINER", "1") == "1"


def main():
    deliverables_dir = os.path.join(root_dir, "deliverables")
    data_dir = os.path.join(deliverables_dir, "data")

    input_dir = os.path.join(data_dir, "tir_200")
    sr_dir = os.path.join(data_dir, "tir_100")
    synthesis_dir = os.path.join(data_dir, "rgb_100")

    # Create directories if they do not exist
    for d in [input_dir, sr_dir, synthesis_dir]:
        os.makedirs(d, exist_ok=True)

    print("=== SUTRAM Batch Processing ===")
    print(f"Deliverables directory: {deliverables_dir}")
    print(f"Place your input images in: {input_dir}")
    print(f"Super Resolved outputs will be saved in: {sr_dir}")
    print(f"RGB outputs will be saved in: {synthesis_dir}\n")

    # Find all PNG/TIFF images in input_dir
    image_patterns = [
        "*.png", "*.PNG",
        "*.jpg", "*.JPG",
        "*.jpeg", "*.JPEG",
        "*.tif", "*.TIF",
        "*.tiff", "*.TIFF"
    ]

    input_files = []
    for pattern in image_patterns:
        input_files.extend(glob.glob(os.path.join(input_dir, pattern)))

    input_files = sorted(list(set(input_files)))

    if not input_files:
        print("No input images found in deliverables/data/tir_200/.")
        print("Please place input 200m thermal PNG/TIFF images in that folder and run the script again.")
        return

    print(f"Found {len(input_files)} image(s) to process.\n")

    for idx, filepath in enumerate(input_files):
        filename = os.path.basename(filepath)
        print(f"[{idx+1}/{len(input_files)}] Processing {filename}...")

        try:
            # Read input using read_thermal_any (handles TIFF, PNG, etc.)
            lr_np = read_thermal_any(filepath)

            # Resize to 256x256 if not already
            lr256 = resize_to(lr_np, 256) if lr_np.shape != (256, 256) else lr_np

            # Calibrate and normalize -> [0, 1] BT tensor
            lr_t = torch.from_numpy(
                calibrate_thermal_to_norm(lr256)
            ).unsqueeze(0).unsqueeze(0).float()

            # Run inference
            res = MODEL.infer(lr_t)

            bt_sr = res["sr_bt"]
            rgb_u8 = normalize_rgb(res["rgb_raw"])
            rgb_view = display_stretch(rgb_u8)

            fused = fuse_reconstruction(
                bt_sr,
                rgb_view,
                res["entropy"],
                K=MODEL.K,
                confidence=res["confidence"]
            )

            # Track-2 photorealistic refiner (keeps colours, adds satellite realism)
            if REFINER_ON:
                fused = refiner.refine(fused)

            # Save Super Resolved Thermal Image
            sr_norm = np.clip((bt_sr - TB_MIN) / (TB_MAX - TB_MIN), 0.0, 1.0)
            sr_u8 = (sr_norm * 255.0).astype(np.uint8)
            sr_img = Image.fromarray(sr_u8, mode="L")

            # Save using the same input name
            out_filename = os.path.splitext(filename)[0] + ".png"
            sr_img.save(os.path.join(sr_dir, out_filename), format="PNG")

            # Save Final RGB Reconstruction
            synthesis_img = Image.fromarray(fused, mode="RGB")
            synthesis_img.save(os.path.join(synthesis_dir, out_filename), format="PNG")

            print(f"  -> Saved tir_100/{out_filename}")
            print(f"  -> Saved rgb_100/{out_filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    print("\nBatch processing completed successfully!")


if __name__ == "__main__":
    main()