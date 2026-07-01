import os
import glob
import logging
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

def percentile_stretch(img, percentiles=(2, 98)):
    """Applies percentile stretch to image for high-quality visualization."""
    if img.ndim == 3:
        stretched = np.zeros_like(img)
        for i in range(img.shape[0]):
            stretched[i] = percentile_stretch(img[i], percentiles)
        return stretched
        
    low, high = np.percentile(img, percentiles)
    if high <= low:
        return np.zeros_like(img)
    stretched = (img - low) / (high - low)
    return np.clip(stretched, 0.0, 1.0)

def generate_sample_results():
    """
    Generates a publication-quality multi-page PDF document and high-res PNGs
    representing the Project VARNA image enhancement outputs.
    """
    output_dir = "output/sample_results"
    os.makedirs(output_dir, exist_ok=True)
    
    sr_paths = glob.glob("output/model_outputs/tir_superresolved_100m/*.tif")
    if not sr_paths:
        logger.error("No model outputs found. Run 'cli.py export' first.")
        return

    # Gather data for available scenes
    scenes_data = []
    for sr_path in sr_paths:
        prod_id = os.path.basename(sr_path).replace(".tif", "")
        lr_path = f"output/downscaled_data/{prod_id}_tir_200m.tif"
        color_path = f"output/model_outputs/colorized_tir_100m/{prod_id}.tif"
        
        if os.path.exists(lr_path) and os.path.exists(color_path):
            with rasterio.open(lr_path) as src:
                lr_img = src.read(1).astype(np.float32)
            with rasterio.open(sr_path) as src:
                sr_img = src.read(1).astype(np.float32)
            with rasterio.open(color_path) as src:
                color_img = src.read().astype(np.float32)
                
            color_rgb = np.stack([color_img[2], color_img[1], color_img[0]], axis=-1)
            
            scenes_data.append({
                "id": prod_id,
                "lr": percentile_stretch(lr_img),
                "sr": percentile_stretch(sr_img),
                "color": percentile_stretch(color_rgb)
            })

    if not scenes_data:
        logger.error("No matching scenes found for sample results generation.")
        return

    pdf_out = os.path.join(output_dir, "Sample_Results.pdf")
    
    with PdfPages(pdf_out) as pdf:
        plt.rcParams['font.family'] = 'sans-serif'
        
        # ----------------------------------------------------
        # PAGE 1: COVER PAGE
        # ----------------------------------------------------
        fig = plt.figure(figsize=(11, 8.5), facecolor="#ffffff")
        ax = fig.add_subplot(111)
        ax.axis("off")
        
        # Background decorations
        fig.patch.set_facecolor('#f7f9fc')
        
        # Text blocks
        plt.text(0.5, 0.70, "PROJECT VARNA", ha="center", va="center", fontsize=38, fontweight="bold", color="#1c2331")
        plt.text(0.5, 0.62, "Bharatiya Antriksh Hackathon (BAH) 2026", ha="center", va="center", fontsize=18, color="#ff4500", fontweight="semibold")
        plt.text(0.5, 0.58, "Submission Category: Thermal-to-Optical Image Enhancement", ha="center", va="center", fontsize=13, color="#555555")
        
        # Separator line
        plt.plot([0.2, 0.8], [0.5, 0.5], color="#1e90ff", lw=2, transform=ax.transAxes)
        
        desc_text = (
            "Abstract & System Description:\n\n"
            "Project VARNA implements a dual-stage neural pipeline designed to super-resolve single-band\n"
            "thermal infrared (TIR) images (200m) and colorize them to optical-grade multispectral RGB (100m).\n"
            "Leveraging a residual super-resolution module and a discretized logistic mixture model, the system\n"
            "reconstructs fine structural components while managing predictive radiometric uncertainty."
        )
        plt.text(0.5, 0.32, desc_text, ha="center", va="center", fontsize=11, linespacing=1.5, color="#333333", bbox=dict(boxstyle="round,pad=1.2", facecolor="#ffffff", edgecolor="#e2e8f0", alpha=0.9))
        
        plt.text(0.5, 0.10, "Developed by Team Antigravity • Official Submission Document", ha="center", va="center", fontsize=9, color="#777777")
        
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        
        # ----------------------------------------------------
        # PAGE 2: GRID OVERVIEW (3-4 representative scenes)
        # ----------------------------------------------------
        overview_scenes = scenes_data[:4]
        num_overview = len(overview_scenes)
        
        fig, axes = plt.subplots(num_overview, 3, figsize=(11, 2.5 * num_overview + 1), facecolor="#ffffff")
        if num_overview == 1:
            axes = np.expand_dims(axes, axis=0)
            
        fig.suptitle("Performance Overview: Multi-Scene Enhancements", fontsize=16, fontweight="bold", y=0.96)
        
        # Column Labels (Top row only)
        axes[0, 0].set_title("Raw Input TIR (200m)", fontsize=11, fontweight="bold", pad=8)
        axes[0, 1].set_title("Super-Resolved TIR (100m)", fontsize=11, fontweight="bold", pad=8)
        axes[0, 2].set_title("Colorized Optical RGB (100m)", fontsize=11, fontweight="bold", pad=8)
        
        for idx, scene in enumerate(overview_scenes):
            # Plot
            axes[idx, 0].imshow(scene["lr"], cmap="inferno")
            axes[idx, 1].imshow(scene["sr"], cmap="inferno")
            axes[idx, 2].imshow(scene["color"])
            
            # Row label
            axes[idx, 0].set_ylabel(scene["id"][:20] + "...", fontsize=9, fontweight="semibold")
            
            # Formatting
            for col in range(3):
                axes[idx, col].set_xticks([])
                axes[idx, col].set_yticks([])
                for spine in axes[idx, col].spines.values():
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.5)
                    
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        
        # ----------------------------------------------------
        # PAGES 3+: DETAILED SINGLE-SCENE RESULTS
        # ----------------------------------------------------
        for scene in scenes_data:
            fig = plt.figure(figsize=(11, 8.5), facecolor="#ffffff")
            gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1.2], wspace=0.12, hspace=0.25)
            
            # 1. Raw TIR
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(scene["lr"], cmap="inferno")
            ax0.set_title("Raw Input TIR (200m)", fontsize=12, fontweight="bold", pad=8)
            ax0.axis("off")
            
            # 2. Super-Resolved TIR
            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(scene["sr"], cmap="inferno")
            ax1.set_title("Super-Resolved TIR (100m)", fontsize=12, fontweight="bold", pad=8)
            ax1.axis("off")
            
            # 3. Colorized RGB
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.imshow(scene["color"])
            ax2.set_title("Colorized Optical RGB (100m)", fontsize=12, fontweight="bold", pad=8)
            ax2.axis("off")
            
            # Metadata Text Block underneath
            ax_text = fig.add_subplot(gs[1, :])
            ax_text.axis("off")
            
            metadata_box = (
                f"Scene Product ID: {scene['id']}\n"
                f"Resolution Enhancement: 200m (TIRS-2) → 100m (OLI Spatial Alignment)\n"
                f"Model Architecture: VARNA (Variational class-Aware Radiance-to-reflectance Network)\n"
                f"Processing Pipeline: Phased PixelShuffle Upscaling & Variational Logistic Mixture Decoder"
            )
            ax_text.text(0.02, 0.5, metadata_box, fontsize=10, fontweight="semibold", color="#1c2331", va="center", linespacing=1.6)
            
            # Title
            fig.suptitle(f"VARNA Visual Reconstruction Report\nScene: {scene['id']}", 
                         fontsize=14, fontweight="bold", y=0.96)
            
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            
            # Also save individual high-res comparison PNG
            fig_png, axes_png = plt.subplots(1, 3, figsize=(18, 6.5), dpi=300, facecolor="#fafafa")
            plt.subplots_adjust(wspace=0.15, bottom=0.1)
            
            axes_png[0].imshow(scene["lr"], cmap="inferno")
            axes_png[0].set_title("1. Raw TIR Input (200m)", fontsize=13, fontweight="bold", pad=12)
            axes_png[0].axis("off")
            
            axes_png[1].imshow(scene["sr"], cmap="inferno")
            axes_png[1].set_title("2. Super-Resolved TIR (100m)", fontsize=13, fontweight="bold", pad=12)
            axes_png[1].axis("off")
            
            axes_png[2].imshow(scene["color"])
            axes_png[2].set_title("3. Synthesized OLI RGB (100m)", fontsize=13, fontweight="bold", pad=12)
            axes_png[2].axis("off")
            
            fig_png.suptitle(f"VARNA Enhancement Output - Scene {scene['id']}", fontsize=15, fontweight="bold", y=0.98)
            
            png_out = os.path.join(output_dir, f"comparison_{scene['id']}.png")
            fig_png.savefig(png_out, bbox_inches="tight", facecolor=fig_png.get_facecolor(), edgecolor='none')
            plt.close(fig_png)
            logger.info(f"Saved comparison PNG to {png_out}")

    logger.info(f"Successfully generated multi-page Sample_Results.pdf at {pdf_out}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_sample_results()
