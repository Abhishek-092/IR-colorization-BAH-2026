import os
import glob
import logging
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

def percentile_stretch(img, percentiles=(2, 98)):
    """Applies percentile stretch to image for high-quality visualization."""
    # Handle multi-channel images
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
    Finds output GeoTIFF files, constructs publication-quality comparisons,
    and exports them to PNG, PDF, and generates a descriptive README.md.
    """
    output_dir = "output/sample_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Locate all products in output/model_outputs
    sr_paths = glob.glob("output/model_outputs/tir_superresolved_100m/*.tif")
    if not sr_paths:
        logger.error("No model outputs found in output/model_outputs/tir_superresolved_100m/. Please run 'cli.py export' first.")
        return
        
    pdf_pages = []
    
    for sr_path in sr_paths:
        prod_id = os.path.basename(sr_path).replace(".tif", "")
        logger.info(f"Generating comparison figure for scene: {prod_id}")
        
        # Paths
        lr_path = f"output/downscaled_data/{prod_id}_tir_200m.tif"
        color_path = f"output/model_outputs/colorized_tir_100m/{prod_id}.tif"
        
        if not (os.path.exists(lr_path) and os.path.exists(color_path)):
            logger.warning(f"Missing low-res or colorized outputs for scene: {prod_id}. Skipping.")
            continue
            
        # Read bands
        with rasterio.open(lr_path) as src:
            lr_img = src.read(1).astype(np.float32)
        with rasterio.open(sr_path) as src:
            sr_img = src.read(1).astype(np.float32)
        with rasterio.open(color_path) as src:
            color_img = src.read().astype(np.float32)
            
        # Convert colorized BGR (Layer order: Blue=0, Green=1, Red=2) to RGB for Matplotlib visualization
        # In export, we write Blue to Band 1, Green to Band 2, Red to Band 3
        # In Matplotlib we need RGB (Red, Green, Blue) -> color_img[2], color_img[1], color_img[0]
        color_rgb = np.stack([color_img[2], color_img[1], color_img[0]], axis=-1)
        
        # Apply visualization stretching
        lr_stretched = percentile_stretch(lr_img)
        sr_stretched = percentile_stretch(sr_img)
        color_stretched = percentile_stretch(color_rgb)
        
        # Create publication quality figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=300, facecolor="#fafafa")
        plt.subplots_adjust(wspace=0.15, bottom=0.1)
        
        # Set clean font
        plt.rcParams['font.family'] = 'sans-serif'
        
        # Left Panel: Raw Low-Res TIR
        axes[0].imshow(lr_stretched, cmap="inferno")
        axes[0].set_title("1. Raw TIR Input\n(Spatial Res: 200m)", fontsize=13, fontweight="bold", pad=12)
        axes[0].axis("off")
        
        # Middle Panel: Super-Resolved TIR
        axes[1].imshow(sr_stretched, cmap="inferno")
        axes[1].set_title("2. Super-Resolved TIR (VARNA)\n(Spatial Res: 100m)", fontsize=13, fontweight="bold", pad=12)
        axes[1].axis("off")
        
        # Right Panel: Colorized TIR
        axes[2].imshow(color_stretched)
        axes[2].set_title("3. Synthesized OLI RGB (VARNA)\n(Spatial Res: 100m)", fontsize=13, fontweight="bold", pad=12)
        axes[2].axis("off")
        
        # Scene Title decoration
        fig.suptitle(f"VARNA Satellite Image Enhancement Pipeline\nScene ID: {prod_id}", 
                     fontsize=15, fontweight="bold", y=0.98, color="#111111")
        
        # Footer caption
        fig.text(0.5, 0.02, "Enhancement workflow: 200m single-band Thermal Infrared (TIR) $\\rightarrow$ Spatial Super-Resolution 2x $\\rightarrow$ Discretized Logistic Mixture Colorization",
                 ha="center", fontsize=10, fontstyle="italic", color="#555555")
        
        # Save PNG
        png_out = os.path.join(output_dir, f"comparison_{prod_id}.png")
        plt.savefig(png_out, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
        logger.info(f"Saved comparison PNG to {png_out}")
        
        pdf_pages.append(fig)
        
    # Export to multi-page PDF containing all scenes
    if pdf_pages:
        pdf_out = os.path.join(output_dir, "Sample_Results.pdf")
        with PdfPages(pdf_out) as pdf:
            for fig in pdf_pages:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        logger.info(f"Saved compiled PDF results to {pdf_out}")
        
    # Generate README.md in sample_results directory
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"""# Project VARNA: Sample Results & Visual Enhancements

This directory contains the visual comparison figures demonstrating the performance of the **VARNA (Variational class-Aware Radiance-to-reflectance Network)** pipeline on Landsat-9 imagery.

## Enhancements Flow
The enhanced output demonstrates the transition from low-resolution thermal input to super-resolved structural details and final multi-spectral optical synthesized colors:
1. **Raw TIR Input (200m)**: Digital Temperature Numbers from the TIRS-2 sensor.
2. **Super-Resolved TIR (100m)**: Enhanced thermal spatial features outputted by the residual PixelShuffle SR head.
3. **Synthesized Colorized TIR (100m)**: RGB reflectance combination generated by the discretized mixture colorization decoder.

## Generated Artifacts
- **Sample_Results.pdf**: Compiled, publication-quality vector-embedded document containing visual comparisons for all scenes.
- **comparison_<product_id>.png**: High-resolution raster figure for individual satellite products.
""")
    logger.info(f"Generated descriptive README.md at {readme_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_sample_results()
