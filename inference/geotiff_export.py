import os
import logging
import rasterio

logger = logging.getLogger(__name__)

def export_sr_geotiff(sr_array, reference_tif_path, output_path):
    """
    Saves the single-band super-resolved TIR array as a georeferenced GeoTIFF.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with rasterio.open(reference_tif_path) as ref:
        profile = ref.profile.copy()
        # Update profile for 2x upscaled spatial resolution
        # Double the width and height
        new_height = ref.height * 2
        new_width = ref.width * 2
        
        # Adjust affine transform scale (divide dx and dy by 2)
        transform = ref.transform
        new_transform = rasterio.Affine(
            transform.a / 2.0, transform.b, transform.c,
            transform.d, transform.e / 2.0, transform.f
        )
        
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform,
            'count': 1,
            'dtype': str(sr_array.dtype)
        })

    with rasterio.open(output_path, 'w', **profile) as dst:
        # Write to band 1
        dst.write(sr_array[0] if sr_array.ndim == 3 else sr_array, 1)
        
    logger.info(f"Successfully saved super-resolved GeoTIFF to {output_path}")

def export_colorized_geotiff(color_array, reference_tif_path, output_path):
    """
    Saves a 3-channel colorized array as a georeferenced GeoTIFF.

    INPUT is the model's RGB output — channel 0 = Red, 1 = Green, 2 = Blue
    (the mixture head is trained on targets stacked as [B4=R, B3=G, B2=B] in
    data_pipeline/prepare_dataset.py, so its decoded colour is RGB).

    OUTPUT file uses the mandatory submission Band Ordering (BGR):
    - Band 1: Blue   (<- input channel 2)
    - Band 2: Green  (<- input channel 1)
    - Band 3: Red    (<- input channel 0)

    The RGB->BGR reorder happens HERE so the rest of the pipeline (training,
    dashboard) can stay in natural RGB while the deliverable is spec-correct.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if color_array.ndim != 3 or color_array.shape[0] != 3:
        raise ValueError(f"Colorized array must be of shape (3, H, W). Got: {color_array.shape}")
        
    with rasterio.open(reference_tif_path) as ref:
        profile = ref.profile.copy()
        # Update profile for 2x upscaled spatial resolution
        new_height = ref.height * 2
        new_width = ref.width * 2
        
        transform = ref.transform
        new_transform = rasterio.Affine(
            transform.a / 2.0, transform.b, transform.c,
            transform.d, transform.e / 2.0, transform.f
        )
        
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform,
            'count': 3,
            'dtype': str(color_array.dtype),
            'photometric': 'rgb'  # tells viewers it's a multi-band image
        })

    with rasterio.open(output_path, 'w', **profile) as dst:
        # Reorder RGB -> BGR on write (submission spec):
        dst.write(color_array[2], 1)   # Blue  (input ch 2) -> Band 1
        dst.write(color_array[1], 2)   # Green (input ch 1) -> Band 2
        dst.write(color_array[0], 3)   # Red   (input ch 0) -> Band 3

    logger.info(f"Successfully saved BGR colorized GeoTIFF to {output_path}")
