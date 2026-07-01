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
    Mandatory Band Ordering (BGR):
    - Layer 1 / Band 1: Blue
    - Layer 2 / Band 2: Green
    - Layer 3 / Band 3: Red
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
        # Write channel 0 (Blue) to Band 1
        dst.write(color_array[0], 1)
        # Write channel 1 (Green) to Band 2
        dst.write(color_array[1], 2)
        # Write channel 2 (Red) to Band 3
        dst.write(color_array[2], 3)
        
    logger.info(f"Successfully saved BGR colorized GeoTIFF to {output_path}")
