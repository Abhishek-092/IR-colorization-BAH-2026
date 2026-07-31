import os
import numpy as np
import rasterio
import pytest
from inference.geotiff_export import export_colorized_geotiff

def test_bgr_band_ordering(tmp_path):
    """
    Asserts that BGR band ordering is correctly written to the output file.
    Channel 0 (Blue) -> Band 1
    Channel 1 (Green) -> Band 2
    Channel 2 (Red) -> Band 3
    """
    # Create reference GeoTIFF
    ref_path = os.path.join(tmp_path, "ref.tif")
    out_path = os.path.join(tmp_path, "out.tif")
    
    # 2x2 dummy single band
    dummy_ref = np.ones((2, 2), dtype=np.uint8) * 100
    
    # Create a basic profile
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': None,
        'width': 2,
        'height': 2,
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': rasterio.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    }
    
    with rasterio.open(ref_path, 'w', **profile) as dst:
        dst.write(dummy_ref, 1)

    # Prepare distinct channel values: Blue=30, Green=20, Red=10
    # Shape: (3, H, W)
    # H_new = H*2 = 4, W_new = W*2 = 4
    color_arr = np.zeros((3, 4, 4), dtype=np.uint8)
    color_arr[0, ...] = 30 # Channel 0: Blue
    color_arr[1, ...] = 20 # Channel 1: Green
    color_arr[2, ...] = 10 # Channel 2: Red

    # Export colorized BGR
    export_colorized_geotiff(color_arr, ref_path, out_path)

    # Read back and assert values
    with rasterio.open(out_path) as src:
        assert src.count == 3
        # Band 1 -> Blue (should be 30)
        assert np.all(src.read(1) == 30)
        # Band 2 -> Green (should be 20)
        assert np.all(src.read(2) == 20)
        # Band 3 -> Red (should be 10)
        assert np.all(src.read(3) == 10)

    print("BGR band ordering assertion: PASSED.")
