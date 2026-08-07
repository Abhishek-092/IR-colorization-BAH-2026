import os
import numpy as np
import rasterio
import pytest
from inference.geotiff_export import export_colorized_geotiff

def test_bgr_band_ordering(tmp_path):
    """
    Semantic check: the export takes the model's RGB array (ch0=Red, ch1=Green,
    ch2=Blue) and must write the mandatory BGR file order:
        input Red   (ch0) -> Band 3
        input Green (ch1) -> Band 2
        input Blue  (ch2) -> Band 1
    A purely positional test would pass even with R/B swapped, so we assert the
    actual colour lands in the correct band.
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

    # Model RGB input: Red=10, Green=20, Blue=30 (ch0=R, ch1=G, ch2=B)
    # Shape: (3, H, W); H_new = H*2 = 4, W_new = W*2 = 4
    color_arr = np.zeros((3, 4, 4), dtype=np.uint8)
    color_arr[0, ...] = 10  # Channel 0: Red
    color_arr[1, ...] = 20  # Channel 1: Green
    color_arr[2, ...] = 30  # Channel 2: Blue

    # Export to the mandatory BGR file order
    export_colorized_geotiff(color_arr, ref_path, out_path)

    # Read back and assert the colour landed in the correct band.
    with rasterio.open(out_path) as src:
        assert src.count == 3
        assert np.all(src.read(1) == 30)  # Band 1 -> Blue  (input ch2)
        assert np.all(src.read(2) == 20)  # Band 2 -> Green (input ch1)
        assert np.all(src.read(3) == 10)  # Band 3 -> Red   (input ch0)

    print("BGR band ordering assertion: PASSED.")
