import os
import numpy as np
import tifffile

def generate_synthetic_landsat9(output_dir, size=(3416, 3416)):
    """
    Generates a simulated Landsat 9 product with B2, B3, B4, and B10 bands.
    Saves them as TIF files in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    height, width = size

    # Create coordinate grid
    y, x = np.mgrid[0:height, 0:width]
    
    # Initialize bands (using uint16 to resemble Landsat-9 DN values)
    # B2: Blue, B3: Green, B4: Red, B10: Thermal Infrared
    b2 = np.zeros(size, dtype=np.uint16)
    b3 = np.zeros(size, dtype=np.uint16)
    b4 = np.zeros(size, dtype=np.uint16)
    b10 = np.zeros(size, dtype=np.uint16)

    # 1. Base background: Forest / Vegetation (Moderate temperature, Green reflectance)
    # Surface Reflectance values (scaled: usually 0 to 10000 or similar in Landsat 9)
    b2 += 1000  # low blue
    b3 += 2500  # moderate-high green
    b4 += 800   # low red
    b10 += 29300 # moderate temperature (~293 Kelvin scaled by 100)

    # Add some noise/texture to background
    noise = np.random.normal(0, 100, size).astype(np.int32)
    b2 = np.clip(b2.astype(np.int32) + noise, 0, 65535).astype(np.uint16)
    b3 = np.clip(b3.astype(np.int32) + noise * 2, 0, 65535).astype(np.uint16)
    b4 = np.clip(b4.astype(np.int32) + noise, 0, 65535).astype(np.uint16)
    b10 = np.clip(b10.astype(np.int32) + noise * 5, 0, 65535).astype(np.uint16)

    # 2. Add a water body (circle in the center-left: cool temperature, dark blue-green reflectance)
    center_y, center_x = height // 2, width // 3
    radius = min(height, width) // 6
    water_mask = ((y - center_y)**2 + (x - center_x)**2) < radius**2

    b2[water_mask] = 1200  # slightly higher blue reflectance relative to others
    b3[water_mask] = 1500  # green
    b4[water_mask] = 400   # very low red (water absorbs red)
    b10[water_mask] = 28500 # cool temperature (~285 Kelvin)

    # 3. Add an urban area (rectangular grid in the top-right: hot temperature, grey/brick reflectance)
    urban_y_start, urban_y_end = height // 10, height // 3
    urban_x_start, urban_x_end = (width * 6) // 10, (width * 9) // 10
    urban_mask = (y >= urban_y_start) & (y < urban_y_end) & (x >= urban_x_start) & (x < urban_x_end)

    b2[urban_mask] = 3000  # grey has high reflectance across visible spectrum
    b3[urban_mask] = 3100
    b4[urban_mask] = 3500  # brick/concrete can be slightly reddish
    b10[urban_mask] = 31000 # warm/hot temperature (~310 Kelvin)

    # Add a grid/street effect to urban area
    street_mask = urban_mask & (((y % 120) < 20) | ((x % 120) < 20))
    b2[street_mask] = 1500  # asphalt / dark streets
    b3[street_mask] = 1500
    b4[street_mask] = 1500
    b10[street_mask] = 31500 # asphalt gets extremely hot

    # 4. Add a desert / bare soil region (bottom-right: warm temperature, sandy/brown reflectance)
    desert_mask = (y > (height * 6) // 10) & (x > (width * 5) // 10)
    b2[desert_mask] = 2500  # high visible reflectance
    b3[desert_mask] = 3500
    b4[desert_mask] = 4800  # reddish-brown sand
    b10[desert_mask] = 30500 # warm (~305 Kelvin)

    # Save the bands as TIFF files
    prefix = "LC09_L2SP_146044_20260701_20260701_02_T1"
    tifffile.imwrite(os.path.join(output_dir, f"{prefix}_B2.TIF"), b2)
    tifffile.imwrite(os.path.join(output_dir, f"{prefix}_B3.TIF"), b3)
    tifffile.imwrite(os.path.join(output_dir, f"{prefix}_B4.TIF"), b4)
    tifffile.imwrite(os.path.join(output_dir, f"{prefix}_B10.TIF"), b10)

    print(f"Generated synthetic Landsat 9 product in {output_dir}")
    print(f"Generated bands size: {size}")

if __name__ == "__main__":
    generate_synthetic_landsat9("input/LC09_L2SP_146044_20260701_20260701_02_T1")
