import numpy as np

# Standard Landsat 9 Band 10 calibration constants
ML_DEFAULT = 0.0003342  # Radiance multiplicative scaling factor
AL_DEFAULT = 0.1        # Radiance additive scaling factor
K1_DEFAULT = 774.89     # Calibration constant 1
K2_DEFAULT = 1321.07    # Calibration constant 2

def dn_to_radiance(dn, ml=ML_DEFAULT, al=AL_DEFAULT):
    """
    Converts Digital Numbers (DN) to spectral radiance.
    L_lambda = ML * DN + AL
    """
    return ml * dn.astype(np.float32) + al

def radiance_to_brightness_temp(radiance, k1=K1_DEFAULT, k2=K2_DEFAULT):
    """
    Converts spectral radiance to Brightness Temperature (T_B) in Kelvin.
    T = K2 / ln((K1 / L_lambda) + 1)
    """
    # Avoid log of zero or negative radiance
    safe_radiance = np.clip(radiance, 1e-6, None)
    return k2 / np.log((k1 / safe_radiance) + 1.0)

def dn_to_brightness_temp(dn, ml=ML_DEFAULT, al=AL_DEFAULT, k1=K1_DEFAULT, k2=K2_DEFAULT):
    """
    Full Stage 0 conversion from raw DN to Brightness Temperature (Kelvin).
    """
    radiance = dn_to_radiance(dn, ml, al)
    return radiance_to_brightness_temp(radiance, k1, k2)
