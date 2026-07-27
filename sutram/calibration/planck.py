import numpy as np
import torch

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
    if isinstance(dn, torch.Tensor):
        return ml * dn.float() + al
    return ml * dn.astype(np.float32) + al

def radiance_to_brightness_temp(radiance, k1=K1_DEFAULT, k2=K2_DEFAULT):
    """
    Converts spectral radiance to Brightness Temperature (T_B) in Kelvin.
    T = K2 / ln((K1 / L_lambda) + 1)
    """
    if isinstance(radiance, torch.Tensor):
        safe_radiance = torch.clamp(radiance, min=1e-6)
        return k2 / torch.log((k1 / safe_radiance) + 1.0)
    
    safe_radiance = np.clip(radiance, 1e-6, None)
    return k2 / np.log((k1 / safe_radiance) + 1.0)

def dn_to_brightness_temp(dn, ml=ML_DEFAULT, al=AL_DEFAULT, k1=K1_DEFAULT, k2=K2_DEFAULT):
    """
    Full Stage 0 conversion from raw DN to Brightness Temperature (Kelvin).
    """
    radiance = dn_to_radiance(dn, ml, al)
    return radiance_to_brightness_temp(radiance, k1, k2)

def brightness_temp_to_dn(tb, ml=ML_DEFAULT, al=AL_DEFAULT, k1=K1_DEFAULT, k2=K2_DEFAULT):
    """
    Converts Brightness Temperature (Kelvin) back to raw DN.
    DN = ((K1 / (exp(K2 / tb) - 1.0)) - AL) / ML
    """
    if isinstance(tb, torch.Tensor):
        radiance = k1 / (torch.exp(k2 / tb) - 1.0)
        dn = (radiance - al) / ml
        return dn
        
    radiance = k1 / (np.exp(k2 / tb) - 1.0)
    dn = (radiance - al) / ml
    return dn

