import numpy as np
import pytest
from sutram.calibration.planck import (
    dn_to_radiance,
    radiance_to_brightness_temp,
    dn_to_brightness_temp,
    K1_DEFAULT,
    K2_DEFAULT
)

def test_planck_calibration_constants():
    """Verify default Landsat-9 physical parameters."""
    assert K1_DEFAULT == 774.89
    assert K2_DEFAULT == 1321.07

def test_planck_radiance_calculation():
    """Verify DN to radiance mapping."""
    dn = np.array([0, 10000, 20000], dtype=np.uint16)
    rad = dn_to_radiance(dn)
    
    # L_lambda = 0.0003342 * DN + 0.1
    expected = 0.0003342 * dn + 0.1
    np.testing.assert_allclose(rad, expected, rtol=1e-5)

def test_planck_brightness_temp_mapping():
    """Verify radiance to Kelvin mapping."""
    # Test a typical radiance value
    rad = np.array([8.0], dtype=np.float32)
    t_b = radiance_to_brightness_temp(rad)
    
    expected = 1321.07 / np.log((774.89 / 8.0) + 1.0)
    np.testing.assert_allclose(t_b, expected, rtol=1e-5)

def test_end_to_end_dn_to_kelvin():
    """Verify combined DN to Kelvin mapping does not output extreme outliers."""
    dn = np.array([30000], dtype=np.uint16)
    t_b = dn_to_brightness_temp(dn)
    
    # 30000 -> rad ~ 10.126 -> Kelvin ~ 296 Kelvin (approx room temp)
    assert 280.0 < t_b[0] < 310.0

def test_planck_roundtrip():
    """Verify DN -> Brightness Temp -> DN roundtrip parity for numpy and torch."""
    import torch
    from sutram.calibration.planck import brightness_temp_to_dn
    
    # Numpy roundtrip
    dns_np = np.array([20000, 25000, 30000, 35000], dtype=np.float32)
    tb_np = dn_to_brightness_temp(dns_np)
    dns_rec_np = brightness_temp_to_dn(tb_np)
    np.testing.assert_allclose(dns_np, dns_rec_np, rtol=1e-4)
    
    # Torch roundtrip
    dns_th = torch.tensor([20000.0, 25000.0, 30000.0, 35000.0])
    tb_th = dn_to_brightness_temp(dns_th)
    dns_rec_th = brightness_temp_to_dn(tb_th)
    torch.testing.assert_close(dns_th, dns_rec_th, rtol=1e-4, atol=1e-4)

