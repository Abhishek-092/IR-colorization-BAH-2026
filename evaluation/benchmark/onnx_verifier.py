import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

# Lazy import onnxruntime to avoid startup issues if not configured
def get_ort_session(onnx_path):
    import onnxruntime as ort
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

def verify_onnx_numerical_parity(pytorch_model, onnx_path, input_shape=(1, 1, 256, 256), rtol=1e-3, atol=1e-3):
    """
    Compares outputs of PyTorch and ONNX models on matching random tensors.
    """
    pytorch_model.eval()
    dummy_input = torch.randn(*input_shape)
    
    # Run PyTorch
    with torch.no_grad():
        pytorch_outs = pytorch_model(dummy_input)
        if isinstance(pytorch_outs, tuple):
            pytorch_np = [out.numpy() for out in pytorch_outs]
        elif isinstance(pytorch_outs, dict):
            pytorch_np = {k: v.numpy() for k, v in pytorch_outs.items()}
        else:
            pytorch_np = [pytorch_outs.numpy()]

    # Run ONNX Runtime
    ort_session = get_ort_session(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    
    ort_inputs = {input_name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # Validate output arrays
    for idx, (py_val, ort_val) in enumerate(zip(pytorch_np, ort_outs)):
        diff = np.abs(py_val - ort_val)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        logger.info(f"Output {idx} - Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
        
        try:
            np.testing.assert_allclose(py_val, ort_val, rtol=rtol, atol=atol)
        except AssertionError as e:
            logger.error(f"ONNX numerical parity check FAILED for output {idx}: {e}")
            return False

    logger.info("ONNX numerical parity verification: PASSED.")
    return True
