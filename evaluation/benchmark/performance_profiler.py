import time
import torch
import logging

logger = logging.getLogger(__name__)

def profile_model_performance(model, input_shape=(1, 1, 256, 256), device="cuda", warmup_runs=10, runs=100):
    """
    Profiles inference latency, memory usage, and throughput.
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device_obj)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=device_obj)
    
    # Warmup
    logger.info("Starting GPU warmup runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            
    if device_obj.type == "cuda":
        torch.cuda.synchronize()

    # Latency benchmarking
    logger.info(f"Running {runs} benchmarking iterations...")
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)

    if device_obj.type == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency_ms = (total_time / runs) * 1000
    throughput = runs / total_time
    
    # Memory profiling
    vram_used = 0
    if device_obj.type == "cuda":
        vram_used = torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2) # MB

    profile_results = {
        "avg_latency_ms": avg_latency_ms,
        "throughput_tiles_sec": throughput,
        "max_vram_mb": vram_used
    }
    
    logger.info(f"Inference latency: {avg_latency_ms:.2f} ms/tile")
    logger.info(f"Throughput: {throughput:.2f} tiles/sec")
    logger.info(f"VRAM Footprint: {vram_used:.2f} MB")
    
    return profile_results
