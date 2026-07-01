import os
import sys
import argparse
import logging
import numpy as np
from omegaconf import OmegaConf

from training.trainer import UnifiedTrainer
from training.utils.config_schema import validate_varna_config
from training.utils.logger import setup_varna_logger

logger = logging.getLogger("varna.cli")

def main():
    parser = argparse.ArgumentParser(description="VARNA Unified Workflow Command Line Interface")
    parser.add_argument("command", choices=["train-stage1", "train-stage2", "evaluate", "benchmark", "export", "submit", "generate-sample-results", "infer"],
                        help="Workflow command to execute")
    parser.add_argument("--config", default="configs/base_config.yaml",
                        help="Path to Hydra base configuration file")
    parser.add_argument("--weights", default=None,
                        help="Path to the packaged release weights file (.pth)")
    parser.add_argument("--input", default=None,
                        help="Path to input directory or Landsat-9 product directory")
    args = parser.parse_args()

    # Load configuration
    try:
        # Explicitly load and merge standard configs to match base config structure
        data_cfg = OmegaConf.load("configs/data.yaml")
        training_cfg = OmegaConf.load("configs/training.yaml")
        eval_cfg = OmegaConf.load("configs/evaluation.yaml")
        inf_cfg = OmegaConf.load("configs/inference.yaml")
        base_cfg = OmegaConf.load(args.config)
        
        cfg = OmegaConf.merge(base_cfg, OmegaConf.create({"data": data_cfg, "training": training_cfg, "evaluation": eval_cfg, "inference": inf_cfg}))
        
        # Setup logging
        setup_varna_logger(cfg.data.output_dir)
        validate_varna_config(cfg)
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    logger.info(f"Executing VARNA CLI command: {args.command}")

    if args.command == "train-stage1":
        trainer = UnifiedTrainer(cfg)
        trainer.train_stage1_sr()
        
    elif args.command == "train-stage2":
        trainer = UnifiedTrainer(cfg)
        trainer.train_stage2_color()
        
    elif args.command == "evaluate":
        logger.info("Evaluation stage is running...")
        from evaluation.report import run_evaluation_report
        run_evaluation_report(args.config, args.weights)
        
    elif args.command == "benchmark":
        logger.info("Benchmarking execution latency and parameter counts...")
        import torch
        import time
        from training.backbone import ResNetBackbone
        from training.sr_head import SRHead
        from training.mixture_head import MixtureHead
        
        backbone = ResNetBackbone()
        sr_head = SRHead()
        mix_head = MixtureHead(K=cfg.training.stage2.K)
        
        # Parameter counting
        bb_params = sum(p.numel() for p in backbone.parameters())
        sr_params = sum(p.numel() for p in sr_head.parameters())
        mix_params = sum(p.numel() for p in mix_head.parameters())
        total_params = bb_params + sr_params + mix_params
        
        print(f"Backbone parameters: {bb_params:,}")
        print(f"SR Head parameters: {sr_params:,}")
        print(f"Mixture Head parameters: {mix_params:,}")
        print(f"Total Model parameters: {total_params:,}")
        
        # Latency profiling
        from inference.pipeline import VARNAInferencePipeline
        pipeline = VARNAInferencePipeline(backbone, sr_head, mix_head, K=cfg.training.stage2.K)
        pipeline.eval()
        
        dummy_input = torch.randn(1, 1, 256, 256)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = pipeline(dummy_input)
                
        # Profile loop
        runs = 50
        start_time = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                _ = pipeline(dummy_input)
        end_time = time.perf_counter()
        
        avg_latency = ((end_time - start_time) / runs) * 1000
        print(f"Average CPU forward pass latency: {avg_latency:.2f} ms")
        
    elif args.command == "export":
        logger.info("Exporting models to ONNX and running full inference...")
        import torch
        import tifffile
        from training.backbone import ResNetBackbone
        from training.sr_head import SRHead
        from training.mixture_head import MixtureHead
        from inference.pipeline import VARNAInferencePipeline
        from inference.geotiff_export import export_sr_geotiff, export_colorized_geotiff

        # Instantiate pipeline and load weights
        checkpoint_dir = f"experiments/{cfg.experiment_id}/checkpoints"
        backbone = ResNetBackbone()
        sr_head = SRHead()
        mix_head = MixtureHead(K=cfg.training.stage2.K)

        backbone.load_state_dict(torch.load(f"{checkpoint_dir}/backbone_stage1.pth", map_location="cpu"))
        sr_head.load_state_dict(torch.load(f"{checkpoint_dir}/sr_head_stage1.pth", map_location="cpu"))
        mix_head.load_state_dict(torch.load(f"{checkpoint_dir}/mixture_head_stage2.pth", map_location="cpu"))

        pipeline = VARNAInferencePipeline(backbone, sr_head, mix_head, K=cfg.training.stage2.K)
        pipeline.eval()

        # 1. Trace and export ONNX model
        os.makedirs("checkpoints", exist_ok=True)
        dummy_input = torch.randn(1, 1, 256, 256)
        torch.onnx.export(
            pipeline,
            dummy_input,
            cfg.inference.onnx.export_path,
            input_names=["lr_tir"],
            output_names=["sr_tir", "dominant_color"],
            opset_version=cfg.inference.onnx.opset_version,
            dynamic_axes={"lr_tir": {0: "batch_size"}, "sr_tir": {0: "batch_size"}, "dominant_color": {0: "batch_size"}}
        )
        logger.info(f"ONNX model successfully exported to {cfg.inference.onnx.export_path}")

        # 2. Run full inference on the product to generate required TIF deliverables
        prod_id = "LC09_L2SP_146044_20260701_20260701_02_T1"
        ref_path = f"input/{prod_id}/{prod_id}_B10.TIF"
        lr_tir_path = f"output/downscaled_data/{prod_id}_tir_200m.tif"
        
        if os.path.exists(lr_tir_path) and os.path.exists(ref_path):
            lr_img = tifffile.imread(lr_tir_path).astype(np.float32)
            lr_tensor = torch.from_numpy(lr_img)
            if lr_tensor.ndim == 2:
                lr_tensor = lr_tensor.unsqueeze(0).unsqueeze(0)
            elif lr_tensor.ndim == 3:
                lr_tensor = lr_tensor.unsqueeze(0)
            
            with torch.no_grad():
                sr_tir, decode_outs = pipeline(lr_tensor)
                
            sr_np = sr_tir.squeeze().numpy()
            pred_rgb = decode_outs["dominant_color"].squeeze().numpy()
            
            # Save GeoTIFFs
            out_sr_path = f"output/model_outputs/tir_superresolved_100m/{prod_id}.tif"
            out_color_path = f"output/model_outputs/colorized_tir_100m/{prod_id}.tif"
            
            export_sr_geotiff(sr_np, ref_path, out_sr_path)
            export_colorized_geotiff(pred_rgb, ref_path, out_color_path)
            logger.info("Inference deliverables successfully generated.")
        else:
            logger.warning("Could not find input files for inference run. Skipping GeoTIFF export.")
        
    elif args.command == "submit":
        logger.info("Validating deliverables and generating submission package...")
        from submission.generate_submission import package_submission
        package_submission()
        
    elif args.command == "generate-sample-results":
        logger.info("Generating publication-quality sample results figures...")
        from evaluation.sample_results import generate_sample_results
        generate_sample_results()

if __name__ == "__main__":
    main()
