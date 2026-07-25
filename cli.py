import os
import sys

# Insert root directory into sys.path to allow running python commands without setting PYTHONPATH
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import argparse
import logging
import glob
import numpy as np
import torch
from omegaconf import OmegaConf

from training.trainer import UnifiedTrainer
from training.backbone import ResNetBackbone
from training.sr_head import SRHead
from training.mixture_head import MixtureHead
from inference.pipeline import SUTRAMInferencePipeline
from training.utils.config_schema import validate_sutram_config
from training.utils.logger import setup_sutram_logger
from data_pipeline.pipeline_state import PipelineState

logger = logging.getLogger("sutram.cli")

def validate_release_checkpoint(checkpoint, checkpoint_path):
    """Refuse stale or ambiguous artifacts at SUTRAM inference time."""
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Release checkpoint is not a metadata package: {checkpoint_path}")
    model_name = str(checkpoint.get("model_name", ""))
    if checkpoint.get("project_id") != "SUTRAM" or "SUTRAM" not in model_name.upper():
        raise ValueError(
            f"Refusing non-SUTRAM or legacy checkpoint: {checkpoint_path}. "
            "Create a fresh SUTRAM release after retraining."
        )
    preprocessing = checkpoint.get("config", {}).get("preprocessing", {})
    if preprocessing.get("tir_representation") != "brightness_temperature_kelvin":
        raise ValueError(
            f"Release checkpoint lacks calibrated-B10 metadata: {checkpoint_path}. "
            "Create a fresh SUTRAM release after retraining."
        )

def main():
    parser = argparse.ArgumentParser(description="SUTRAM Unified Workflow Command Line Interface")
    parser.add_argument("command", choices=["train-stage1", "train-stage2", "validate-dataset", "evaluate", "benchmark", "export", "submit", "generate-sample-results", "infer"],
                        help="Workflow command to execute")
    parser.add_argument("--config", default="configs/base_config.yaml",
                        help="Path to Hydra base configuration file")
    parser.add_argument("--weights", default=None,
                        help="Path to the packaged release weights file (.pth)")
    parser.add_argument("--input", default=None,
                        help="Path to input directory or Landsat-9 product directory")
    parser.add_argument("--force", action="store_true",
                        help="Force execution (e.g. force training even if checkpoints exist)")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="The dataset split to evaluate (val or test)")
    args = parser.parse_args()

    # Load configuration
    try:
        data_cfg = OmegaConf.load("configs/data.yaml")
        training_cfg = OmegaConf.load("configs/training.yaml")
        eval_cfg = OmegaConf.load("configs/evaluation.yaml")
        inf_cfg = OmegaConf.load("configs/inference.yaml")
        base_cfg = OmegaConf.load(args.config)
        
        cfg = OmegaConf.merge(base_cfg, OmegaConf.create({"data": data_cfg, "training": training_cfg, "evaluation": eval_cfg, "inference": inf_cfg}))
        
        setup_sutram_logger(cfg.data.output_dir)
        validate_sutram_config(cfg)
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    logger.info(f"Executing SUTRAM CLI command: {args.command}")

    state = PipelineState()

    if args.command == "train-stage1":
        checkpoint_dir = os.path.join("experiments", cfg.experiment_id, "checkpoints")
        
        if not args.force and state.is_stage1_valid(checkpoint_dir):
            logger.info("Stage 1 checkpoints (backbone & sr_head) already exist and pipeline state is valid. Skipping Stage 1 training. Use --force to retrain.")
        else:
            # Pre-training validation gate: fail fast on P0 integrity errors
            from data_pipeline.split_validator import run_all_validation
            run_all_validation(
                OmegaConf.to_container(cfg.data.splits, resolve=True),
                cfg.data.patches_dir,
                input_dir=cfg.data.input_dir
            )
            trainer = UnifiedTrainer(cfg)
            trainer.train_stage1_sr()
        
    elif args.command == "train-stage2":
        checkpoint_dir = os.path.join("experiments", cfg.experiment_id, "checkpoints")
        
        if not args.force and state.is_stage2_valid(checkpoint_dir):
            logger.info("Stage 2 checkpoint (mixture_head) already exists and pipeline state is valid. Skipping Stage 2 training. Use --force to retrain.")
        else:
            # Pre-training validation gate: fail fast on P0 integrity errors
            from data_pipeline.split_validator import run_all_validation
            run_all_validation(
                OmegaConf.to_container(cfg.data.splits, resolve=True),
                cfg.data.patches_dir,
                input_dir=cfg.data.input_dir
            )
            trainer = UnifiedTrainer(cfg)
            trainer.train_stage2_color()
            
    elif args.command == "validate-dataset":
        logger.info("Executing dataset validation gate...")
        from data_pipeline.split_validator import run_all_validation
        try:
            run_all_validation(
                OmegaConf.to_container(cfg.data.splits, resolve=True),
                cfg.data.patches_dir,
                input_dir=cfg.data.input_dir
            )
            logger.info("Dataset validation gate: SUCCESS! All pre-training integrity checks passed.")
        except Exception as e:
            logger.error(f"Dataset validation gate: FAILED! {e}")
            sys.exit(1)
        
    elif args.command == "evaluate":
        logger.info(f"Evaluation stage is running on split: {args.split}...")
        from evaluation.report import run_evaluation_report
        run_evaluation_report(args.config, args.weights, split=args.split)
        
    elif args.command == "benchmark":
        logger.info("Benchmarking execution latency and parameter counts...")
        import time
        
        backbone = ResNetBackbone()
        sr_head = SRHead()
        mix_head = MixtureHead(K=cfg.training.stage2.K)
        
        bb_params = sum(p.numel() for p in backbone.parameters())
        sr_params = sum(p.numel() for p in sr_head.parameters())
        mix_params = sum(p.numel() for p in mix_head.parameters())
        total_params = bb_params + sr_params + mix_params
        
        print(f"Backbone parameters: {bb_params:,}")
        print(f"SR Head parameters: {sr_params:,}")
        print(f"Mixture Head parameters: {mix_params:,}")
        print(f"Total Model parameters: {total_params:,}")
        
        pipeline = SUTRAMInferencePipeline(backbone, sr_head, mix_head, K=cfg.training.stage2.K)
        pipeline.eval()
        
        dummy_input = torch.randn(1, 1, 256, 256)
        
        for _ in range(5):
            with torch.no_grad():
                _ = pipeline(dummy_input)
                
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
        import tifffile
        from inference.geotiff_export import export_sr_geotiff, export_colorized_geotiff

        checkpoint_dir = f"experiments/{cfg.experiment_id}/checkpoints"
        backbone = ResNetBackbone()
        sr_head = SRHead()
        mix_head = MixtureHead(K=cfg.training.stage2.K)

        bb_path = os.path.join(checkpoint_dir, "backbone_stage1.pth")
        sr_path = os.path.join(checkpoint_dir, "sr_head_stage1.pth")
        mix_path = os.path.join(checkpoint_dir, "mixture_head_stage2.pth")

        if not (os.path.exists(bb_path) and os.path.exists(sr_path) and os.path.exists(mix_path)):
            logger.error("Missing required stage checkpoints for export.")
            sys.exit(1)

        backbone.load_state_dict(torch.load(bb_path, map_location="cpu"))
        sr_head.load_state_dict(torch.load(sr_path, map_location="cpu"))
        mix_head.load_state_dict(torch.load(mix_path, map_location="cpu"))

        pipeline = SUTRAMInferencePipeline(backbone, sr_head, mix_head, K=cfg.training.stage2.K)
        pipeline.eval()

        os.makedirs("output/onnx", exist_ok=True)
        dummy_in = torch.randn(1, 1, 256, 256)
        torch.onnx.export(
            pipeline, dummy_in, "output/onnx/sutram_pipeline.onnx",
            input_names=["input_tir_200m"],
            output_names=["sr_tir_100m", "color_rgb_100m"],
            dynamic_axes={"input_tir_200m": {2: "height", 3: "width"}}
        )
        logger.info("Successfully exported ONNX model to output/onnx/sutram_pipeline.onnx")

    elif args.command == "submit":
        logger.info("Packaging submission deliverables...")
        from scripts.generate_sample_patches_zip import main as gen_zip
        gen_zip()

if __name__ == "__main__":
    main()
