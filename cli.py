import sys
import argparse
import logging
from omegaconf import OmegaConf

from training.trainer import UnifiedTrainer
from training.utils.config_schema import validate_varna_config
from training.utils.logger import setup_varna_logger

logger = logging.getLogger("varna.cli")

def main():
    parser = argparse.ArgumentParser(description="VARNA Unified Workflow Command Line Interface")
    parser.add_argument("command", choices=["train-stage1", "train-stage2", "evaluate", "benchmark", "export", "submit"],
                        help="Workflow command to execute")
    parser.add_argument("--config", default="configs/base_config.yaml",
                        help="Path to Hydra base configuration file")
    args = parser.parse_args()

    # Load configuration
    try:
        cfg = OmegaConf.load(args.config)
        # Manually compose defaults since we are running without Hydra's CLI wrapper for speed
        if "defaults" in cfg:
            for default_cfg in cfg.defaults:
                if isinstance(default_cfg, str) and default_cfg != "_self_":
                    child_path = f"configs/{default_cfg}.yaml"
                    child_cfg = OmegaConf.load(child_path)
                    cfg = OmegaConf.merge(child_cfg, cfg)
        
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
        # Implement evaluation report trigger
        from evaluation.metrics import compute_psnr
        logger.info("Metrics calculation loaded.")
        
    elif args.command == "benchmark":
        logger.info("Benchmarking execution latency...")
        
    elif args.command == "export":
        logger.info("Exporting models to ONNX...")
        
    elif args.command == "submit":
        logger.info("Validating deliverables and generating submission package...")
        from submission.generate_submission import package_submission
        package_submission()

if __name__ == "__main__":
    main()
