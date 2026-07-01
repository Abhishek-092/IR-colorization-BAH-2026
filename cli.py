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
        run_evaluation_report(args.config)
        
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
