import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from training.backbone import ResNetBackbone
from training.sr_head import SRHead
from training.mixture_head import MixtureHead
from training.loss_functions import (
    DegradationConsistencyLoss,
    EdgeGradientLoss,
    DiscretizedLogisticMixtureNLLLoss
)
from data_pipeline.dataset_loader import PatchDataset
from data_pipeline.dataset_report import generate_dataset_report
from data_pipeline.pipeline_state import PipelineState
from training.utils.seed import set_seed

logger = logging.getLogger(__name__)

class UnifiedTrainer:
    """
    Unified training framework for Phase 1 (SR) and Phase 2 (Color Mixture).
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        set_seed(cfg.seed)

        # Setup paths
        self.exp_dir = os.path.join("experiments", cfg.experiment_id)
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Validate splits to prevent leakage and enforce strict criteria
        from data_pipeline.split_validator import run_all_validation
        import omegaconf
        splits_dict = omegaconf.OmegaConf.to_container(cfg.data.splits, resolve=True)
        run_all_validation(splits_dict, cfg.data.patches_dir, input_dir=cfg.data.input_dir)

        # Init models
        self.backbone = ResNetBackbone(in_channels=1).to(self.device)
        self.sr_head = SRHead().to(self.device)
        self.mixture_head = MixtureHead(K=cfg.training.stage2.K).to(self.device)

        # Setup dataset and loader with async multi-worker prefetching
        self.train_dataset = PatchDataset(
            patches_dir=cfg.data.patches_dir,
            product_ids=cfg.data.splits.train,
            augment=True
        )
        self.val_dataset = PatchDataset(
            patches_dir=cfg.data.patches_dir,
            product_ids=cfg.data.splits.val
        )
        
        num_workers = min(4, os.cpu_count() or 1)
        pin_memory = torch.cuda.is_available()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.training.stage1.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.training.stage1.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        logger.info(f"Loaded train samples: {len(self.train_dataset)}, val samples: {len(self.val_dataset)} (num_workers={num_workers}, pin_memory={pin_memory})")

    def train_stage1_sr(self):
        """Trains the backbone and SR head deterministically."""
        logger.info("--- Starting Stage 1: Super-Resolution Training ---")
        
        bb_path = os.path.join(self.checkpoint_dir, "backbone_stage1.pth")
        sr_path = os.path.join(self.checkpoint_dir, "sr_head_stage1.pth")
        if os.path.exists(bb_path):
            self.backbone.load_state_dict(torch.load(bb_path, map_location=self.device))
            logger.info("Loaded existing Stage 1 backbone weights successfully.")
        if os.path.exists(sr_path):
            self.sr_head.load_state_dict(torch.load(sr_path, map_location=self.device))
            logger.info("Loaded existing Stage 1 SR head weights successfully.")
        optimizer = optim.AdamW(
            list(self.backbone.parameters()) + list(self.sr_head.parameters()),
            lr=self.cfg.training.stage1.lr,
            weight_decay=self.cfg.training.stage1.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.training.stage1.epochs
        )

        # Loss functions
        l1_criterion = nn_L1Loss()
        degradation_criterion = DegradationConsistencyLoss().to(self.device)
        edge_criterion = EdgeGradientLoss().to(self.device)

        best_psnr = -float("inf")
        epochs = self.cfg.training.stage1.epochs

        for epoch in range(1, epochs + 1):
            self.backbone.train()
            self.sr_head.train()
            train_loss = 0.0

            for batch in self.train_loader:
                lr_tir = batch["tir_200m"].to(self.device)
                hr_tir = batch["tir_100m_512"].to(self.device)

                optimizer.zero_grad()
                features = self.backbone(lr_tir)
                pred_hr = self.sr_head(features, lr_tir)

                # Compute loss
                loss_l1 = l1_criterion(pred_hr, hr_tir)
                loss_deg = degradation_criterion(pred_hr, lr_tir)
                loss_edge = edge_criterion(pred_hr, hr_tir)

                w = self.cfg.training.stage1.loss_weights
                loss = w.l1 * loss_l1 + w.degradation * loss_deg + w.edge * loss_edge

                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.backbone.parameters()) + list(self.sr_head.parameters()), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()
            train_loss /= len(self.train_loader)

            # Validation
            val_psnr = self._validate_sr(l1_criterion)
            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val PSNR: {val_psnr:.2f} dB")

            # Checkpoint save
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                bb_temp = os.path.join(self.checkpoint_dir, "backbone_stage1.pth.tmp")
                sr_temp = os.path.join(self.checkpoint_dir, "sr_head_stage1.pth.tmp")
                torch.save(self.backbone.state_dict(), bb_temp)
                torch.save(self.sr_head.state_dict(), sr_temp)
                os.replace(bb_temp, os.path.join(self.checkpoint_dir, "backbone_stage1.pth"))
                os.replace(sr_temp, os.path.join(self.checkpoint_dir, "sr_head_stage1.pth"))
                logger.info(f"New best validation PSNR locked: {best_psnr:.2f} dB")

        # Record stage 1 completion in central state
        PipelineState().record_stage1_completion(self.checkpoint_dir)
        logger.info("Stage 1 completion recorded in pipeline_state.json")

    def _validate_sr(self, l1_criterion):
        self.backbone.eval()
        self.sr_head.eval()
        total_psnr = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                lr_tir = batch["tir_200m"].to(self.device)
                hr_tir = batch["tir_100m_512"].to(self.device)

                features = self.backbone(lr_tir)
                pred_hr = self.sr_head(features, lr_tir)

                mse = F.mse_loss(pred_hr, hr_tir)
                if mse > 0:
                    psnr = 10 * torch.log10(1.0 / mse)
                    total_psnr += psnr.item()
                count += 1

        return total_psnr / max(1, count)

    def train_stage2_color(self):
        """Trains the mixture head while adapting the shared backbone."""
        logger.info("--- Starting Stage 2: Mixture Colorization Training ---")
        
        # Load backbone weights from Stage 1
        bb_path = os.path.join(self.checkpoint_dir, "backbone_stage1.pth")
        if os.path.exists(bb_path):
            self.backbone.load_state_dict(torch.load(bb_path, map_location=self.device))
            logger.info("Loaded Stage 1 backbone weights successfully.")

        mix_path = os.path.join(self.checkpoint_dir, "mixture_head_stage2.pth")
        loaded_mix = False
        if os.path.exists(mix_path):
            self.mixture_head.load_state_dict(torch.load(mix_path, map_location=self.device))
            logger.info("Loaded existing Stage 2 mixture head weights successfully.")
            loaded_mix = True

        # Unfreeze Backbone to allow features to adapt to color targets; keep SR Head frozen
        for p in self.backbone.parameters():
            p.requires_grad = True
        for p in self.sr_head.parameters():
            p.requires_grad = False

        # Quantile-based component-mean initialization
        if not loaded_mix:
            self._init_mixture_means_with_quantiles()

        optimizer = optim.AdamW(
            list(self.mixture_head.parameters()) + list(self.backbone.parameters()),
            lr=self.cfg.training.stage2.lr,
            weight_decay=self.cfg.training.stage2.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.training.stage2.epochs
        )

        nll_criterion = DiscretizedLogisticMixtureNLLLoss(
            epsilon=self.cfg.training.stage2.epsilon
        )

        epochs = self.cfg.training.stage2.epochs
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            self.backbone.train()
            self.mixture_head.train()
            self.sr_head.eval()
            train_loss = 0.0

            temp_init = self.cfg.training.stage2.temp_init
            temp_final = self.cfg.training.stage2.temp_final
            tau = max(temp_final, temp_init - (temp_init - temp_final) * (epoch / max(1, epochs // 2)))

            for batch in self.train_loader:
                lr_tir = batch["tir_200m"].to(self.device)
                target_rgb = batch["rgb_100m_512"].to(self.device)

                optimizer.zero_grad()
                features = self.backbone(lr_tir)
                pred_sr = self.sr_head(features, lr_tir).detach()

                logit_weights, means, log_scales = self.mixture_head(features, pred_sr)
                logit_weights = logit_weights / tau

                loss = nll_criterion(logit_weights, means, log_scales, target_rgb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.mixture_head.parameters()) + list(self.backbone.parameters()), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()
            train_loss /= len(self.train_loader)

            val_loss = self._validate_color(nll_criterion, tau)
            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} (temp={tau:.2f})")

            if val_loss < best_loss:
                best_loss = val_loss
                bb_temp = os.path.join(self.checkpoint_dir, "backbone_stage1.pth.tmp")
                mix_temp = os.path.join(self.checkpoint_dir, "mixture_head_stage2.pth.tmp")
                torch.save(self.backbone.state_dict(), bb_temp)
                torch.save(self.mixture_head.state_dict(), mix_temp)
                os.replace(bb_temp, os.path.join(self.checkpoint_dir, "backbone_stage1.pth"))
                os.replace(mix_temp, os.path.join(self.checkpoint_dir, "mixture_head_stage2.pth"))
                logger.info(f"New best validation color loss locked: {best_loss:.4f}")

        # Record stage 2 completion in central state
        PipelineState().record_stage2_completion(self.checkpoint_dir)
        logger.info("Stage 2 completion recorded in pipeline_state.json")

    def _validate_color(self, criterion, tau=1.0):
        self.backbone.eval()
        self.sr_head.eval()
        self.mixture_head.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                lr_tir = batch["tir_200m"].to(self.device)
                target_rgb = batch["rgb_100m_512"].to(self.device)

                features = self.backbone(lr_tir)
                pred_sr = self.sr_head(features, lr_tir)
                logit_weights, means, log_scales = self.mixture_head(features, pred_sr)
                
                logit_weights = logit_weights / tau

                loss = criterion(logit_weights, means, log_scales, target_rgb)
                total_loss += loss.item()
                count += 1

        return total_loss / max(1, count)

    def _init_mixture_means_with_quantiles(self):
        """Initializes components' mean bias parameters from empirical quantiles."""
        logger.info("Running empirical quantile initialization for mixture components...")
        report = generate_dataset_report(
            self.cfg.data.patches_dir,
            product_ids=list(self.cfg.data.splits.train),
        )
        if not report or not isinstance(report, dict):
            logger.warning("Could not build dataset report; skipping quantile mean init.")
            return

        rgb_info = report.get("rgb")
        if not isinstance(rgb_info, dict):
            logger.warning("Dataset report missing RGB metrics; skipping quantile mean init.")
            return

        if self.mixture_head.proj.bias is None:
            logger.warning("MixtureHead projection layer has no bias tensor; skipping quantile mean init.")
            return

        K = int(self.cfg.training.stage2.K)
        qr = np.array(rgb_info.get("quantiles_r", [50.0] * K))
        qg = np.array(rgb_info.get("quantiles_g", [50.0] * K))
        qb = np.array(rgb_info.get("quantiles_b", [50.0] * K))

        with torch.no_grad():
            self.mixture_head.proj.bias.fill_(0.0)
            for k in range(K):
                self.mixture_head.proj.bias.data[K + k * 3 + 0] = float(qr[min(k, len(qr)-1)])
                self.mixture_head.proj.bias.data[K + k * 3 + 1] = float(qg[min(k, len(qg)-1)])
                self.mixture_head.proj.bias.data[K + k * 3 + 2] = float(qb[min(k, len(qb)-1)])
        
        logger.info(f"Successfully initialized {K} components' means with empirical quantiles.")

def nn_L1Loss():
    return nn.L1Loss()
