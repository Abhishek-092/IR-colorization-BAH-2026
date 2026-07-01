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

        # Init models
        self.backbone = ResNetBackbone(in_channels=1).to(self.device)
        self.sr_head = SRHead().to(self.device)
        self.mixture_head = MixtureHead(K=cfg.training.stage2.K).to(self.device)

        # Setup dataset and loader
        self.train_dataset = PatchDataset(
            patches_dir=cfg.data.patches_dir,
            product_ids=cfg.data.splits.train
        )
        self.val_dataset = PatchDataset(
            patches_dir=cfg.data.patches_dir,
            product_ids=cfg.data.splits.val
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.training.stage1.batch_size,
            shuffle=True,
            drop_last=False
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.training.stage1.batch_size,
            shuffle=False
        )

        logger.info(f"Loaded train samples: {len(self.train_dataset)}, val samples: {len(self.val_dataset)}")

    def train_stage1_sr(self):
        """Trains the backbone and SR head deterministically."""
        logger.info("--- Starting Stage 1: Super-Resolution Training ---")
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
                # low-res input: tir_200m (B, 1, 256, 256)
                # target high-res: tir_100m_512 (B, 1, 512, 512)
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
                torch.save(self.backbone.state_dict(), os.path.join(self.checkpoint_dir, "backbone_stage1.pth"))
                torch.save(self.sr_head.state_dict(), os.path.join(self.checkpoint_dir, "sr_head_stage1.pth"))
                logger.info(f"New best validation PSNR locked: {best_psnr:.2f} dB")

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
        """Trains the mixture head with backbone weights frozen."""
        logger.info("--- Starting Stage 2: Mixture Colorization Training ---")
        
        # Load backbone weights from Stage 1
        bb_path = os.path.join(self.checkpoint_dir, "backbone_stage1.pth")
        if os.path.exists(bb_path):
            self.backbone.load_state_dict(torch.load(bb_path, map_location=self.device))
            logger.info("Loaded Stage 1 backbone weights successfully.")

        # Freeze Backbone and SR Head
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.sr_head.parameters():
            p.requires_grad = False

        # Quantile-based component-mean initialization
        self._init_mixture_means_with_quantiles()

        optimizer = optim.AdamW(
            self.mixture_head.parameters(),
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
            self.mixture_head.train()
            train_loss = 0.0

            # Calculate Softmax temperature annealing: decay linearly from temp_init to temp_final
            temp_init = self.cfg.training.stage2.temp_init
            temp_final = self.cfg.training.stage2.temp_final
            tau = max(temp_final, temp_init - (temp_init - temp_final) * (epoch / max(1, epochs // 2)))

            for batch in self.train_loader:
                lr_tir = batch["tir_200m"].to(self.device)
                target_rgb = batch["rgb_100m_512"].to(self.device)

                optimizer.zero_grad()
                with torch.no_grad():
                    features = self.backbone(lr_tir)
                    pred_sr = self.sr_head(features, lr_tir)

                # Predict mixture parameters
                logit_weights, means, log_scales = self.mixture_head(features, pred_sr)

                # Apply softmax temperature division to logits
                logit_weights = logit_weights / tau

                loss = nll_criterion(logit_weights, means, log_scales, target_rgb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()
            train_loss /= len(self.train_loader)

            # Validation Loss
            val_loss = self._validate_color(nll_criterion)
            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} (temp={tau:.2f})")

            # Checkpoint save
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.mixture_head.state_dict(), os.path.join(self.checkpoint_dir, "mixture_head_stage2.pth"))
                logger.info(f"New best validation color loss locked: {best_loss:.4f}")

    def _validate_color(self, criterion):
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

                loss = criterion(logit_weights, means, log_scales, target_rgb)
                total_loss += loss.item()
                count += 1

        return total_loss / max(1, count)

    def _init_mixture_means_with_quantiles(self):
        """Initializes components' mean bias parameters from empirical quantiles."""
        logger.info("Running empirical quantile initialization for mixture components...")
        report = generate_dataset_report(self.cfg.data.patches_dir)
        if not report:
            logger.warning("Could not build dataset report; skipping quantile mean init.")
            return

        K = self.cfg.training.stage2.K
        
        # We assign each component's mean bias based on the quantiles of RGB
        # Quantiles are estimated from data for R, G, B channels
        # Calculate K linearly spaced quantiles between 10% and 90%
        quantiles = np.linspace(10, 90, K)
        
        # Read empirical values from report
        # If report lacks them, fall back to safe approximations
        qr = np.array(report["rgb"].get("quantiles_r", [50.0] * K))
        qg = np.array(report["rgb"].get("quantiles_g", [50.0] * K))
        qb = np.array(report["rgb"].get("quantiles_b", [50.0] * K))

        # Assign bias values inside projection conv of mixture head
        # self.mixture_head.proj is a Conv2d(64, 7*K, 1)
        # Bias tensor size: 7*K
        with torch.no_grad():
            # Zero-out weights and set default biases
            self.mixture_head.proj.bias.fill_(0.0)
            
            for k in range(K):
                # Channel indexing match: bias[K + k*3 + c]
                # channel 0 = Blue, channel 1 = Green, channel 2 = Red
                # Scale from original 0-10000 reflectance range to [0, 255]
                self.mixture_head.proj.bias.data[K + k * 3 + 0] = float(qb[min(k, len(qb)-1)]) * (255.0 / 10000.0)
                self.mixture_head.proj.bias.data[K + k * 3 + 1] = float(qg[min(k, len(qg)-1)]) * (255.0 / 10000.0)
                self.mixture_head.proj.bias.data[K + k * 3 + 2] = float(qr[min(k, len(qr)-1)]) * (255.0 / 10000.0)
        
        logger.info(f"Successfully initialized {K} components' means with empirical quantiles.")

def nn_L1Loss():
    return nn.L1Loss()
