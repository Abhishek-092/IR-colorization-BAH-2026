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
        self.device = self._resolve_device(cfg.device)
        logger.info(f"Training device resolved to: {self.device}")
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
            product_ids=cfg.data.splits.train,
            augment=True
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

    @staticmethod
    def _resolve_device(requested):
        """Pick the best available device. Honors an explicit CUDA/MPS request
        when available, otherwise falls back CUDA -> MPS (Apple Silicon) -> CPU.
        The old logic only checked CUDA and silently ran on CPU on Macs."""
        req = str(requested).lower()
        if req.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        if req == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

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

        # Load the trained SR head from Stage 1. Stages run as separate processes
        # (cli train-stage1 then train-stage2), so without this the SR head would
        # stay randomly initialized and the mixture head would be trained on SR
        # conditioning that does NOT match what inference re-loads — a silent
        # train/deploy mismatch. Load it here so Stage 2 sees the real SR output.
        sr_path = os.path.join(self.checkpoint_dir, "sr_head_stage1.pth")
        if os.path.exists(sr_path):
            self.sr_head.load_state_dict(torch.load(sr_path, map_location=self.device))
            logger.info("Loaded Stage 1 SR head weights successfully.")

        # Freeze ALL Stage-1 weights (backbone + SR head), per the two-stage
        # design: Stage 2 trains the mixture head only. Fine-tuning the shared
        # backbone here would (a) silently invalidate the frozen SR head that
        # consumes its features and (b) create a deploy mismatch, because the
        # stage-2 checkpoint stores only the mixture head while inference re-loads
        # the stage-1 backbone.
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.sr_head.parameters():
            p.requires_grad = False
        self.backbone.eval()   # also freeze BatchNorm running stats
        self.sr_head.eval()

        # Quantile-based component-mean initialization
        self._init_mixture_means_with_quantiles()

        # --- Colour-rarity rebalancing (Zhang et al. style) -----------------
        # Bin the training palette into a 16^3 RGB histogram and weight each
        # pixel's NLL by inverse bin frequency: rare colours (water-blue,
        # snow-white) stop being sacrificed to the dominant land palette.
        self.colour_lut = self._build_colour_rebalance_lut().to(self.device)

        # --- Class-balanced patch sampler -----------------------------------
        # Oversample patches that CONTAIN rare colours so every epoch actually
        # sees water/snow often enough for the rebalanced loss to act on them.
        stage2_loader = self._make_balanced_stage2_loader()

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

            # Anti-collapse regularizer weight, annealed linearly to 0 over
            # training. Early on it pushes the mixture to use all K components
            # (preventing the single-component collapse that produced flat colour);
            # late training removes the pressure so the model can specialise.
            lam0 = float(getattr(self.cfg.training.stage2, "entropy_reg", 0.0))
            lam = lam0 * max(0.0, 1.0 - epoch / max(1, epochs))
            K = self.cfg.training.stage2.K
            max_ent = float(np.log(K))
            # Weight of the direct L1 colour anchor (see loss composition below).
            anchor_w = float(getattr(self.cfg.training.stage2, "mean_anchor", 0.05))

            for batch in stage2_loader:
                lr_tir = batch["tir_200m"].to(self.device)
                target_rgb = batch["rgb_100m_512"].to(self.device)

                optimizer.zero_grad()
                # Stage-1 weights are frozen: run backbone + SR head without grad;
                # they are pure (fixed) conditioning for the mixture head.
                with torch.no_grad():
                    features = self.backbone(lr_tir)
                    pred_sr = self.sr_head(features, lr_tir)

                # Predict mixture parameters
                logit_weights, means, log_scales = self.mixture_head(features, pred_sr)

                # Apply softmax temperature division to logits
                logit_weights = logit_weights / tau

                # Per-pixel rarity weights from the colour LUT (16^3 bins)
                with torch.no_grad():
                    idx = torch.clamp(target_rgb, 0, 255).long() // 16     # (B,3,H,W)
                    pix_w = self.colour_lut[idx[:, 0], idx[:, 1], idx[:, 2]]  # (B,H,W)

                loss = nll_criterion(logit_weights, means, log_scales, target_rgb,
                                     pixel_weights=pix_w)

                pi = torch.softmax(logit_weights, dim=1)              # (B,K,H,W)

                # Direct colour anchor: L1 between the mixture mean and the target
                # RGB. The NLL shapes the *distribution*; this term guarantees the
                # colour itself is supervised, so component means can never drift
                # into a degenerate region the NLL alone fails to pull them out of.
                if anchor_w > 0.0:
                    mix_mean = (pi.unsqueeze(2) * means).sum(dim=1)   # (B,3,H,W)
                    loss = loss + anchor_w * F.l1_loss(mix_mean, target_rgb)

                # Load-balancing anti-collapse term: maximize the entropy of the
                # batch-averaged component usage so no single component absorbs
                # every pixel. Subtracting it from the loss (with annealed lam)
                # rewards balanced usage without forcing per-pixel indecision.
                if lam > 0.0:
                    pi_bar = pi.mean(dim=(0, 2, 3))                   # (K,)
                    usage_entropy = -(pi_bar * torch.log(pi_bar + 1e-8)).sum()
                    loss = loss - lam * (usage_entropy / max_ent)     # normalized to [0,1]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mixture_head.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()
            train_loss /= len(stage2_loader)

            # Validation Loss
            val_loss = self._validate_color(nll_criterion)
            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} (temp={tau:.2f})")

            # Checkpoint save
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.mixture_head.state_dict(), os.path.join(self.checkpoint_dir, "mixture_head_stage2.pth"))
                logger.info(f"New best validation color loss locked: {best_loss:.4f}")

    def _build_colour_rebalance_lut(self, alpha=0.5, max_w=8.0, sample_stride=8):
        """16^3 RGB-bin inverse-frequency weights from the training targets.
        w = (p_smoothed)^-alpha, normalized to mean 1 over observed pixels and
        clipped to [0.25, max_w]. alpha=0.5 is the Zhang et al. compromise:
        strong enough to save rare colours, weak enough not to destabilise."""
        logger.info("Building colour-rebalance LUT from training targets...")
        hist = np.zeros((16, 16, 16), dtype=np.float64)
        for i in range(len(self.train_dataset)):
            rgb = self.train_dataset[i]["rgb_100m_512"].numpy()
            px = np.clip(rgb[:, ::sample_stride, ::sample_stride], 0, 255).astype(np.int64) // 16
            np.add.at(hist, (px[0].ravel(), px[1].ravel(), px[2].ravel()), 1.0)
        p = hist / max(hist.sum(), 1.0)
        # uniform smoothing over the simplex prevents infinite weights on unseen bins
        p_smooth = 0.85 * p + 0.15 / p.size
        w = p_smooth ** (-alpha)
        # normalize so the EXPECTED weight over the data distribution is 1
        w = w / max((w * p).sum(), 1e-12)
        w = np.clip(w, 0.25, max_w)
        logger.info(f"Colour LUT ready: weight range {w.min():.2f}..{w.max():.2f}")
        return torch.tensor(w, dtype=torch.float32)

    def _make_balanced_stage2_loader(self, boost=6.0, sample_stride=16):
        """WeightedRandomSampler over training patches: weight 1 + boost*rare,
        where rare = fraction of pixels that are bluish (water), very bright
        (snow/sand) or strongly saturated — the content the flat sampler
        underexposes. Epoch length is kept equal to the dataset size."""
        weights = []
        for i in range(len(self.train_dataset)):
            rgb = self.train_dataset[i]["rgb_100m_512"].numpy()
            px = np.clip(rgb[:, ::sample_stride, ::sample_stride], 0, 255)
            r, g, b = px[0], px[1], px[2]
            bright = (r + g + b) / 3.0
            bluish = (b > r + 8) & (b > g + 4)
            snowy = bright > 180
            mx = px.max(0); mn = px.min(0)
            sat = (mx - mn) / (mx + 1e-6)
            rare = float((bluish | snowy | (sat > 0.5)).mean())
            weights.append(1.0 + boost * rare)
        weights = torch.tensor(weights, dtype=torch.double)
        logger.info(f"Balanced sampler: patch weights 1.00..{weights.max():.2f} "
                    f"(mean {weights.mean():.2f})")
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(self.train_dataset), replacement=True)
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.training.stage2.batch_size,
                          sampler=sampler, drop_last=False)

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
                # Patches store RGB on the display 0-255 scale (calibrated in
                # prepare_dataset.py), so the empirical quantiles ARE the target
                # scale — no rescaling. (The old *(255/10000) assumed raw
                # 0-10000 reflectance and initialized every mean near black on
                # 0-255 data, which the NLL could not recover from.)
                self.mixture_head.proj.bias.data[K + k * 3 + 0] = float(qb[min(k, len(qb)-1)])
                self.mixture_head.proj.bias.data[K + k * 3 + 1] = float(qg[min(k, len(qg)-1)])
                self.mixture_head.proj.bias.data[K + k * 3 + 2] = float(qr[min(k, len(qr)-1)])
        
        logger.info(f"Successfully initialized {K} components' means with empirical quantiles.")

def nn_L1Loss():
    return nn.L1Loss()
