# VARNA — Implementation Plan
**From Accepted Proposal to Working System: Repository, Pipeline, and Deployment Design**
*Prepared by: Principal Investigator, for the implementation team*

This document assumes the VARNA scientific design (prior report) has been accepted. It is written for engineers who need to start writing code today. Every section states the scientifically "correct" choice, the hackathon-feasible choice, and which one we are actually building — because under a fixed deadline, every module must be justified twice: once on rigor, once on hours-to-implement.

**Assumed timeline:** ~3–4 weeks of pre-finale build time (typical SIH internal-to-finale runway) + a 36-hour grand finale window for integration, polish, and live demo. The architecture below is sized so the *entire* pipeline is trainable on a single mid-range GPU (T4/RTX 3060-class, 8–16 GB VRAM) within that runway — this constraint dominates almost every downstream decision more than anything else in this document.

---

## 1. Repository Structure

```
varna/
├── README.md
├── pyproject.toml                  # single env for whole team — no per-module envs
├── configs/
│   ├── data.yaml
│   ├── stage1_sr.yaml
│   ├── stage2_classifier.yaml
│   ├── stage3_color.yaml
│   └── inference.yaml
├── data/
│   ├── raw/                        # untouched GEE/USGS downloads, gitignored
│   ├── interim/                    # calibrated, co-registered, tiled
│   ├── processed/                  # train/val/test tensors (memmapped .npy / WebDataset shards)
│   └── splits/                     # geographic train/val/test split lists (see §3.4)
├── varna/
│   ├── calibration/
│   │   └── planck.py               # Stage 0: DN -> radiance -> brightness temp
│   ├── datasets/
│   │   ├── landsat9_dataset.py
│   │   └── degradation.py          # PSF + downsample simulator for SR pairs
│   ├── models/
│   │   ├── sr_unrolled.py          # Stage 1
│   │   ├── classifier_head.py      # Stage 2
│   │   ├── color_decoder.py        # Stage 3 (mixture density head)
│   │   └── backbone.py             # shared encoder
│   ├── losses/
│   │   ├── sr_losses.py
│   │   ├── class_losses.py
│   │   └── color_losses.py         # NLL, structural-consistency, calibration
│   ├── train/
│   │   ├── train_stage1.py
│   │   ├── train_stage2.py
│   │   ├── train_stage3.py
│   │   └── finetune_joint.py       # optional, low-LR end-to-end pass
│   ├── inference/
│   │   ├── pipeline.py             # fused forward pass, triage logic
│   │   ├── export_onnx.py
│   │   └── triage.py
│   └── eval/
│       ├── metrics.py              # PSNR/SSIM/FID/calibration/per-class color acc.
│       └── report.py               # auto-generates evaluation tables/plots
├── serving/
│   ├── app.py                      # FastAPI inference server
│   └── Dockerfile
├── demo/
│   └── streamlit_app.py            # judge-facing visual demo
├── scripts/
│   ├── download_landsat9.py
│   ├── build_pseudo_labels.py      # land-cover weak labels
│   └── run_full_eval.sh
└── tests/
    └── ...                        # unit tests on calibration math, shapes, losses
```

**Justification:** one config-driven mono-repo, not separate repos per stage. With a small team and a hard deadline, the cost of cross-repo version drift (engineer A trains Stage 1 against a calibration function that engineer B has since changed) is far more dangerous than the modest loss of "clean separation." Config files (Hydra/OmegaConf) make stage boundaries explicit without paying the cost of separate packages.

---

## 2. Tech Stack Decisions

| Component | Scientifically "ideal" | Hackathon choice | Reasoning |
|---|---|---|---|
| Framework | Custom CUDA kernels for unrolled SR | **PyTorch**, standard layers only | Team velocity matters more than 10–15% speed; PyTorch → ONNX → TensorRT path is well-trodden and de-risked |
| Data access | Direct USGS EarthExplorer bulk order (slow, manual) | **Google Earth Engine Python API** | GEE gives pre-calibrated, cloud-masked, co-registered Landsat 9 Collection 2 Level-2 product on demand — removes the single biggest time sink (data wrangling) from the critical path |
| Land-cover weak labels | Custom TES-based emissivity retrieval | **ESA WorldCover 10 m, reprojected/aggregated** via GEE | Already exists, globally available, zero labeling cost |
| Experiment tracking | Custom logging | **Weights & Biases (free tier)** | Needed for judges' technical Q&A — live loss curves and calibration plots are persuasive evidence of rigor |
| Serving | Custom C++ inference server | **FastAPI + ONNX Runtime / TensorRT** | Good enough latency, far less engineering risk |

---

## 3. Data Pipeline

### 3.1 Source
Landsat 9 Collection 2 Level-2 (Surface Reflectance + Surface Temperature products), pulled via GEE for a curated set of regions chosen to maximize land-cover diversity: at minimum one each of (a) coastal/water-dominant, (b) dense urban, (c) agricultural plains, (d) arid/bare-soil, (e) forested. **Trade-off:** scientific completeness would call for stratified global sampling across climate zones; hackathon feasibility caps this at 5–8 Indian regions (ISRO's operational domain anyway), which is both more defensible to the judges (directly relevant to ISRO's use case) and an order of magnitude less data to manage.

### 3.2 Preprocessing (Stage 0, implemented once, cached)
1. Pull OLI surface reflectance (RGB-equivalent bands) and TIRS-2 surface temperature, already co-registered by the Level-2 product (GEE handles this — **do not re-implement geometric co-registration**, it is the single largest time-sink if done manually).
2. Convert TIRS-2 to brightness temperature using `planck.py` (closed-form, a few lines — see §1).
3. Tile into 256×256 patches at 100 m, with 10% overlap discarded at inference stitching time to avoid edge artifacts.
4. Simulate the 200 m input by applying a Gaussian PSF (σ tuned to the published TIRS PSF, ≈1.5 px) and 2× downsampling — `degradation.py`. **This produces the only labels Stage 1 needs and requires no manual annotation.**
5. Generate weak land-cover labels per tile by reprojecting ESA WorldCover and majority-voting into a reduced 5-class taxonomy (water / vegetation / bare-soil / built-up / cloud-shadow-nodata) — fewer classes than WorldCover's native 11, deliberately: **trade-off** — finer taxonomy is scientifically richer but with limited training data per class, 5 broad classes keep per-class sample counts high enough to actually learn calibrated statistics within the timeline.

### 3.3 Storage format
WebDataset shards (`.tar` of paired `.npy` tensors) rather than a database. Sequential-read-friendly, trivially parallelizable across data-loader workers, and requires no infrastructure beyond local disk — appropriate for a hackathon compute budget.

### 3.4 Splits
**Geographic split, not random pixel/tile split.** Train on 4 regions, hold out 1–2 entire regions for validation/test. Random tile-level splitting would leak spatially correlated information (adjacent tiles from the same scene) between train and test, inflating reported metrics — a mistake judges familiar with remote sensing will specifically probe for. This is a place where we spend a small amount of extra setup time to avoid a result that would not survive scrutiny.

---

## 4. Model Architecture (concrete, implementable)

### 4.1 Shared Backbone
A small ResNet-style encoder, 4 stages, channel widths [32, 64, 128, 256], ~2–3M parameters. **Why so small:** the inputs are single-channel, spatially smooth (band-limited by the thermal PSF) — there is no evidence a heavier backbone (ResNet-50-scale, 25M+ params) improves results on this signal, and a smaller model trains in hours instead of days on a single GPU, which is the binding constraint. This is the single largest scientific-elegance-vs-feasibility trade-off in the whole project, and it is the correct one: an oversized backbone on a deadline risks an under-trained model, which is strictly worse than a properly-trained small model.

### 4.2 Stage 1 — Unrolled SR
- **Scientifically proposed:** 3–5 unrolled iterations of (data-consistency step + learned denoiser).
- **Hackathon implementation:** **3 iterations fixed** (not a tunable knob explored at length), denoiser = 4-layer residual CNN block reused across iterations (weight sharing across iterations, not 3 separate blocks) — this is both more parameter-efficient and mathematically appropriate (an unrolled algorithm conceptually applies the *same* prior at each step). Data-consistency step is implemented as a closed-form least-squares update against the known degradation operator (no learned component needed — it's pure linear algebra, a few lines of PyTorch using the known Gaussian blur kernel).
- **Trade-off explicitly flagged:** a fully learned, data-driven PSF estimation (rather than the published nominal PSF) would be more accurate but adds an entire estimation sub-problem with its own failure modes; we use the **published nominal PSF as a fixed, known constant**. This is defensible — it's how operational thermal SR literature typically proceeds — and removes a degree of freedom that the team does not have time to validate properly.

### 4.3 Stage 2 — Class Posterior
A 1×1 conv head on top of the shared backbone features, 5-way softmax output, plus one 3×3 conv layer for local context before the head (since land cover has spatial coherence, a pure pixel-wise classifier with zero receptive-field context underperforms). Trained against the WorldCover-derived weak labels with a 5×5 CRF-style smoothness term implemented simply as a total-variation penalty on the softmax output (full CRF inference is scientifically nicer but adds inference latency and implementation complexity disproportionate to the gain — **rejected for hackathon scope**).

### 4.4 Stage 3 — Color Decoder
- **Scientifically proposed:** full mixture-density network (K Gaussian components per class, context-modulated).
- **Hackathon implementation:** **single Gaussian per class** (i.e., K=1), with mean and a 3×3 diagonal covariance predicted per pixel, modulated by a small conv head conditioned on (class posterior ⊕ backbone features). A single Gaussian is a strict special case of the mixture model — it is the same mathematical framework with K reduced from "several" to "one" to cut training instability and the number of hyperparameters that need tuning under time pressure, while still providing the calibrated mean + uncertainty output that is the framework's core scientific claim. **This is presented to judges explicitly as a scoped-down instance of the proposed architecture, not a different one** — important for defending the implementation against the original design during Q&A.
- Loss: Gaussian NLL (closed form, trivial to implement and stable to train) + low-weight (λ≈0.05) perceptual term using a frozen, pretrained VGG feature extractor (off-the-shelf, zero extra training cost) + structural-consistency gradient-correlation term from the proposal (Sobel filters on both luminance and T_B, a few lines).

### 4.5 Parameter Budget Summary
| Module | Params | Rationale |
|---|---|---|
| Shared backbone | ~2.5M | Sized to the information content of single-channel thermal input |
| SR denoiser block (shared across iterations) | ~0.3M | Lightweight, reused 3× |
| Class head | ~0.1M | Simple, 5-way |
| Color decoder head | ~0.5M | Single Gaussian, modest |
| **Total** | **~3.4M** | Trains end-to-end on 8GB VRAM with batch size 16–32 at 256×256 |

---

## 5. Training Pipeline

### 5.1 Phased training (kept exactly as proposed — not simplified)
1. **Stage 1 alone** (~1 day on a single GPU): degradation-consistency + BT L1 + edge-gradient loss. Convergence checked via PSNR plateau on held-out region.
2. **Stage 2 alone** (~few hours, smaller task): cross-entropy + TV smoothness, backbone frozen from Stage 1 (transfer the SR-trained features rather than training Stage 2 from scratch — saves time and gives the classifier features that already encode calibrated brightness temperature).
3. **Stage 3** (~1 day): Gaussian NLL + perceptual + structural-consistency, Stages 1–2 **frozen**. Joint end-to-end fine-tuning at low LR (10⁻⁵) for a final short pass (a few hours) **only if time permits** — flagged as optional/stretch goal, not a launch blocker, since the phased model is already a complete, defensible system without it.

**Why phased, not joint-from-scratch, even under time pressure:** joint training of all three stages from random initialization is the single most likely way to burn the entire compute budget on an unstable, non-converging run two days before the deadline. Phased training with frozen intermediate stages is slower in wall-clock optimality but dramatically lower-risk — given a hard deadline, **risk reduction dominates marginal accuracy gains**. This is the most important engineering-versus-elegance call in the whole plan.

### 5.2 Hyperparameters (starting points, not exhaustively tuned)
- Optimizer: AdamW, lr 3e-4 (Stage 1/2), 1e-4 (Stage 3), cosine decay.
- Batch size: 16 (256×256 tiles) — fits comfortably in 8–16GB.
- Augmentation: random flip/rotate only (90°/180°/270°) — **no aggressive photometric augmentation on the TIR channel**, since artificial brightness/contrast jitter on thermal data does not correspond to any physically realizable scenario and would teach the model invariances that don't exist in real acquisitions.
- Early stopping on Stage 1 PSNR (true ground truth available, so this is a trustworthy stopping criterion); Stage 3 stopped on calibration error on held-out region, not on NLL alone (NLL can be deceptively low while uncertainty is miscalibrated).

### 5.3 Compute budget
Single GPU, ~2.5–3 days total wall clock across all three phases — fits inside a 3–4 week pre-finale runway with substantial margin for debugging, leaving the finale 36 hours for integration/demo/polish rather than first-time training.

---

## 6. Inference Engine

### 6.1 Pipeline implementation
`pipeline.py` implements the fused forward pass exactly as specified in the proposal (triage → calibration → SR → class posterior → color decoder → abstention masking), as a single `torch.nn.Module` wrapping all three sub-networks, so it exports to ONNX as one graph rather than three separate ones (avoids host-device round-trip overhead between stages).

### 6.2 Triage simplification
**Scientifically proposed:** tile-level entropy/variance routing to two distinct compute tracks (different iteration counts, different model capacity).
**Hackathon implementation:** keep entropy/variance computation (cheap, a few lines), but route to **the same network with iteration count switched between 1 and 3** (rather than maintaining two separately-trained model variants). This captures most of the latency benefit (the SR unrolling is the dominant cost) without doubling the number of models the team has to train, validate, and keep in sync — a meaningful implementation-complexity reduction for a small accuracy cost on already-easy tiles.

### 6.3 Abstention threshold
Calibrated post-hoc on the validation set by choosing the uncertainty threshold that achieves a target Expected Calibration Error, rather than learned jointly (a learned threshold is more elegant but adds an optimization target competing with the main losses; a simple post-hoc calibration sweep, standard practice, is implementable in an afternoon and is exactly as defensible scientifically).

### 6.4 Export and serving
- `export_onnx.py`: trace the fused module, export FP32 ONNX, validate numerical parity against the PyTorch model on a held-out batch (tolerance check — this step is not optional, silent ONNX export bugs are a common late-stage failure).
- TensorRT engine build with FP16 (not INT8) as the default for the demo: **trade-off** — INT8 needs a calibration dataset and a validation pass to confirm brightness-temperature error stays within sensor noise floor (as specified in the proposal), which is the scientifically correct deployment target, but is a stretch goal for the finale, not the baseline demo path, because a botched INT8 calibration that silently degrades temperature accuracy is worse for a live demo than an honestly-reported FP16 latency number.
- FastAPI server (`serving/app.py`) wrapping the TensorRT/ONNX Runtime engine, batched tile endpoint, returns SR-TIR, RGB, and confidence map as separate PNG/array payloads.

---

## 7. Evaluation Methodology (implementation)

`eval/metrics.py` implements, in this priority order (matching what the proposal claims, so results and claims are consistent under judge questioning):
1. PSNR/SSIM/BT-RMSE for Stage 1 against true 100 m TIRS-2 — **the headline, hardest-to-dispute number**.
2. Per-class color accuracy: mean RGB error per predicted class vs. true OLI class-conditional statistics on held-out region.
3. Calibration: Expected Calibration Error + sparsification plot (rejected-fraction vs. residual error), generated automatically via `eval/report.py` as a single figure — this is the plot that most directly demonstrates the "minimizes hallucination, and proves it" claim to judges.
4. FID, reported as a secondary/sanity metric only, with explicit framing in the demo narrative that it is not the primary evidence of correctness (pre-empting the obvious judge question "but does it just look real or is it actually right").
5. Latency: ms/tile by triage track, plotted as accuracy-vs-latency trade-off curve.

`run_full_eval.sh` runs all of the above end-to-end and regenerates the report — this should be run the night before the finale demo and the numbers in the slide deck must match the script output exactly (a common, avoidable credibility loss is a deck with numbers that don't reproduce live).

---

## 8. Demo Design (judge-facing, often underweighted by technical teams)

`demo/streamlit_app.py`: upload or pick a held-out tile → side-by-side panel of (raw 200 m TIR, SR 100 m TIR, synthesized RGB, confidence/abstention overlay). The confidence overlay is shown **by default, not as an optional toggle** — the entire scientific argument of the project is that uncertainty is a first-class output, and the demo should make that visually unmissable rather than something a judge has to ask for. Include one held-out heterogeneous (urban-rural fringe) tile pre-loaded as the default example, since that is where the framework's advantage over a naive GAN baseline is most visible — homogeneous water/desert tiles will not differentiate VARNA from a baseline in a live demo and should be avoided as the lead example.

---

## 9. Implementation Schedule

| Week | Deliverable |
|---|---|
| 1 | Data pipeline (§3) end-to-end; calibration unit-tested; weak labels generated for all regions |
| 2 | Stage 1 trained and evaluated (PSNR/SSIM numbers locked); Stage 2 trained |
| 3 | Stage 3 trained; calibration sweep; full eval report generated; ONNX export validated |
| 4 (pre-finale buffer) | TensorRT FP16 build; FastAPI serving; Streamlit demo; slide deck built from `eval/report.py` output directly |
| Finale (36h) | Joint fine-tune (stretch goal) if time allows; INT8 attempt (stretch goal); live demo rehearsal; Q&A prep on the trade-off table in this document |

---

## 10. Summary of Every Elegance-vs-Feasibility Call Made

| Decision point | Scientific ideal | What we build | Risk if we'd chosen the ideal under this timeline |
|---|---|---|---|
| Backbone size | Large, pretrained vision backbone | ~2.5M-param custom small encoder | Under-trained on deadline, longer iteration cycles |
| SR unrolling depth | 5 iterations, possibly learned PSF | 3 fixed iterations, known PSF | Estimation sub-problem adds unvalidated failure mode |
| Color decoder | Full mixture density (K>1) | Single Gaussian (K=1) per class | Training instability, harder to debug under time pressure |
| Land-cover taxonomy | Fine-grained (10+ classes) | 5 broad classes | Sparse per-class statistics, miscalibrated uncertainty |
| CRF smoothness | Full structured CRF inference | TV penalty approximation | Inference latency, implementation time disproportionate to gain |
| Training schedule | Joint end-to-end from scratch | Phased, frozen-stage training | High risk of non-convergence close to deadline |
| Quantization | INT8 with validated calibration | FP16 baseline, INT8 as stretch | Silent temperature-accuracy degradation in a live demo |
| Triage | Two separately trained model variants | One model, iteration-count switch | Doubles training/validation burden for marginal latency gain |

Every row above preserves the **mathematical structure** of the original proposal (each is a legitimate special case or approximation of the proposed design, not a different idea substituted in) — this is deliberate, so the implemented system can be presented to judges as *the proposal, scoped to what can be properly validated in the available time*, rather than as a retreat from it.
