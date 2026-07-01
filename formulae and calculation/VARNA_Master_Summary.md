# VARNA — Project Master Summary
**Variational class-Aware Radiance-to-reflectance Network for hallucination-Aware colorization of Thermal Infrared Imagery**
*Bharatiya Antariksh Hackathon (BAH) 2026 — consolidated project document*

This document is the single-file summary of the entire VARNA project: what is being built, why it is built this way, and exactly how a four-member team implements it within the BAH timeline. It synthesizes the scientific proposal, the hostile architectural review, the formulation review, the engineering stability analysis, the implementation plan, the scientific validation plan, and the two engineering-systems reviews into one coherent account. Nothing here is new; this is the project as it currently stands, end to end.

---

## 1. The Problem

The official task — "Infrared Image Colorization and Enhancement" — bundles two operations that are physically unrelated and must be designed, trained, and evaluated as **separate sub-problems**:

1. **Spatial super-resolution of TIR (200 m → 100 m)** — a within-modality inverse problem (deblurring + upsampling a thermal radiance field). Comparatively well-posed, since the degradation operator is known.
2. **TIR → RGB synthesis** — a cross-modality inference problem. Thermal emission (governed by surface temperature and emissivity) and visible reflectance (governed by spectral reflectance — chemistry, roughness, moisture) are different physical processes that share only an indirect common cause: surface material. **There is no deterministic or even approximately bijective mapping from TIR to RGB.** Any model claiming to "reconstruct" RGB from TIR is performing conditional estimation constrained by learned material statistics, not physical inversion — and this fact, not an engineering shortfall, is what makes the colorization problem fundamentally different from the SR problem.

**Why existing methods fail:** Pix2Pix/CycleGAN-style translation and diffusion-based colorization treat TIR→RGB as direct pixel regression/generation. None of them separate "what surface is this" from "what does this surface look like," none can express low confidence, and adversarial losses are mode-seeking — they reward confident fabrication over honest uncertainty. The result, well documented in thermal-colorization literature, is visually convincing but locally false output: **hallucination**.

**Realistic ceiling:** structure can be recovered with high fidelity (legitimate SR problem). Color can only be assigned at the resolution of land-cover class statistics, with explicit, quantified uncertainty. A framework promising pixel-accurate "true color" from TIR alone is scientifically indefensible and would be flagged immediately by any competent reviewer.

---

## 2. The Scientific Formulation (Frozen)

### 2.1 Why a deterministic model is mathematically wrong, not just imprecise
For a fixed brightness temperature T_B, multiple physically distinct materials are consistent with that radiance (dry asphalt and dark basaltic soil can share T_B but differ completely in RGB). The true generative relationship is **RGB | T_B ~ a genuinely multimodal distribution, not a function**. Under squared-error training, a deterministic model converges to the conditional expectation E[RGB | T_B] — the posterior mean. When the posterior is multimodal, this mean lands **between** the modes, in a region the true distribution assigns low density: a color belonging to no real material. This is not a training failure to fix with a better network; it is what minimizing expected squared error *means* whenever the target is multimodal. Three independent arguments (information theory / data-processing inequality, ill-posed-inverse-problem uniqueness, Bayesian sufficiency-statistic theory) each independently confirm a deterministic point estimate cannot represent this estimand, regardless of model capacity or data volume.

### 2.2 The adopted formulation
VARNA estimates the full conditional distribution directly, as a **discretized logistic mixture**:

```
p(RGB | T_B) = Σ_{k=1}^{K} π_k(T_B) · Π_{c∈{R,G,B}} [σ((x_c+0.5−μ_{k,c})/s_{k,c}) − σ((x_c−0.5−μ_{k,c})/s_{k,c})]
```

- **K ∈ {6, 7, 8}**, selected via a held-out-NLL sweep (run only after engineering stability fixes are in place — see §4).
- **σ** is the logistic sigmoid; each factor is the probability mass a discretized logistic places on the integer pixel bin — by construction this cannot assign density outside the valid [0,255] range, correcting a real (if small) physical inconsistency a continuous Gaussian mixture would have, at zero extra implementation cost.
- **Training objective:** maximize the log-likelihood directly (NLL loss, log-sum-exp stabilized) — one loss, no auxiliary terms, no λ-weighted side objectives.
- **P** (deterministic radiometric calibration: digital numbers → brightness temperature via the Planck relation) is not a second random variable; it parameterizes which fixed transform produces T_B in the first place.

### 2.3 Decoding rule — the load-bearing distinction
Two downstream quantities are computed from {π_k, μ_k, s_k} with no sampling, no iteration:
- **Reported color:** μ_{k*}, the mean of the **dominant component** (k* = argmax_k π_k) — *not* the mixture mean Σ_k π_k μ_k. The mixture mean would reintroduce the invalid-blend problem at one remove; the dominant-component mean always corresponds to one internally coherent physical hypothesis, preserving sharp, physically meaningful transitions at material boundaries instead of synthesizing a soft, anatomically incorrect gradient.
- **Uncertainty**, decomposed by the law of total variance into two physically distinct, separately interpretable terms:
  - **Within-mode variance** (Σ_k π_k s_k² c_logistic) — genuine radiometric/measurement noise around one physical hypothesis.
  - **Between-mode variance** (Σ_k π_k(μ_k−μ̄)²) — disagreement among distinct, competing physical hypotheses, i.e. genuine material ambiguity.
  - **Entropy of {π_k}** — an independent, interpretable multimodality signal.

### 2.4 Why this is the correct, and minimal, formulation
A candidate-comparison review considered and rejected five heavier alternatives — normalizing flows, conditional flow matching, energy-based models, Dirichlet process mixtures, conditional VAEs, conditional diffusion — each on a combination of disproportionate engineering cost, training instability, or multi-step sampling latency that directly violates the low-inference-time BAH criterion. RGB per pixel is three numbers; a small mixture already has universal-approximation coverage of the plausible densities at this dimensionality, making the heavier machinery built for high-dimensional structured outputs unjustified here (Occam's razor, applied explicitly, not assumed). The only adopted strengthening over the originally proposed Gaussian mixture is the discretized-logistic correction (§2.2) — identical architecture, identical training loop, identical inference cost.

**An important architectural consequence:** once the colorization problem is stated as a single closed-form mixture head, the mixture weights π_k(T_B) play exactly the role an earlier, separately-trained classification stage was meant to play — except no separate stage exists. A prior architectural revision included a supervised land-cover classifier (trained against an external dataset, ESA WorldCover) as a distinct module; this was **rejected** in hostile review because it amounts to a land-cover-mapping subsystem wearing a colorization costume, with no defensible one-sentence justification and an unquantified registration-error risk between the external taxonomy and the Landsat grid. The mixture formulation makes that entire module unnecessary: there is no longer any justification for a separate classification stage, supervised or self-supervised — it was always implicit in the mixture weights.

### 2.5 Why SR remains deterministic
The SR sub-problem is conditioned on a known degradation operator, making it comparatively close to a well-posed (near-unique) inverse problem — unlike colorization's genuinely many-to-one mapping. Applying probabilistic treatment uniformly "out of habit" would be the less defensible choice; treating each sub-problem according to its own actual structure is the principled position. The SR head is a **single-pass residual CNN** (not the originally proposed multi-iteration unrolled optimization, which was rejected — see §3) trained with a soft auxiliary loss: re-degrade the network's output with the known PSF + downsample operator and penalize its distance from the actual 200 m input, plus an edge-gradient term and a direct brightness-temperature L1 term against true 100 m TIRS-2.

---

## 3. What Was Considered and Rejected (Hostile Review Findings)

A dedicated hostile-review pass, mandated to "find every reason to reject, survivors only," removed four components from earlier architectural revisions:

| Rejected | Why | Replaced with |
|---|---|---|
| Supervised land-cover classifier (external WorldCover labels) | Scope risk — becomes a separate remote-sensing application; unquantified registration-error budget; no defensible one-sentence justification under reviewer questioning | Nothing separate — absorbed entirely into the mixture weights π_k (§2.4) |
| 3-iteration unrolled-optimization SR with hard PSF data-consistency constraint | Requires an unvalidated hard PSF assumption baked into the graph; 3× inference cost, directly opposing the low-latency priority | Single-pass residual CNN with the same physics expressed as a *soft* auxiliary loss — gracefully degrades under PSF mismatch, strictly one forward pass |
| Full heteroscedastic covariance + dedicated ECE-style calibration loss | A second research-level claim stacked on the first; ECE is statistically noisy at hackathon-scale held-out sets, adding a 4th tunable λ for a claim the data can't fully support | Heteroscedastic mean+variance kept (one extra output channel, standard NLL — cheap); calibration *measured* post hoc via sparsification plots, not trained for |
| Pretrained ImageNet/VGG perceptual loss | Unexamined domain-transfer assumption (natural-photograph texture statistics applied to thermally-derived pseudo-RGB); external dependency for a 0.05-weighted, marginal term | Removed entirely; NLL + SR structural fidelity sufficient; SSIM fallback only if time remains |

This pass is also why the formulation is now described as "the leanest version of VARNA that has appeared across this entire review process — leaner because it is more correct, not despite it."

---

## 4. Engineering Stability (How the Math Is Realized Without Breaking)

The mixture-density-network literature documents well-known failure modes; each has a minimal, formulation-preserving fix. **None of these change π_k(T_B), μ_{k,c}(T_B), s_{k,c}(T_B), the discretized-logistic likelihood, or the NLL objective** — every fix is a reparameterization, initialization scheme, training schedule, arithmetic identity, execution-precision boundary, or post-processing step.

| Risk | Mechanism | Minimal fix |
|---|---|---|
| Variance (degenerate-component) collapse | A component's scale shrinks toward zero around one training pixel — locally optimal, but memorization, not generalization | Parameterize scale as `softplus(raw) + ε`, with ε fixed at the discretization's own bin width (not an arbitrary knob) |
| Mode-redundancy collapse | Components converge to near-identical parameters from symmetric initialization — no force in the loss pushes them apart | One-time, data-dependent init: place each component's mean at a different quantile of the empirical marginal RGB distribution |
| Component starvation | Gradient-routing "rich get richer" dynamic suppresses low-responsibility components throughout training | Anneal a temperature on the mixture-weight softmax early in training, τ→1 by the end (recovers the exact original objective at convergence) |
| log-sum-exp overflow/underflow | Naive computation of log Σ_k π_k·L_k overflows/underflows, especially in FP16 | Standard log-sum-exp trick (subtract max before exponentiating) |
| Catastrophic cancellation in the CDF difference | σ(a)−σ(b) loses significant digits for sharply peaked components | Stable log1p/expm1-based identity; boundary bins (x=0, x=255) handled as one-sided tail probabilities |
| Silent precision-related calibration degradation | FP16/INT8 preserves mean color but can distort distribution shape (e.g., flattening sharp components) | Force the entire decode chain (softmax → log-sum-exp → CDF diff → argmax → variance decomposition) to FP32, as an explicit module boundary, never left to export-tool default casting |
| Spatial mode-selection flicker | At near-tied components, small input noise flips which component is argmax-selected — visible salt-and-pepper artifact in otherwise-uniform regions | Fixed, non-trainable spatial smoothing of π_k before argmax, at decode time only |

**Tuning order matters and is fixed:** (1) variance floor + component-mean init, (2) softmax annealing schedule, (3) K via held-out NLL — *only after* (1)–(2) are stable, since tuning K first risks selecting a value that compensates for an optimization pathology rather than reflecting true data multimodality.

**Memory/compute cost of the mixture head:** 7K output channels (1 weight + 3 means + 3 log-scales per component) vs. 3 for plain regression — at K=6–8, 42–56 channels at the output layer only, a small linear cost confined to the head, not the ~2.5M-parameter shared backbone. Total system size: **~3.4M parameters**, trainable end-to-end on 8GB VRAM at batch size 16–32, 256×256 tiles.

---

## 5. System Architecture

```
raw TIR tile (200 m)
        │
        ▼
 Stage 0 — radiometric calibration
 T_B = g(DN_raw; P)   [Planck inversion, closed-form]
        │
        ▼
 Shared backbone (~2.5M params)
        │
        ├──► SR head (deterministic, single-pass residual CNN)
        │         → SR-TIR output (100 m)
        │         loss: L1(T_B) + degradation-consistency + edge-gradient
        │
        └──► Mixture head (7K channels: π_k, μ_k, s_k)
                  │
                  ▼
          Decode submodule (forced FP32)
                  │
                  ├──► dominant-mode RGB (μ_{k*})        — primary reported color
                  ├──► secondary hypothesis (μ_{k2}, π_{k2})
                  ├──► within-mode / between-mode variance
                  └──► entropy({π_k})
```

One shared backbone, two heads, **one forward pass per tile** — no sampling, no iteration, no second trained model anywhere in the production path. Everything else in the project (analyst tooling, explainability, deployment scheduling, the demo) is built as a consumer of this single computation's output edge, never inserted into the forward pass itself.

---

## 6. Data Pipeline

- **Source:** Landsat 9 Collection 2 Level-2 (Surface Reflectance + Surface Temperature), via Google Earth Engine — pre-calibrated, cloud-masked, co-registered, removing the single biggest time sink (manual data wrangling and geometric co-registration) from the critical path.
- **Regions:** 5–8 curated Indian regions spanning coastal/water, dense urban, agricultural plains, arid/bare-soil, and forested classes — both maximizing land-cover diversity and staying directly relevant to ISRO's operational domain.
- **Ground truth, both genuinely real, not synthetic or pseudo-labeled:**
  - SR target: real native 100 m TIRS-2, degraded via the known PSF + 2× downsampling to simulate the 200 m input — self-supervised.
  - Color target: real OLI-derived RGB, resampled and co-registered to the TIRS grid — physically real supervision, not another network's guess.
- **Tiling:** 256×256 patches at 100 m, 10% overlap discarded at stitching to avoid edge artifacts.
- **Augmentation:** flip/rotate only (90°/180°/270°). Deliberately **no photometric augmentation** on the TIR channel — artificial brightness/contrast jitter does not correspond to any physically realizable acquisition scenario and would teach invariances the real sensor never exhibits.
- **Splits:** geographic, not random — tiles from the same region/overpass are spatially and radiometrically correlated, so a per-tile random split would leak information and inflate every metric. Cross-validation uses **leave-one-region-out (LORO)**, not standard k-fold, since 5–8 regions are too few to be exchangeable folds.

---

## 7. Training

Three phases, **deliberately kept phased rather than joint-from-scratch**, because joint training of all components from random initialization is the single most likely way to burn the entire compute budget on a non-converging run two days before deadline — phased training with frozen intermediate stages is slower in wall-clock optimality but dramatically lower-risk, and under a hard deadline risk reduction dominates marginal accuracy gains:

1. **SR head alone** (~1 day on a single GPU): degradation-consistency + brightness-temperature L1 + edge-gradient loss. Convergence: held-out PSNR plateau.
2. **Mixture (color) head** (~1 day, backbone features transferred from Stage 1): NLL loss on the discretized logistic mixture, with the stability fixes (§4) active and tuned in their required order. Convergence: held-out NLL plateau, confirmed by sparsification AUC, **not** NLL alone (NLL can be deceptively low while uncertainty is miscalibrated).
3. **Optional joint fine-tune** at low LR (10⁻⁵), a few hours, only if time permits — explicitly a stretch goal, not a launch blocker, since the phased model is already a complete, defensible system without it.

- **Optimizer:** AdamW, lr 3e-4 (Stage 1), 1e-4 (Stage 2), cosine decay. Batch size 16, fits comfortably in 8–16GB.
- **Compute budget:** ~2.5–3 days total wall clock across all phases, inside a 3–4 week pre-finale runway, leaving the 36-hour finale for integration/demo/polish rather than first-time training.
- **Reproducibility:** config-driven (Hydra/OmegaConf), single mono-repo (not per-stage repos — cross-repo version drift is more dangerous than the modest loss of "clean separation" for a small team under deadline), every reported number traceable to a config + git commit hash + W&B run ID + fixed data-split manifest.

---

## 8. Evaluation

| Output | Metric | Role |
|---|---|---|
| SR-TIR | PSNR, SSIM, brightness-temperature RMSE | Headline, hardest-to-dispute number — true ground truth exists |
| RGB | Per-class color accuracy (predicted dominant-mode color vs. true OLI class-conditional statistics) | The metric that actually matches what the model can legitimately claim to predict |
| RGB | FID | Secondary/sanity check only — explicitly **not** primary evidence of correctness, since it measures distributional plausibility, not per-pixel correctness |
| Calibration | Sparsification curve (error vs. fraction of highest-uncertainty pixels discarded), with an oracle and a random-rejection control curve | Primary calibration evidence — directly operationalizes "minimizes hallucination" as a measurable, reportable curve |
| Calibration | Expected Calibration Error (ECE), with bootstrap confidence interval | Secondary — known to be statistically noisy at hackathon-scale held-out sets, reported with its own CI rather than as a bare number |
| Hallucination | Off-manifold rate (high-confidence pixels whose dominant-mode color falls outside a pre-registered tolerance of the true class statistic) | Quantitative hallucination metric, confidence threshold fixed *before* test-set evaluation to avoid circularity |
| Latency | ms/tile, by precision/export configuration | Direct evidence for the low-inference-time BAH criterion |

**Statistical discipline:** ≥5 random seeds per headline number; block-bootstrap over tiles (not pixels) for confidence intervals, since within-tile pixels are spatially correlated; paired permutation tests with Holm-Bonferroni correction across the full baseline set before declaring any comparison significant; a pre-registered power analysis to confirm the held-out tile count can detect the smallest effect size considered meaningful.

**Baselines compared against:** deterministic regression (same backbone, single-head — isolates the effect of the density-estimation reformulation itself), Pix2Pix, CycleGAN, SPADE, Palette, conditional diffusion — each chosen to test a specific hypothesis or represent a specific weakness (mode-seeking adversarial loss, physically unjustified cycle-consistency, externally-supplied semantic conditioning, sampling-based latency cost).

**Falsification condition, stated explicitly:** if VARNA's off-manifold rate, per-class color accuracy, and sparsification AUC are not statistically distinguishable from the same-backbone deterministic-regression baseline after seed-averaging and multiple-comparison correction, that result directly falsifies the claim that the density-estimation reformulation provides a measurable practical benefit — regardless of the theoretical argument's validity.

---

## 9. Engineering Extensions — What Survived Review

Five candidate engineering extensions were proposed and individually reviewed against a fixed rule (which BAH criterion does it improve, by how much, at what cost, can the system work without it, is the complexity justified). Two review passes reached consistent verdicts:

| Extension | Verdict | Why |
|---|---|---|
| **Adaptive Inference Engine** (early-exit, confidence-gated decoding, hierarchical inference) | **Removed** | No iterative compute path exists to adapt (SR is already single-pass; mixture decode is already O(H×W×K×3), negligible next to the backbone). Worse, content-gated computation risks giving differently-processed tiles inconsistent calibration, contaminating the variance decomposition for reasons unrelated to genuine physical ambiguity — a direct violation of the brief's own instruction to reject anything introducing probabilistic bias. |
| **Confidence-driven Analyst Mode** | **Retained**, minimal | A stateless sort/percentile-index over the variance fields the model already computes — zero new inference, genuinely operationally useful (surfaces which tiles need human review), scoped as a query-pattern (percentile threshold), not an exhaustive ranked list, to remain meaningful at million-tile scale. |
| **Probabilistic Interpretation Layer / Explainability Layer** | **Retained, unconditionally** | Dominant hypothesis, secondary hypothesis, and the variance decomposition are *already computed* in the one forward pass — `topk(2)` instead of `argmax` is the only code change. No GradCAM/SHAP/attention-map machinery needed, because interpretability here is a property of the output representation itself, which is the entire reason this formulation was adopted over deterministic regression. |
| **Deployment Optimization** | **Retained**, scoped carefully | ONNX export, TensorRT, FP16 backbone with a **forced-FP32 decode boundary** (the export graph is deliberately split at the backbone/decode boundary, since log-sum-exp/sigmoid-difference ops aren't always efficiently fused by graph optimizers), async tile prefetch, host/GPU decode overlap. INT8 retained only as a validated stretch goal, never the baseline demo path (a botched INT8 calibration silently degrading temperature accuracy is worse for a live demo than an honest FP16 number). Multi-GPU **pipeline/model parallelism explicitly rejected**; if scaled beyond one GPU, only **data parallelism** (N independent copies of the same pipeline) is justified, because the formulation estimates p(RGB|T_B) independently per pixel — there is nothing to split. |
| **Live Benchmark / Mission Demonstration** | **Retained**, restructured | A five-step, ~105-second scripted demonstration (well under the 2-minute requirement): (1) show the deterministic baseline producing an off-distribution blended color at a known thermal-class-confusion pixel, (2) show VARNA's mixture resolving the same pixel into two named, physically real hypotheses, (3) show the confidence-ranked queue surfacing genuinely ambiguous regions on a full scene, (4) a live-timed single forward pass proving latency, (5) a physics-consistency residual overlay reusing that same forward pass's SR output. Every step but one (the single live forward pass) reuses a pre-computed, cached artifact, minimizing live-demo failure surface. |

**One unifying design fact:** the percentile-variance threshold gating both the analyst triage index and the explanation-record storage policy is the *same* threshold, computed once — not two separately-tuned cutoffs. This is the clearest instance of the project's standing rule: if two components can be merged, merge them.

---

## 10. Deployment

| Scope | Configuration |
|---|---|
| **BAH 2026 prototype** | Single GPU, T4/RTX-3060-class, 8–16GB VRAM. Fused ONNX/TensorRT graph for the shared backbone (FP16), decode submodule forced to FP32 as a separate, explicit export boundary. Async tile prefetch overlapping I/O with GPU compute; host/GPU decode overlap with the next tile's backbone dispatch. |
| **Quantization validation** | Not just mean-color/RMSE parity — explicitly extended to NLL and sparsification-curve parity between quantized and FP32 models, since quantization can preserve mean color while flattening sharp mixture components and silently distorting calibration. ONNX/TensorRT numerical-parity checks specifically include near-degenerate pixels (s_k near the variance floor ε), the numerically fragile region identified in the stability analysis. |
| **Future multi-GPU / ISRO cluster** | Data-parallel replication only — N identical copies of the single-GPU pipeline, no cross-GPU communication except final result aggregation, a direct consequence of the formulation's per-pixel independence. Throughput scales linearly in GPU count, with no communication overhead to erode that scaling. Storage at scale: quantized/compressed per-tile outputs, explanation records stored only above the same percentile-variance threshold used for analyst triage. Reproducibility discipline (config+git+W&B+split-manifest) becomes more, not less, important at cluster scale to prevent silent cross-node drift. |

---

## 11. Repository and Tech Stack

```
varna/
├── configs/                 # data.yaml, stage1_sr.yaml, stage2_color.yaml, inference.yaml
├── data/{raw,interim,processed,splits}/
├── varna/
│   ├── calibration/planck.py            # Stage 0
│   ├── datasets/{landsat9_dataset,degradation}.py
│   ├── models/{backbone,sr_head,mixture_head}.py
│   ├── losses/{sr_losses,mixture_losses}.py     # NLL, log-sum-exp stable
│   ├── train/{train_stage1,train_stage2,finetune_joint}.py
│   ├── inference/{pipeline,export_onnx,triage}.py
│   └── eval/{metrics,report}.py         # PSNR/SSIM/FID/sparsification/per-class accuracy
├── serving/{app.py,Dockerfile}          # FastAPI + ONNX Runtime/TensorRT
├── demo/streamlit_app.py                # five-step mission demonstration
└── scripts/{download_landsat9,run_full_eval.sh}
```

| Component | Choice | Why |
|---|---|---|
| Framework | PyTorch, standard layers | Team velocity over a 10–15% speed gain from custom CUDA; well-trodden PyTorch→ONNX→TensorRT path |
| Data access | Google Earth Engine Python API | Removes the single biggest time sink (data wrangling, manual co-registration) from the critical path |
| Weak labels | *(none — removed with the rejected classifier module, §3)* | — |
| Experiment tracking | Weights & Biases (free tier) | Live loss curves and calibration plots are persuasive evidence of rigor under judge Q&A |
| Serving | FastAPI + ONNX Runtime / TensorRT | Good-enough latency, far less engineering risk than a custom C++ server |

**Single mono-repo, config-driven, one environment for the whole team** — the cost of cross-repo version drift between engineers is far more dangerous than the modest loss of "clean separation" under a hard deadline.

---

## 12. Schedule

| Week | Deliverable |
|---|---|
| 1 | Data pipeline end-to-end; calibration unit-tested |
| 2 | SR head trained and evaluated (PSNR/SSIM numbers locked) |
| 3 | Mixture head trained; stability fixes verified; calibration sweep; full eval report generated; ONNX export validated |
| 4 (pre-finale buffer) | TensorRT FP16 build; FastAPI serving; five-step demo built and rehearsed; slide deck built directly from `eval/report.py` output |
| Finale (36h) | Joint fine-tune (stretch goal); INT8 attempt (stretch goal); live demo rehearsal; Q&A prep |

---

## 13. Honest Limitations (Stated, Not Hidden)

- Cannot and does not claim to recover true spectral reflectance from thermal radiance alone — an information-theoretic impossibility given single-channel input, not an engineering shortfall.
- Color outputs are class-conditional statistical estimates, accurate at land-cover-category granularity, not validated for individual-object material identification.
- Night-time RGB has no real ground truth available (OLI is passive, produces no signal without sunlight) — excluded from quantitative evaluation entirely, addressed only as future work.
- Mixed-pixel (sub-resolution heterogeneity) tiles have a genuinely ill-defined "true" RGB at the pixel level — a reporting category, not a model failure, and not penalized as one.
- Geographic/seasonal/sensor distribution shift degrades calibration quality; confidence estimates require re-calibration outside the training distribution's representativeness.
- Single-sensor (Landsat 9), single-country (India) training and evaluation domain for BAH 2026 — generalization beyond this is future work, not a current claim.

---

## 14. One-Paragraph Summary

Thermal radiance and visible-light reflectance are governed by different physical processes — emission versus reflection — so a single thermal measurement is consistent with more than one physically real surface color. Every prior image-translation approach either ignores this fact, producing a single confident but potentially invalid color, or represents it only implicitly through expensive repeated sampling. VARNA instead predicts the small set of physically plausible colors directly, in one forward pass, via a discretized logistic mixture — reports the most likely one (the dominant-mode color, not a blended mixture mean), and reports how confident it is and why, separating uncertainty into "this pixel is genuinely ambiguous between materials" and "this pixel's most likely material is itself noisy." Every engineering decision in this project — the rejected classifier module, the rejected unrolled SR, the rejected calibration loss, the rejected perceptual loss, the rejected adaptive inference engine, the rejected multi-GPU pipeline parallelism — exists to protect this one scientific contribution from unjustified complexity, on a single GPU, within a four-week-plus-36-hour timeline.
