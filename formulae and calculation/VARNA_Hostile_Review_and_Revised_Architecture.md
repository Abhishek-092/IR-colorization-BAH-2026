# VARNA — Hostile Technical Review (BAH 2026 Internal Funding Screen) and Revised Architecture

**Reviewing body:** Senior ISRO SAC Scientist, Remote Sensing Scientist, Thermal Imaging Physicist, Computer Vision Researcher, HPC Engineer, IEEE TGRS Reviewer, BAH 2026 Technical Evaluator.
**Mandate:** find every reason to reject. Survivors only.

---

## PART A — Findings

### Finding 1 — REJECT: Stage 2 as a supervised land-cover classifier trained on an external dataset

**Why it fails:**
This is the single largest risk in the entire proposal, and it is a risk of *scope*, not of math. The proposal's Stage 2 ingests ESA WorldCover, reprojects it, builds a 5-class taxonomy, and trains a classifier with cross-entropy against semantic land-cover labels. To any reviewer reading the official problem statement, this *is* a land-cover mapping module wearing a colorization costume. The explicit constraint says: if a module starts becoming a separate remote-sensing application, remove it. A supervised classifier trained against an external semantic land-cover product, with its own taxonomy, its own dataset dependency, and its own literature (land-cover classification is a mature, separate sub-field of remote sensing with its own benchmarks) is exactly that. A panel will ask: "Why does a thermal colorization system need a land-cover classification subsystem with external ground truth?" There is no defensible one-sentence answer to that question, and a four-member team cannot afford a module whose justification requires a paragraph.

There is a second, independent problem: **dataset assumption risk**. WorldCover is 10 m resolution; reprojecting/aggregating to a 100 m Landsat grid introduces majority-vote label noise at every tile boundary, and there is no quantified registration-error budget anywhere in the proposal. A four-week team does not have time to validate this dependency, and an IEEE TGRS reviewer would immediately ask for a registration-error analysis the team cannot produce in time.

**What replaces it:**
A **self-supervised radiometric vector-quantization (VQ) layer** operating only on features the SR network has already learned from the thermal signal itself — no external dataset, no semantic taxonomy, no land-cover labels of any kind. A small learned codebook (K = 6–8 vectors) quantizes each pixel's backbone feature vector to its nearest codebook entry, trained end-to-end with a standard VQ commitment loss (as in VQ-VAE — a single, well-understood, cheap auxiliary loss, not a new sub-system). The codebook entries are never given semantic names; they are simply "radiometric cluster 1...K," discovered purely from thermal statistics.

**Why this is scientifically stronger, not just safer:**
The mathematical role is identical to the proposal's class posterior — it still mediates color through a discrete latent so color is not regressed directly from radiance — but it removes the only external dataset dependency in the whole pipeline, removes an entire training phase (no cross-entropy training run, no weak-label generation pipeline, no smoothness-loss tuning), and is *more* defensible against the "this is land-cover mapping" objection because there is, by construction, no semantic label anywhere in the system. It also strengthens the novelty claim: the color decoder is conditioned on a representation discovered from the physics of the thermal signal alone, not borrowed from an existing external Earth-observation product.

---

### Finding 2 — REJECT: 3-iteration unrolled optimization SR with a hard data-consistency step

**Why it fails:**
Deep unrolling is the correct idea in a PhD-thesis sense and the wrong idea in a four-person, single-GPU, three-week sense. The architectural unroll requires the published TIRS nominal PSF to be accurate enough to serve as a hard constraint inside the network graph; if the assumed PSF is even moderately wrong, the data-consistency step injects a *bias*, not a correction, and the team has no time budget to validate the PSF assumption against real degraded/native pairs before training begins. It also means three sequential forward-backward passes through the same block at both train and inference time — three times the latency of a single-pass network, working directly against the proposal's own stated priority of low inference time per tile. A panel familiar with model-based deep learning will ask how the PSF mismatch was characterized; the honest answer ("we used the nominal published value and did not validate it") is a weak position to defend live.

**What replaces it:**
A **single-pass residual CNN** (same small backbone, no iteration loop) trained with the *same* physical knowledge expressed as a **soft auxiliary loss** rather than a hard architectural constraint: re-degrade the network's output with the known PSF + downsample operator and penalize its distance from the actual 200 m input. This is mathematically the same physical idea (the model is still required to produce an output consistent with the known degradation), but it is now a loss term the network can trade off against noisy or imperfect PSF assumptions, rather than a hard constraint baked into the architecture.

**Why this is scientifically stronger:**
A soft penalty degrades gracefully if the PSF estimate is imperfect; a hard architectural constraint does not. The network is also now strictly one forward pass — a direct, measurable improvement to the headline "low inference time" metric, with no loss of physical grounding, since the same physics still appears explicitly in the loss function and can be shown to the judges as a named term in the loss equation.

---

### Finding 3 — REJECT: Full heteroscedastic covariance prediction + a fourth dedicated calibration loss (ECE-style)

**Why it fails:**
Predicting a per-pixel covariance and then training a *separate* calibration objective on top of it is two research-level claims stacked on top of each other. Expected Calibration Error is a statistically meaningful quantity only with enough held-out samples per confidence bin; on a four-region, hackathon-scale held-out set, a reported ECE number is likely to be noisy enough that a sharp reviewer will ask for confidence intervals on the calibration metric itself — a question the team will not be able to answer. Optimizing a dedicated calibration loss alongside the NLL loss also adds a fourth hyperparameter (λ) to tune under time pressure, for a claim ("our uncertainty is formally calibrated") that is disproportionately strong relative to what three weeks of data can actually support.

**What replaces it:**
Keep the heteroscedastic mean+variance output (this part is cheap — one extra output channel and a standard Gaussian NLL loss, not a separate sub-system) but **drop the dedicated calibration loss entirely**. Calibration quality is *measured*, post hoc, via a sparsification plot (error vs. fraction of highest-uncertainty pixels removed) — a plot that is intuitive to explain to judges in one sentence ("when we discard the pixels we're least sure about, error goes down — that's evidence the uncertainty is meaningful") and requires no additional training objective.

**Why this is scientifically stronger:**
It makes exactly one claim — "our predicted uncertainty correlates with actual error" — and that claim is directly, visibly demonstrated by a single plot, rather than asserting formal statistical calibration that the dataset size cannot support. A smaller, fully defensible claim survives hostile questioning; a larger, fragile claim does not.

---

### Finding 4 — REJECT: Pretrained ImageNet/VGG perceptual loss term

**Why it fails:**
VGG features were learned on natural RGB photographs. Using them to score synthesized pseudo-RGB derived from thermal radiance introduces an unexamined domain-transfer assumption: there is no stated reason to believe ImageNet-trained texture statistics are the right perceptual prior for this signal. It is also an external pretrained dependency that adds engineering surface area (model download, version pinning, feature-layer choice) for a loss term carrying a 0.05 weight — i.e., by the team's own proposed weighting, a marginal contribution.

**What replaces it:**
Nothing. Remove it. The NLL loss on the heteroscedastic RGB output, combined with the SR network's own structural fidelity, is sufficient. If texture sharpness is visibly lacking in early results, a structural similarity (SSIM) term computed directly on the predicted vs. true RGB — no external network required — is the fallback, and only if time remains.

**Why this is scientifically stronger:**
Removing an unjustified external dependency and an unexamined domain-transfer assumption is a strict improvement in scientific hygiene, and it deletes one more hyperparameter and one more thing that can silently fail (e.g., wrong VGG layer choice, version mismatch) during the 36-hour finale.

---

### Finding 5 — REJECT: Adaptive tile-scheduling / triage system

**Why it fails:**
The triage mechanism (entropy/variance routing to different iteration counts) was designed around the now-removed 3-iteration unrolled SR. With a single-pass SR network (Finding 2), the latency gain available from triage is far smaller, because there is no longer a multi-iteration loop to skip. What remains is added inference-path complexity — a routing decision, two code paths to validate, two sets of edge cases to demo — for a benefit that will not even be visible in a live demo processing a handful of tiles. This is close to the textbook definition of a module that exists to sound sophisticated in the writeup without a measurable demo-time benefit.

**What replaces it:**
Nothing, for the hackathon build. A single-pass network is already fast; the latency story is carried entirely by Finding 2's simplification. Triage is retained only as a **named future-work item** ("tile-level compute routing once a multi-resolution model variant exists"), which costs nothing to mention and nothing to implement.

**Why this is scientifically stronger:**
One fewer code path means one fewer thing that breaks live, and the team's engineering hours go toward validating the core pipeline instead of an optimization whose benefit is no longer architecturally large enough to be worth the complexity.

---

### Finding 6 — REJECT (as currently framed): hard binary abstention ("no color produced")

**Why it fails:**
The proposal's abstention mechanism *withholds* a color value entirely for low-confidence pixels, falling back to grayscale. In a live demo, if the abstention threshold is even moderately conservative, large fractions of a displayed image may revert to gray — which a judge will read as "the system often fails to do the thing it was built to do," not as "the system is appropriately humble." This inverts the intended impression.

**What replaces it:**
**Always produce a color estimate** (the mean of the predicted Gaussian); never suppress output. Uncertainty is communicated exclusively through the confidence overlay (already part of the proposal) shown alongside the RGB image, not by altering the RGB image itself.

**Why this is scientifically stronger:**
The scientific claim — "we know where we are likely wrong" — is preserved exactly, and is in fact *more* visible (a continuous confidence map carries more information than a binary colored/grayed-out decision), while removing the demo risk of an image that looks broken.

---

### Finding 7 — SURVIVES, with a presentation-risk note: physics-based brightness-temperature calibration (Stage 0)

**Assessment:** This stage is a closed-form physical computation (inverse Planck function with published sensor constants), costs nothing to implement, and is unambiguously inside the official problem statement (TIR super-resolution requires radiometrically correct input). No rejection grounds found.

**Presentation risk flagged:** the proposal must consistently call this quantity **brightness temperature**, never "land surface temperature" — LST formally requires an emissivity correction the system does not perform, and a thermal-imaging physicist on the panel will immediately probe this distinction. This is a one-word fix, not a redesign, but it must be enforced consistently across the slide deck and the code's variable names (a panel member skimming `LST` in a script while a slide claims "we do not perform LST retrieval" is exactly the kind of inconsistency that erodes credibility fastest).

---

### Finding 8 — SURVIVES: physics-informed degradation-consistency loss (revised, soft form)

**Assessment:** As revised in Finding 2, this is simple (one re-blur-and-compare operation), directly addresses "physics-informed modelling," is cheap to compute, and is trivially explainable to judges in one sentence with a clear loss-equation slide. No rejection grounds found.

---

### Finding 9 — Novelty re-assessment

The original proposal's novelty claim rested partly on "class-conditioned colorization." A hostile panel could reasonably ask whether this is just classification-then-lookup dressed up as generation. **With Finding 1's revision, this objection is substantially defused**: there is no external semantic class anywhere in the system, no land-cover taxonomy, no land-cover dataset. The novelty claim becomes: *color is mediated by a discrete latent vocabulary discovered purely from the thermal signal itself, with calibrated-by-construction uncertainty surfaced as a first-class output* — a claim that is both narrower (easier to defend) and harder to dismiss as "just classification," precisely because there is no classification task being solved anywhere in the pipeline.

---

### Finding 10 — Engineering feasibility re-assessment (4 engineers, single GPU, BAH 2026 timeline)

The revised pipeline has **three** trainable stages instead of five, **three** loss terms instead of seven-plus, and **one** external dataset dependency removed entirely. This maps cleanly to a four-person team: (1) data pipeline + calibration, (2) SR network + degradation loss, (3) VQ layer + color decoder, (4) export/serving/eval/demo. No engineer owns a module whose validation depends on an external dataset's label quality, and no engineer owns a module (triage, calibration loss, perceptual loss) whose primary justification is no longer present in the architecture. This is the single biggest feasibility improvement from the review.

---

## PART B — Revised Architecture

**Name:** VARNA (unchanged — same project, same core claim: color is recoverable only through a class-like latent, not directly from radiance; the latent is now self-discovered rather than externally supervised, which is a strengthening of the original idea, not a departure from it).

### Pipeline (three stages, one forward pass each — no iteration loops, no routing logic)

```
raw 200 m TIR
   │
   ▼
[Stage 0] Radiometric calibration (closed-form, Planck inversion)
   │  → brightness temperature, T_B (200 m)
   ▼
[Stage 1] Single-pass physics-informed SR network
   │  loss: L1(T_B_SR, T_B_true) + degradation-consistency penalty
   │  → T_B_SR (100 m)  ───────────────────────────► deliverable #1 (Super-Resolved TIR)
   ▼
[Stage 2] Self-supervised radiometric VQ layer (K=6–8 codebook, learned jointly)
   │  loss: VQ commitment + codebook loss
   │  → discrete radiometric latent per pixel (no semantic meaning, no external labels)
   ▼
[Stage 3] Heteroscedastic color decoder (mean + variance Gaussian, conditioned on VQ code + local features)
   │  loss: Gaussian NLL
   │  → RGB mean (always produced) + per-pixel uncertainty
   ▼
output: {SR-TIR (100 m), RGB (100 m), confidence overlay} ──► deliverable #2 (RGB), evaluated via sparsification plot
```

### What was removed, and why it does not weaken the proposal
- External land-cover dataset and taxonomy → removed; replaced by self-supervised codebook (Finding 1). **Net effect: strengthens scope-compliance and novelty.**
- Multi-iteration unrolled optimization → removed; replaced by single-pass network with a soft physics loss (Finding 2). **Net effect: faster inference, same physical grounding, more robust to PSF misestimation.**
- Dedicated calibration loss / ECE claims → removed; replaced by post-hoc sparsification evaluation (Finding 3). **Net effect: smaller, fully defensible claim.**
- VGG perceptual loss → removed entirely (Finding 4). **Net effect: one fewer external dependency, one fewer point of silent failure.**
- Adaptive tile triage → removed from the hackathon build, retained only as future work (Finding 5). **Net effect: one fewer live code path; latency story now carried by single-pass architecture alone.**
- Hard binary abstention → softened to always-color-plus-overlay (Finding 6). **Net effect: removes the single largest live-demo visual risk.**

### What survived unchanged
- Closed-form radiometric calibration (Stage 0).
- The core scientific claim of the project: color cannot be regressed directly from thermal radiance; it must be mediated through a class-like latent, with uncertainty exposed as a first-class output rather than hidden inside a confident wrong answer.
- Evaluation against true 100 m TIRS for Stage 1 (the only deliverable with genuine pixel-level ground truth) as the headline, hardest-to-dispute metric.

### Training/engineering load (4-person mapping)
| Engineer | Owns |
|---|---|
| 1 | Data pipeline, Stage 0 calibration, geographic splits |
| 2 | Stage 1 SR network + degradation loss |
| 3 | Stage 2 VQ layer + Stage 3 color decoder + NLL/sparsification |
| 4 | ONNX/TensorRT export, serving, demo, evaluation report |

### One-sentence answers the team should now be able to give under questioning
- *"Why does color depend on a learned cluster instead of direct regression?"* — because thermal and visible reflectance are governed by different physical processes, so a one-to-one mapping does not exist; the cluster is discovered purely from the thermal signal, with no external land-cover data.
- *"Why single-pass and not iterative refinement?"* — because the physical constraint is enforced as a loss term, not an architectural assumption, which is both faster and more robust to PSF uncertainty.
- *"How do you know the uncertainty map means anything?"* — because error measurably drops when the most uncertain pixels are excluded; shown directly on a sparsification plot, not asserted.
- *"Isn't this just land-cover classification?"* — no semantic label, no external dataset, and no land-cover taxonomy exists anywhere in the system; the latent is an unnamed, self-discovered radiometric code.

This is the version of VARNA that should go into the BAH 2026 funding review: fewer moving parts, no module whose defense requires more than two sentences, no claim larger than the dataset can support, and every remaining component traceable directly to one of the six permitted objectives in the official problem statement.
