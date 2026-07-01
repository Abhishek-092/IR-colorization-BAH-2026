# VARNA — ISRO Engineering Systems Design Review Board
**BAH 2026 Technical Screen — Operational-Scale Revision**
**Board composition (as convened):** Senior ISRO Systems Architect, Satellite Image Processing Engineer, HPC Engineer, Deployment Engineer, Remote Sensing Engineer, Computer Vision Engineer, Scientific Software Architect, BAH Technical Evaluator.

**Relationship to the prior engineering review:** this board reaches the same verdicts as the previous systems review on Features 1, 2, and 4, for the same reasons, and does not re-litigate them. This document exists to do two things the prior review did not: (a) analyze the architecture under the stated operational target — millions of tiles, not a single demo tile — including multi-GPU and future ISRO cluster deployment, and (b) redesign the demonstration around the five specific points this board was asked to prove, rather than the general comparison panel proposed previously. Feature 3 is carried forward unchanged in substance under its new name.

**Frozen, restated once:** conditional density estimation, discretized logistic mixture (K=6–8), NLL training objective, dominant-mode decode, within/between-mode variance decomposition, single shared backbone (~2.5M params) with a deterministic SR head and a mixture head. Nothing below touches any of this.

---

## FEATURE 1 — Adaptive Inference Engine

**Verdict: removed.** Reasoning unchanged from the prior review: the unrolled SR loop was already eliminated in favor of a single-pass residual CNN, and the mixture-decode arithmetic is O(H×W×K×3) — linear and negligible next to the backbone's convolutional cost. There is no iterative compute path left for confidence-gated decoding, early termination, or adaptive computation to act on.

**New consideration at million-tile scale:** one might expect adaptive computation to matter more at scale, since even small per-tile savings compound. This board explicitly checked that intuition and rejects it: at million-tile scale, the dominant lever is **embarrassingly parallel throughput** (more GPUs running the same fixed-cost forward pass), not per-tile adaptive savings on an already-cheap pass. Adaptive routing logic adds a *sequential dependency* (a routing decision before compute can proceed) that is specifically harmful to throughput-oriented batch parallelism — it is the wrong optimization target for this operational profile, not merely an unnecessary one. **Tile complexity estimation** survives only as a static, pre-computed *priority* tag for scheduling order (see Feature 4), never as a gate that changes how a tile is computed — gating computation by content risks exactly the "inconsistent uncertainty" failure mode the brief asks this board to check for: a tile processed on an abbreviated path could receive a systematically different calibration than one processed on the full path, contaminating the variance decomposition's interpretability (Feature 3) for reasons that have nothing to do with the tile's actual physical ambiguity. This is the calibration-bias argument the brief specifically asked this board to test for, and it is sufficient on its own to reject any content-gated computation path, independent of the throughput argument above.

---

## FEATURE 2 — Confidence-driven Analyst Mode

**Verdict: retained, scoped to ranking/tagging.** Unchanged in design from the prior review: a stateless sort over the already-computed total-variance and within/between-mode tag fields, costing nothing beyond a sort.

**New consideration at million-tile scale:** "automatic review queues" (named explicitly in this brief, not the prior one) must be scoped carefully here. At the scale of millions of tiles, presenting a flat ranked list is operationally useless — a human cannot review a million-tile-derived queue. The correct minimal design is **percentile-based triage, not exhaustive ranking**: store the per-tile aggregate variance as a sortable index field (not a full re-sort of the entire corpus per query), and let an analyst query "top N most ambiguous tiles in region/time-window X" against that index. This is a query-pattern decision, not a new subsystem — it reuses the same per-tile scalar already proposed, indexed rather than globally sorted, which is the only change required to make the feature meaningful at operational scale rather than demo scale.

---

## FEATURE 3 — Probabilistic Interpretation Layer
*(named "Explainability Layer" in the prior review; substance unchanged, retained here under the new name for continuity with this board's terminology)*

**Verdict: retained, unconditionally.** As established previously: dominant hypothesis (μ_{k*}), secondary hypothesis (μ_{k2}, via `topk(2)` instead of `argmax` on the existing π_k vector), and the within/between-mode variance decomposition with entropy(π) are all already-computed outputs of the single forward pass. No additional network, no GradCAM/SHAP/attention-map machinery — precisely because the brief is correct that interpretability here is a property of the *output representation* (a distribution with named components), not something that needs to be reverse-engineered out of an opaque model after the fact, which is the whole reason this formulation was adopted over deterministic regression in the first place (formulation review §1–2).

**New consideration at million-tile scale:** the explanation record (dominant + secondary hypothesis, two variance terms, entropy) must be stored or discarded per tile — at a million tiles, storing the full record for every tile is a real, bounded storage decision, not a free one. Recommended: store the explanation record only for tiles whose total variance exceeds a fixed percentile threshold (the same threshold used for Feature 2's triage index), discarding it for confidently-resolved tiles where the dominant hypothesis is, by construction, almost certainly correct. This keeps the interpretability layer's storage cost proportional to genuine ambiguity in the corpus, not to corpus size.

---

## FEATURE 4 — Deployment Optimization

**Verdict: retained**, with the BAH-scope content unchanged from the prior review (ONNX/TensorRT export, FP16 backbone with FP32-forced decode, async tile prefetch, host/GPU decode overlap), **plus an explicit multi-GPU and future-cluster analysis**, which the prior review deferred.

### 4.1 Single-GPU (BAH 2026 prototype)
Unchanged: T4/RTX 3060-class, 8–16GB VRAM, FP16 backbone with FP32-forced decode boundary, fused ONNX/TensorRT graph for the backbone, async prefetch overlapping I/O with the previous batch's compute. This remains the only configuration the four-member team builds and validates within the hackathon timeline.

### 4.2 Multi-GPU deployment — what kind, and why it is the obvious answer given the mathematics
The frozen formulation estimates p(RGB | T_B) **independently per pixel** (engineering stability analysis §4) and the inference pipeline processes tiles independently of one another (no cross-tile state, no recurrent or sequential dependency between tiles). This is precisely the condition under which **data parallelism is the correct and only justified form of multi-GPU scaling** — replicate the identical single-GPU pipeline across N GPUs, partition the tile stream across them, and aggregate results. **Model parallelism or pipeline parallelism across GPUs is not justified and is explicitly rejected**, for the same reason it was rejected in the prior review: the model is small (~3.4M parameters total) and fits comfortably on a single GPU with large margin; splitting a model that already fits on one device across multiple devices would introduce inter-GPU communication overhead for a model that has no need to be split. Given the mathematics — per-tile independence — the engineering conclusion is not a design choice but a direct consequence: **N GPUs running the same fused single-GPU graph independently, with no cross-GPU communication except final result aggregation, is the only architecture this formulation justifies.**

### 4.3 Throughput estimate at operational (million-tile) scale
Order-of-magnitude only, explicitly flagged as an estimate requiring empirical confirmation (per the scientific validation plan's Part 8 deployment-validation protocol, not asserted here as measured fact): a ~3.4M-parameter backbone-plus-heads network on a single T4-class GPU at FP16, on a 256×256 tile, is plausibly in the range of several hundred tiles/second given the network's small size relative to typical vision-model throughput benchmarks at this resolution. At that order of magnitude, one million tiles would be on the order of tens of minutes on a single GPU, and proportionally faster (linearly, given the independence argument in §4.2) across a small data-parallel cluster — e.g., roughly an order of magnitude faster across ten GPUs, since there is no communication overhead to erode that scaling. **This board does not certify a specific number**; it certifies that the *scaling law* is linear in GPU count, which is the operationally relevant fact for capacity planning, and recommends the BAH prototype measure its actual single-GPU throughput (Part 8 of the validation plan) and extrapolate linearly rather than re-deriving a new estimate at deployment time.

### 4.4 Future ISRO cluster deployment
Three considerations specific to a future operational (non-BAH) deployment, stated as forward-looking design constraints, not BAH 2026 deliverables:
- **Storage efficiency at scale:** per-tile outputs (SR-TIR, RGB, confidence/explanation record) should be stored in a quantized, compressed form for full-corpus archival (e.g., 8-bit RGB, float16 or quantized variance fields), with the percentile-gated explanation-record storage policy from Feature 3 reducing the interpretability layer's footprint specifically.
- **Maintainability/reproducibility at scale:** the existing config-driven, single-repo, git-hash-and-W&B-tracked reproducibility discipline (implementation plan §1, §7 of the validation plan) becomes more, not less, important at cluster scale, since silent drift across nodes (different ONNX/TensorRT engine builds, different CUDA/driver versions per node) is a real operational risk that a single-GPU hackathon prototype cannot surface; this is flagged as a deployment-readiness item for any future cluster rollout, explicitly out of BAH 2026 scope.
- **CPU/GPU overlap and memory bandwidth at scale:** the same async-prefetch and host/GPU-decode-overlap design from the single-GPU case applies per-node in a cluster, unchanged — there is nothing about cluster scale that changes the per-node engineering, only the orchestration layer above it (tile-stream partitioning, result aggregation), which is infrastructure, not a VARNA-specific engineering decision, and is therefore explicitly out of this board's scope.

### 4.5 Energy consumption
Not separately measured for BAH 2026 (no power-telemetry requirement on typical hackathon hardware), but qualitatively bounded by the same throughput argument: a small, single-forward-pass network at FP16 with no iterative or sampling-based inference (the explicit reason diffusion-class alternatives were rejected throughout this project's review history) has materially lower energy-per-tile than any sampling-based multimodal alternative, by the same multi-step-vs-single-step argument already established for latency.

---

## FEATURE 5 — Mission Demonstration
*(redesigned from the prior review's seven-panel comparison interface; this board was asked to prove five specific points, not run a general side-by-side, and the demo is redesigned around that requirement)*

### 1. Technical Review
**Should it exist? Yes**, but restructured: the prior review's panel (Original → Existing Method → VARNA → Difference → Uncertainty → Latency → Physics Consistency) is a reasonable *information layout*; this board's brief asks for a *scientific argument*, delivered in under two minutes, structured as five sequential claims with evidence, not seven simultaneous panels. The content is largely the same underlying artifacts; the structure changes from a comparison grid to a five-step demonstration script.

### 2. Scientific Justification
Each of the five required points maps directly to one or more official BAH criteria:

| Demonstration point | BAH criterion(s) addressed |
|---|---|
| 1. Why deterministic colorization is fundamentally incorrect | Hallucination reduction (motivates why it's measured at all) |
| 2. How VARNA models ambiguity | Physics-informed modelling, hallucination reduction |
| 3. Why uncertainty improves trustworthiness | Hallucination reduction, visual hallucination inspection |
| 4. Why the system remains computationally efficient | Low inference time per tile |
| 5. How physics consistency is preserved | Physics-informed modelling, structural integrity |

### 3. Engineering Design — the five-step script

**Step 1 — Why deterministic colorization is fundamentally incorrect (≈20s).**
Show a single, pre-selected pixel at a known thermal-class boundary (e.g., a documented dry-asphalt/dark-basaltic-soil pair from the formulation review's own example) where two materials share near-identical T_B but different true RGB. Show, side by side: the deterministic-regression baseline's single output color at that pixel, and the *true* OLI colors of both materials. The deterministic output should visibly sit between the two true colors — a color belonging to neither material. This is a pre-computed, cached comparison (the baseline model is not run live), chosen specifically because it makes the formulation review's §1 argument (the posterior-mean failure mode) visible rather than asserted.

**Step 2 — How VARNA models ambiguity (≈20s).**
At the same pixel, show VARNA's full output: π_k bar chart (mixture weights), with the two dominant components' μ_k colors swatched directly against the true colors of the two materials from Step 1. This is the Feature-3 explanation record, already computed, re-used here rather than recomputed for the demo.

**Step 3 — Why uncertainty improves trustworthiness (≈25s).**
Show the Feature-2 confidence-ranked queue on a full held-out scene (the heterogeneous urban-rural fringe tile, per the implementation plan's demo design): the system surfaces its own least-confident regions, and a quick visual check confirms those flagged regions are, in fact, the materially ambiguous ones (boundaries, mixed pixels) — directly demonstrating that the uncertainty signal is *informative*, not decorative, which is the sparsification-curve argument from the validation plan made visible rather than reported as a number.

**Step 4 — Why the system remains computationally efficient (≈20s).**
A live timer around one live VARNA forward pass on a fresh, previously unseen tile (the one live computation in the entire demo, per the prior review's failure-surface-minimization principle), displayed next to the *cached* latency figure for the diffusion-class baseline from the validation plan's baseline comparison (Part 3) — not re-run live, since multi-step sampling latency is already established and re-demonstrating it live only adds risk for no new information.

**Step 5 — How physics consistency is preserved (≈20s).**
Re-degrade VARNA's live SR output (from Step 4's same forward pass) with the known PSF + downsample operator and overlay the residual against the actual 200m input — the same physics-consistency panel from the prior review's Feature 5 design, one downsample-and-difference operation, no new computation.

**Total: ≈105 seconds**, under the two-minute requirement with margin, by design — every step reuses an already-computed artifact except the single live forward pass in Step 4, which doubles as both the efficiency demonstration and the source tile for Step 5's physics check.

### 4. Implementation Plan
One engineer, roughly a day and a half: select and freeze the Step 1/2 boundary-pixel example (requires inspecting the training data once to find a documented, clean thermal-class-confusion pair); wire the Step 3 queue view to the Feature 2 index; build the Step 4 timer and cached-baseline-latency lookup; build the Step 5 residual overlay (already speced in the prior review). Rehearsed as a fixed five-step script, not an open-ended interactive demo, to keep the under-two-minute guarantee robust to live-demo conditions.

### 5. Expected Impact
Not metric-improving; directly targets the brief's stated success condition (convince a technical committee in under two minutes) by structuring the demo as a five-claim scientific argument with evidence at each step, rather than a general-purpose comparison tool a judge has to interpret unassisted.

### 6. Trade-offs
A scripted five-step demo is less flexible than an open comparison panel if a judge wants to explore a different tile interactively — acceptable, because the brief explicitly asks for a demonstration that "resembles a scientific experiment, not a product showcase," and a fixed, pre-validated script is more robust under live-demo conditions (the same risk-minimization argument applied throughout this project's engineering reviews) than an interactive tool whose behavior on an arbitrary judge-chosen tile cannot be fully rehearsed. **Verdict: retained, restructured as a five-step scripted demonstration rather than the prior review's seven-panel comparison grid.**

---

## SYSTEM INTEGRATION

The data-flow diagram from the prior review is unchanged in structure (one shared backbone, two heads, one forward pass per tile, with Features 2/3/5 all reading off the decode step's output edge and Feature 4 wrapping the whole pipeline as deployment infrastructure). Two integration points are new in this revision:

- **Feature 4 ↔ Feature 2/3 storage policy:** the percentile-variance threshold used to decide which tiles get a stored explanation record (Feature 3, million-tile storage policy) and which tiles enter the analyst triage index (Feature 2) is the **same threshold**, computed once per deployment batch from the same already-computed variance field — not two separate thresholds requiring separate tuning. This is a direct instance of the "if a capability naturally emerges from an existing subsystem, remove the redundant module" instruction: a second threshold would be redundant machinery for the same underlying signal.
- **Feature 4 ↔ multi-GPU:** because the formulation is per-pixel-independent (engineering stability analysis §4) and the pipeline is per-tile-independent by construction, the cluster-scale architecture (§4.2–4.4) requires no new subsystem at all — it is N copies of the existing single-GPU pipeline plus an orchestration layer outside VARNA's own scope. This is the clearest instance in this entire review of the brief's framing being literally true: "given the mathematics, this is the obvious engineering implementation."

---

## FINAL PRODUCTION ARCHITECTURE OF VARNA (BAH 2026, operational-scale-aware)

**Mathematical core:** unchanged — Stage 0 calibration, shared backbone, deterministic SR head, discretized-logistic-mixture color head, dominant-mode decode, variance decomposition.

**Engineering layer, final verdicts:**
1. ~~Adaptive Inference Engine~~ — **removed**, now additionally on calibration-integrity grounds (content-gated computation risks contaminating the variance decomposition's meaning, not only wasting engineering effort on a non-bottleneck).
2. **Confidence-driven Analyst Mode** — retained, scoped to a percentile-indexed triage query, not an exhaustive ranked list, to remain meaningful at million-tile scale.
3. **Probabilistic Interpretation Layer** — retained unconditionally, zero additional inference cost, storage scoped by the same percentile threshold as Feature 2.
4. **Deployment Optimization** — retained: single-GPU FP16/FP32-split pipeline for BAH 2026; **data-parallel (not model- or pipeline-parallel) replication** as the only justified path to multi-GPU and future ISRO cluster scale, a direct consequence of the formulation's per-pixel independence.
5. **Mission Demonstration** — retained, restructured as a five-step, ≈105-second scripted scientific argument, with exactly one live computation (a single VARNA forward pass) serving double duty as both the efficiency proof and the physics-consistency source.

**What remains deliberately absent, restated:** no second trained model running live, no content-gated or adaptive compute path inside the forward pass, no model/pipeline parallelism, no UI or analyst-facing behavior that is not a direct, threshold-consistent readout of the decode step's existing output fields. The system that ships to BAH 2026 and the system that would, in principle, scale to an operational million-tile ISRO deployment are **the same single-GPU pipeline**, differing only in how many independent copies of it run and how their outputs are aggregated — which is the intended sense in which this architecture is inevitable given the mathematics, not merely minimal by preference.
