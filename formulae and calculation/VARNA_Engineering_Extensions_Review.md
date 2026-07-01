# VARNA — Engineering Systems Review and Final Production Architecture
**Prepared by: ISRO Systems Architect / Deployment Engineer, BAH 2026 Technical Screen**
**Scope constraint honored throughout:** the mathematical formulation — conditional density estimation, discretized logistic mixture, NLL training objective, dominant-mode decode, within/between-mode variance decomposition — is frozen and untouched below. Every section is a judgment about *engineering around* that fixed object, not a modification of it. Where a proposed feature would require a new trained component, that is treated as a disqualifying property, not a design detail to work around.

**Official BAH evaluation criteria, restated for reference (per the original problem statement and the TIR colorization framework, §8):** PSNR/SSIM/BT-RMSE (SR), per-class color accuracy and FID (colorization, FID secondary), hallucination reduction (quantitative, sparsification/off-manifold rate), physics-informed modelling, structural integrity, low inference time per tile. Every feature below is scored against this fixed list, not against general AI-product appeal.

---

## FEATURE 1 — Adaptive Inference Engine

### 1. Technical Review
**Should it exist? No, not as an independent subsystem.**

The premise of an adaptive inference engine — early-exit, hierarchical inference, confidence-gated decoding, selective probability evaluation — assumes there is a large, variable, *iterative* compute cost to adapt. That assumption was true of an earlier VARNA revision (the unrolled, multi-iteration SR block). It is no longer true: the hostile architectural review replaced the unrolled SR with a **single-pass residual CNN**, specifically because three sequential forward-backward passes worked against the low-latency requirement. The frozen formulation's color head is likewise a single closed-form forward pass — no sampling, no iteration. There is, by construction, no loop left to early-exit out of, and nothing to hierarchically defer.

Independently, the engineering stability analysis already established that the mixture-decode arithmetic (softmax, log-sum-exp, CDF difference, variance decomposition) costs O(H×W×K×3) — linear, negligible, and dominated in practice by the backbone's convolutional FLOPs (~3.4M parameters total, per the implementation plan's sizing table). **Selective probability evaluation** (skipping low-weight components) and **confidence-gated decoding** (skipping the decode step on "easy" tiles) would therefore be optimizing a part of the pipeline that is already not the bottleneck — a textbook case of engineering effort misallocated relative to where the latency budget actually goes.

### 2. Scientific Justification
No official BAH criterion is improved. "Low inference time per tile" is the only criterion this feature category could plausibly touch, and the analysis above shows the targeted cost is not the dominant cost. There is no PSNR/SSIM/FID/hallucination/structural-integrity/physics-consistency benefit from any of the listed mechanisms.

### 3. Engineering Design
Not retained as a subsystem. (See §6, System Integration, for the one component of this idea — tile-priority scheduling — that survives, relocated into Feature 4.)

### 4. Implementation Plan
None — explicitly not built.

### 5. Expected Impact
None quantifiable, because the targeted cost component is not the latency-dominant one. Building speculative early-exit infrastructure against an already-cheap decode step would not produce a measurable wall-clock improvement worth reporting to judges.

### 6. Trade-offs
Rejecting this avoids: a new conditional-routing control path with its own failure modes (misrouted tiles, threshold-tuning burden competing with the K-sweep and stability-fix tuning order already established), additional code surface for a four-engineer team to validate under time pressure, and a feature that — if built — would need its own ablation to even prove it helped, consuming validation effort better spent on the hypotheses already defined in the validation plan. **Verdict: removed.**

---

## FEATURE 2 — Confidence-driven Analyst Mode

### 1. Technical Review
**Should it exist? Yes, in a minimal form.**

Uncertainty should actively influence workflow, not merely be visualized — but the only version of "actively influence" that survives scrutiny is **passive prioritization**, not active intervention. The model already computes, per pixel, in one forward pass: total predictive variance, its within-mode/between-mode decomposition, and the entropy of π_k. Sorting tiles or regions by these already-computed quantities to produce an inspection priority queue costs nothing beyond a sort operation. Anything beyond that — adaptive UI behavior, real-time analyst-attention tracking, closed-loop active learning during the demo — is out of scope for a hackathon timeline and was already flagged as future work (active learning loops) in the original framework, not a BAH-2026 deliverable.

### 2. Scientific Justification
Directly operationalizes the **hallucination-reduction** and **visual hallucination inspection** criteria: instead of a judge or analyst having to manually find the heterogeneous, ambiguous regions where hallucination is most likely (as the implementation plan's demo design already does for the *single pre-loaded example tile*), this generalizes that same idea to *any* scene at full operational scale — the system surfaces where to look, rather than requiring an analyst to scan a full scene uniformly.

### 3. Engineering Design
- **Architecture:** a stateless post-processing step on the already-computed per-tile output tensors {π_k, μ_k, s_k}. No new model, no new training.
- **Data flow:** inference pipeline output → variance-decomposition fields (already produced by the frozen decode step, per the formulation's §4.5) → ranking module computes per-tile (or per-region, by spatial aggregation) total variance and a tag (`ambiguous-material` if between-mode dominates, `noisy-signal` if within-mode dominates, using the same distinction validated in the scientific validation plan's H4) → sorted list/heatmap served alongside the existing RGB/confidence outputs.
- **Algorithm:** O(N log N) sort over N tiles/regions per scene; no iteration over pixels beyond the already-computed aggregate statistics.
- **Runtime/memory:** negligible — a few extra scalar/array reductions per tile, computed from outputs that already exist in memory.

### 4. Implementation Plan
One engineer, roughly a day: extend `eval/metrics.py`/`inference/pipeline.py` to emit per-tile aggregate variance and the dominant-uncertainty-source tag; add a sort/rank endpoint to `serving/app.py`; surface as a ranked list (and a simple heatmap overlay, reusing the confidence-map payload already planned) in the Streamlit demo.

### 5. Expected Impact
Not a metric-improving feature in the PSNR/SSIM/FID sense — it does not change any reported accuracy number. Its value is operational: directly demonstrable in under a minute to a judge or analyst ("here are the five tiles in this scene the system itself flags as least trustworthy, and here is why for each one"), which is a stronger, more concrete demonstration of the hallucination-reduction claim than a static example.

### 6. Trade-offs
Adds one new lightweight serving endpoint and a small amount of UI surface. Acceptable because it is built entirely from fields the frozen formulation already produces — there is no new uncertainty to compute, only a new way to read out the existing decomposition. **Verdict: retained, scoped to ranking + tagging only.**

---

## FEATURE 3 — Explainability Layer

### 1. Technical Review
**Should it exist? Yes — this is the strongest-justified of the five features**, because it requires building literally nothing new. The brief's own framing ("determine whether the probabilistic formulation itself naturally produces explainable outputs") is answered directly by the formulation review's §4.5: a single forward pass already yields the dominant hypothesis (μ_{k*}), the full mixture {π_k, μ_k, s_k}, and the within/between-mode variance decomposition. A **secondary hypothesis** (μ_{k2}, the mean of the second-highest-weight component) is a one-line addition — `argsort` instead of `argmax` on the already-computed π_k vector, not a new inference path.

### 2. Scientific Justification
Directly improves **hallucination reduction** (made inspectable, not just measurable) and **physics-informed modelling** as evaluation criteria: a judge or analyst can be shown, for any pixel, *why* the model is uncertain (two competing physical hypotheses vs. one noisy hypothesis) rather than being told a single opaque confidence number. This is exactly the distinction the variance decomposition exists to make legible.

### 3. Engineering Design
- **Architecture:** extend the existing decode submodule's output schema (the FP32 boundary already established in the engineering stability analysis, §3c/§7) to additionally emit: secondary-component index k2, μ_{k2}, π_{k2}, and the entropy of {π_k}. All are computable from tensors already resident in the decode step — no additional forward pass through the backbone.
- **Data flow:** backbone features → mixture head → {π_k, μ_k, s_k} (existing) → decode step (existing, FP32-forced) → **explanation record** = {dominant: (μ_{k*}, π_{k*}), secondary: (μ_{k2}, π_{k2}), within-var, between-var, entropy(π)} → served as an additional small payload alongside RGB/confidence outputs.
- **Runtime/memory:** one extra `topk(2)` instead of `argmax` on a length-K vector (K=6–8) per pixel — immeasurably small relative to the backbone forward pass.

### 4. Implementation Plan
A single engineer, well under a day: modify the existing decode function to return the explanation record instead of only the dominant-mode output; extend the serving payload schema; for the demo, render dominant/secondary hypothesis colors side-by-side with the entropy value at a clicked pixel.

### 5. Expected Impact
Zero cost, by construction (§3). Expected impact is entirely in reviewer/judge legibility: this is the single feature most directly able to turn the formulation review's mathematical argument (dominant-mode vs. mixture-mean, §1/§2/§4.5 of that document) into something a non-specialist judge can see and understand within seconds, which materially strengthens the project's defensibility under Q&A (validation plan, Part 9, Q4 and Q48).

### 6. Trade-offs
None of substance — the only addition is a slightly larger output schema and a small amount of serving/UI code. This is the rare case in this review where there is no real trade-off to weigh. **Verdict: retained, unconditionally.**

---

## FEATURE 4 — Deployment Optimization

### 1. Technical Review
**Should it exist? Yes — but almost entirely as already specified in the implementation plan and engineering stability analysis, with two narrow, justified additions and one explicit rejection.**

Most of the listed topics (ONNX export, TensorRT, mixed precision/FP16, kernel fusion) are not new proposals — they restate decisions already made: FP16 backbone as the demo baseline with INT8 as a validated stretch goal (implementation plan §6.4), a single fused inference graph for the backbone (TIR framework §6), and an explicit **backbone/decode export split** with the decode arithmetic forced to FP32 (engineering stability analysis §7) because log-sum-exp and sigmoid-difference operations are not always efficiently fused by ONNX Runtime/TensorRT's optimizer. Restating these here without change would be redundant, not engineering review.

Two items genuinely add value at negligible cost and were not yet explicit in prior documents:
- **Asynchronous tile loading/prefetch**, overlapping the I/O-bound tile-read/calibration step with the GPU-bound backbone forward pass of the *previous* batch. Since the backbone is small (~3.4M parameters), I/O and host-side preprocessing are plausible bottlenecks relative to compute, making this a real, not speculative, win.
- **CPU/GPU overlap specifically at the backbone/decode boundary**: since the decode step is forced to FP32 and is already understood (engineering analysis §6) to be cheap and possibly CPU-resident depending on the export split, overlapping it with the *next* tile's GPU-side backbone pass is a legitimate pipeline-overlap (not pipeline-parallelism) optimization.

One item is explicitly rejected: **pipeline parallelism** in the multi-GPU sense is not implementable — the entire system, per the implementation plan's foundational hardware constraint, is sized to a single mid-range GPU (T4/RTX 3060-class). Designing for multi-GPU parallelism would be designing for hardware the team does not have and cannot validate against within the BAH timeline.

**Memory optimization** beyond what is already true requires no new work: the implementation plan's own sizing table shows the full system trains and runs at ~3.4M parameters, comfortably within 8–16GB VRAM at batch 16–32; there is no identified memory pressure to optimize against, and inventing one would be solving a problem that does not exist.

### 2. Scientific Justification
Directly improves **low inference time per tile** (the explicit BAH low-latency criterion) and indirectly supports the **physics-informed modelling** criterion's credibility, since the FP32 decode boundary is what protects the calibration quality the entire formulation exists to provide (engineering analysis §3c) — a deployment choice that is itself a hallucination-reduction safeguard, not merely a performance one.

### 3. Engineering Design
- **Architecture:** unchanged from the implementation plan/engineering analysis, plus the two additions above as standard producer-consumer queue overlap between the data-loading thread/process and the GPU inference stream (e.g., `torch.utils.data.DataLoader` with `pin_memory=True` and `num_workers>0`, or an equivalent async queue in the FastAPI serving path), and an async dispatch of the FP32 decode step on the host while the next tile's backbone pass is issued on the GPU stream.
- **Data flow:** tile read/calibrate (CPU, async) → backbone forward (GPU, fused ONNX/TensorRT graph, FP16) → mixture-head raw outputs (GPU) → decode (FP32, host or separate low-cost GPU kernel, overlapped with next tile's backbone dispatch) → output payload.
- **Runtime/memory:** no change to per-tile compute; expected benefit is in *sustained throughput* (tiles/sec under batch/stream load) rather than single-tile latency, since overlap only pays off when there is a next tile already queued.

### 4. Implementation Plan
Already largely implemented as specified in the implementation plan's `export_onnx.py`/serving stack; the two additions are a half-day each: wire an async data-loading queue ahead of the existing inference call, and confirm (via profiling, per the validation plan's Part 8 deployment-validation table) that the decode step does not stall the GPU stream.

### 5. Expected Impact
Quantified relative to the existing baseline: async prefetch and CPU/GPU overlap target **sustained throughput on batch/streaming workloads**, not the single-tile latency number already reported elsewhere; expected benefit is bounded by how I/O-bound the pipeline turns out to be in practice, and must be measured (Part 8 of the validation plan: tile streaming performance), not assumed — this review does not assert a specific percentage speedup without that measurement.

### 6. Trade-offs
The two additions are standard, well-understood serving patterns with low implementation risk and no change to the export/precision boundary already validated. Pipeline parallelism is rejected outright because the cost (an entire untested multi-GPU code path) has no benefit on the team's actual hardware target. **Verdict: retained as specified in prior documents, plus async prefetch and host/GPU decode overlap; multi-GPU pipeline parallelism explicitly excluded.**

---

## FEATURE 5 — Live Benchmark Mode

### 1. Technical Review
**Should it exist? Yes, as a direct extension of the demo already designed in the implementation plan (§8), not a new subsystem.**

The implementation plan already specifies a Streamlit panel showing raw 200m TIR, SR 100m TIR, synthesized RGB, and a confidence/abstention overlay shown by default, on a pre-loaded heterogeneous (urban-rural fringe) tile chosen specifically because that is where VARNA's advantage over a naive baseline is most visible. The proposed comparison chain (Original → Existing Method → VARNA → Difference → Uncertainty → Latency → Physics Consistency) is a real strengthening of that design — specifically, the addition of a **side-by-side baseline comparison** — but every element in the chain must be a readout of something already computed, not something computed live for the demo, because live training or live baseline inference inside a two-minute judge interaction is an unacceptable failure-mode surface for a finale demo.

### 2. Scientific Justification
Directly supports every criterion simultaneously by design intent — this is a presentation layer, not a new scientific claim — but its *engineering* justification rests narrowly on reusing the cached, pre-computed outputs from Part 3's baseline comparison (a deterministic-regression or Pix2Pix baseline, already trained for the scientific validation plan) and Feature 3's explanation record, rather than introducing any new computation.

### 3. Engineering Design
- **Architecture:** an extension of `demo/streamlit_app.py`. Panels, each a readout of pre-computed artifacts:
  1. **Original Thermal** — raw 200m input tile (static asset).
  2. **Existing Method** — pre-computed baseline (deterministic-regression or Pix2Pix, cached output, *not* run live) output on the same tile.
  3. **VARNA** — pre-computed dominant-mode RGB output (or live single forward pass — acceptable, since this is the one inference call cheap enough to run live, per Feature 1's finding that the backbone is small and fast).
  4. **Difference** — pixelwise absolute-difference image between VARNA and baseline outputs, computed once and cached (a single subtraction, trivial to compute live if needed since it is cheaper than the inference itself).
  5. **Uncertainty** — Feature 3's explanation-record overlay (entropy/variance heatmap), already free.
  6. **Latency** — a live timer around the one live inference call (VARNA's forward pass), displayed directly rather than asserted, addressing the validation plan's reviewer Q42/Q46 ("don't just claim low latency, show it live").
  7. **Physics Consistency** — re-degrade VARNA's SR output with the known PSF + downsample operator (the same operator already used to construct training pairs, implementation plan §3.2) and display the residual against the actual 200m input — a direct visualization of the SR loss's own degradation-consistency term, costing one extra downsample-and-difference operation, not a new model or metric.
- **Data flow:** all panels except (3) Latency and (6) VARNA's own forward pass are static, pre-computed assets loaded once at demo start; only the live VARNA inference call and its timer execute during the judge interaction, minimizing live-demo failure surface.
- **Runtime/memory:** negligible beyond the single live forward pass already budgeted elsewhere; cached baseline/difference images add a small, fixed asset-storage cost, not a runtime cost.

### 4. Implementation Plan
One engineer, roughly a day, building directly on the existing Streamlit app: pre-generate and cache baseline-model outputs and difference images for the existing pre-loaded heterogeneous tile (and 1–2 backups) ahead of the finale; add the physics-consistency residual panel (one downsample+diff call); wire a live timer around the existing VARNA inference call; lay out the seven panels in the comparison chain order specified.

### 5. Expected Impact
Not metric-improving; directly addresses the brief's own success criterion — "convince technical reviewers within two minutes" — by making the formulation's central claims (hallucination reduction, physics-consistency, low latency) simultaneously visible rather than requiring a judge to read a slide and take a claim on faith. This is the demo-layer counterpart to Feature 3's explanation record: both exist to make already-computed, already-true facts about the model legible, not to compute anything new.

### 6. Trade-offs
The only added complexity is asset pre-caching discipline (the cached baseline/difference images must be regenerated if the underlying models are retrained — a process-hygiene risk, not a technical one, and the same discipline `run_full_eval.sh` already enforces per the implementation plan's reproducibility requirement). Acceptable because it reuses artifacts the project produces anyway (Part 3's baselines, Feature 3's explanation record, the SR loss's own degradation-consistency term) rather than introducing new computation. **Verdict: retained, as an extension of the existing demo design, not a new subsystem.**

---

## SYSTEM INTEGRATION — Why the Architecture Stays Simple

Of five proposed extensions, **one is removed outright (Adaptive Inference Engine)**, one survives only in a reduced, relocated form (the tile-priority idea folds into ordinary batch scheduling under Feature 4, not as its own subsystem), and three survive largely unchanged in scope because they were, on inspection, **free or nearly free readouts of computation the frozen formulation already performs**, not new computation. This pattern is not a coincidence — it is the direct consequence of having already pushed the genuine scientific complexity into a single closed-form mixture-density forward pass (formulation review §6: "no separate classification stage, no codebook, no calibration-loss term, no perceptual loss term... this is the leanest version of VARNA that has appeared across this entire review process"). A formulation that already outputs a full distribution, a variance decomposition, and a dominant/secondary hypothesis pair in one pass has very little left for a feature layer to *compute* — only things left to *read out, rank, cache, and display*.

### Final data flow (single fused forward pass remains the only inference event)

```
 raw TIR tile (200 m)
        │
        ▼
 [async prefetch / triage tag: low-info vs heterogeneous]   ← Feature 4 (relocated Feature-1 idea)
        │
        ▼
 Stage 0: radiometric calibration (T_B = g(DN_raw; P))        [FROZEN]
        │
        ▼
 Shared backbone (≈2.5M params, fused ONNX/TensorRT, FP16)    [FROZEN]
        │
        ├──► SR head (deterministic, single-pass residual CNN)        [FROZEN]
        │         │
        │         ▼
        │    SR-TIR (100 m) ──► re-degrade (PSF+downsample) ──► physics-consistency residual   ← Feature 5 panel 7
        │
        └──► Mixture head (7K channels: π_k, μ_k, s_k)                [FROZEN]
                  │
                  ▼
         Decode submodule (forced FP32)                               [FROZEN math, engineered boundary]
                  │
                  ├──► dominant-mode RGB (μ_{k*})                     [FROZEN — primary output]
                  ├──► secondary hypothesis (μ_{k2}, π_{k2})           ← Feature 3 (free readout)
                  ├──► within/between-mode variance + entropy(π)      ← Feature 3 (free readout)
                  │
                  ▼
         Explanation record {dominant, secondary, variances, entropy}
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Confidence-ranked      Live demo panel
 inspection queue       (cached baseline + difference,
 ← Feature 2             live VARNA + timer, uncertainty
   (sort over scene)       overlay, physics residual)
                          ← Feature 5
```

**Why this stays simple:** every surviving feature attaches to an output edge of the frozen forward pass — none inserts itself into the forward pass, none requires a second trained model running live, and none changes what is computed, only what is *done with* what was already computed. The one rejected feature (Adaptive Inference Engine) is rejected precisely because it would have inserted new conditional logic *into* the compute path for a cost component that profiling already shows is not the bottleneck.

---

## FINAL PRODUCTION ARCHITECTURE OF VARNA (BAH 2026)

**Mathematical core (unchanged, frozen):**
- Stage 0 — deterministic radiometric calibration.
- Shared backbone (~2.5M params).
- SR head — deterministic single-pass residual CNN, soft PSF-consistency loss.
- Mixture head — discretized logistic mixture (K=6–8), NLL training objective, dominant-mode decode, within/between-mode variance decomposition.

**Engineering extensions that survived review:**
1. ~~Adaptive Inference Engine~~ — **removed**; the one defensible idea inside it (tile-priority scheduling) absorbed into ordinary batch scheduling under Deployment Optimization.
2. **Confidence-driven Analyst Mode** — retained, scoped to a stateless ranking/tagging readout of existing variance-decomposition fields.
3. **Explainability Layer** — retained, unconditionally; a zero-extra-cost extension of the existing decode step's output schema (dominant + secondary hypothesis, variance decomposition, entropy).
4. **Deployment Optimization** — retained as already specified (ONNX/TensorRT, FP16 backbone, FP32 decode boundary, INT8 stretch goal), plus async tile prefetch and host/GPU decode overlap; multi-GPU pipeline parallelism explicitly excluded as infeasible on the team's hardware target.
5. **Live Benchmark Mode** — retained as a direct extension of the existing demo design, built entirely from cached or already-cheap-to-compute artifacts (baseline outputs, difference image, explanation record, physics-consistency residual, live latency timer).

**What was deliberately not added:** no second trained model running at inference time, no new loss terms, no new architectural branch inside the backbone or either head, no multi-GPU requirement, no UI behavior that is not a direct readout of an existing output field. The production system remains, end to end, **one shared backbone and two heads, executed once per tile**, with everything else — ranking, explanation, deployment scheduling, and the demo — built as bounded, justified consumers of that single computation. This is, by design, the same discipline the formulation review applied to the mathematics applied now to the systems layer: every addition had to answer the five-question test in this document's rule of direct contribution, and most of what was proposed did not.
