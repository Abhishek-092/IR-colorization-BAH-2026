# VARNA: Variational class-Aware Radiance-to-reflectance Network for hallucination-Aware colorization of Thermal Infrared Imagery

**A Technical Design Proposal — Review Board Consensus Document**
*Prepared in the style of an internal SAC / IEEE TGRS design report*

---

## SECTION 1 — Scientific Analysis of the Problem

### 1.1 What is actually being asked

The stated objective — "Infrared Image Colorization and Enhancement" — bundles two operations that are physically unrelated:

1. **Spatial super-resolution of TIR (200 m → 100 m):** a within-modality inverse problem (deblurring + upsampling of a thermal radiance field).
2. **TIR → RGB synthesis:** a **cross-modality inference problem**, where the target quantity (visible-band surface reflectance) is not encoded, even indirectly, in the source signal (thermal emission).

Treating these as one pipeline, as the baseline does, hides the fact that they have completely different error structures, different ground-truth availability, and different failure modes. The review board's first finding is that **these two tasks must be designed, trained, and evaluated as distinct sub-problems with different epistemic guarantees**, not as stages of one monolithic translation network.

### 1.2 Why TIR → RGB is fundamentally difficult

A pixel's value in a thermal channel is governed by the radiative transfer / Planck relation:

L_sensor(λ_TIR) = ε · B(λ_TIR, T_s) · τ + L_atm↑

where T_s is land surface temperature (LST), ε is surface emissivity at thermal wavelengths, τ is atmospheric transmissivity, and L_atm↑ is upwelling atmospheric radiance. The information content of a single TIR channel is therefore (T_s, ε) **convolved with atmospheric state** — a one-or-two-parameter physical quantity per pixel.

A pixel's RGB value, by contrast, is governed by **reflected** solar radiance:

L_sensor(λ_RGB) = ρ(λ_RGB) · E_sun(λ_RGB) · cos(θ) · τ_vis + path radiance

where ρ(λ) is spectral reflectance — a function shaped by chemical composition, surface roughness, and moisture, sampled at three independent wavelengths.

These two physical processes (thermal emission vs. solar reflection) are governed by **different surface properties**. Temperature and emissivity do not determine spectral reflectance. Two surfaces with near-identical brightness temperature (dry asphalt and dark basaltic soil, for instance) can have completely different RGB reflectance, and two surfaces with very different RGB color (green grass vs. brown dry grass) can show nearly identical thermal signatures at certain times of day. **There is no deterministic or even approximately bijective mapping from TIR to RGB.** Any model that claims to "reconstruct" RGB from TIR is, strictly, performing **conditional hallucination constrained by learned land-cover statistics** — not physical inversion. This must be stated explicitly in the proposal rather than obscured behind generative-model language, because it is the central scientific fact that determines every downstream design choice.

### 1.3 What information is physically present, and what is absent

**Present in single-channel TIR:**
- Spatial structure: edges, boundaries, shapes (driven by emissivity/temperature contrast — coastlines, urban-rural boundaries, water bodies, large built-up clusters).
- Approximate land surface temperature field (after radiometric calibration).
- A *weak* land-cover discriminability signal, because broad classes have characteristic thermal behaviour (open water is thermally stable and cool; impervious urban surfaces run hot in daytime and stay warm at night due to high thermal inertia; vegetation is cooler than bare soil due to evapotranspirative cooling).

**Absent in single-channel TIR:**
- Any direct spectral reflectance information — chrominance does not exist in this modality.
- Fine within-class material discrimination (e.g., crop type, building material, soil mineralogy) without auxiliary cues.
- Disambiguation of metameric thermal classes — many materials share thermal signatures (shadow vs. dark wet soil vs. asphalt; bare soil vs. dry vegetation at certain hours).

This sets the realistic ceiling for the project: **structure can be recovered with high fidelity (legitimate SR problem); color can only be assigned at the resolution of land-cover class statistics, with explicit, quantified uncertainty.** Any framework promising pixel-accurate "true color" reconstruction from TIR alone is scientifically indefensible, and a reviewer at IEEE TGRS or an ISRO scientist would immediately flag this as overclaiming.

### 1.4 Where hallucination originates

1. **Underdetermined inverse mapping** — one thermal value maps to many plausible RGB values; a network trained with pixel-wise or adversarial losses will pick the *statistically dominant* answer for the dataset, confidently, even where it is wrong for the specific pixel.
2. **Adversarial loss objectives** are mode-seeking: a discriminator rewards "looks like a real photograph," not "is radiometrically/semantically correct." This actively encourages confident fabrication of texture and color that has no support in the input.
3. **Domain shift** — illumination, season, acquisition time-of-day differences between TIR and any reference RGB used in training propagate into spurious correlations (e.g., model learns "warm pixel near water = sandy beach" from training scenes, then paints sand on a warm industrial rooftop).
4. **Super-resolution of a band-limited signal** — thermal sensors have a point-spread function (PSF) substantially wider than one ground pixel (≈2 pixel widths for Landsat TIRS), so detail finer than the PSF does not exist in the data. GAN-based SR will synthesize plausible-looking high-frequency texture that is not physically present — a second, independent source of hallucination layered on top of the colorization problem.

### 1.5 Why existing methods fail

Pix2Pix / CycleGAN-style image-to-image translation and most diffusion-based colorization pipelines treat TIR→RGB as a direct pixel regression/generation task. They do not separate "what surface is this" from "what does this surface look like," so the network must implicitly learn both a classifier and a generator inside one black box, with no mechanism to express low confidence. The result, well documented in thermal colorization literature (driving/night-vision colorization, FLIR-to-RGB GANs), is visually convincing but locally false output — exactly what the problem statement asks the board to avoid.

---

## SECTION 2 — Critical Review of the Baseline and Candidate Approaches

| Approach | Strength | Why it is insufficient here |
|---|---|---|
| GAN (Pix2Pix/CycleGAN) direct TIR→RGB | Sharp, photorealistic output | Mode-seeking, no uncertainty, highest hallucination risk |
| Diffusion-based colorization | Good distributional realism | Multi-step sampling → high inference latency, unacceptable at million-tile scale; still lacks physical grounding |
| Generic SR networks (ESRGAN, SwinIR) on TIR | High PSNR/SSIM on natural images | Trained assumptions (sharp high-frequency texture exists) violate the TIR PSF physics — introduces structural hallucination |
| Naive sequential pipeline (SR → colorize) | Simple, matches baseline | Compounds error: hallucinated SR artifacts get "colored" as if real; no error decomposition or attribution |

**Board consensus:** none of these are wrong in isolation, but all of them conflate two epistemically different operations and offer no mechanism to *know when they don't know*. The framework must be redesigned around explicit decomposition and calibrated abstention rather than end-to-end black-box translation.

---

## SECTION 3 — Proposed Framework: VARNA

**VARNA** (वर्ण) is deliberately chosen: in Sanskrit it denotes simultaneously **"color"** and **"class/category"** — which is precisely the framework's central scientific claim: *color in this problem can only be recovered through class, not directly from radiance.*

**Expanded form:** *Variational class-Aware Radiance-to-reflectance Network for hallucination-Aware colorization.*

VARNA reformulates the task from

> TIR → RGB (direct translation)

to

> TIR → (Land-Surface-Temperature field, Land-cover class posterior, Confidence) → class-conditioned reflectance distribution → RGB + Uncertainty Map

This reformulation is the single largest departure from the baseline and from the 80% of expected submissions, and is the core deliverable an IEEE/ISRO reviewer should recognize as novel: **the network is explicitly forbidden from regressing RGB directly from TIR radiance; it must route every color decision through an interpretable, separately-supervisable land-cover posterior.**

---

## SECTION 4 — Complete Framework Description

### Stage 0 — Radiometric Calibration (deterministic, not learned)
Raw TIR digital numbers are converted to at-sensor spectral radiance, then to brightness temperature T_B via the inverse Planck function using sensor calibration constants (K1, K2 for Landsat-class TIRS). This is a closed-form physical step, not a network — it removes sensor/acquisition-time radiometric variability before any learning occurs, reducing the domain-shift component of hallucination described in §1.4. Computational cost: negligible (<0.1 ms/tile, vectorized).

### Stage 1 — Degradation-Aware Super-Resolution (200 m → 100 m)
Rather than blind SR (learn any mapping from low-res to high-res), VARNA explicitly models the forward degradation operator:

y = (h * x)↓s + n

where h is the TIRS PSF (known, approximately Gaussian, ~1.5–2 pixel FWHM), ↓s is the downsampling-by-2 operator, and n is sensor noise. SR is posed as a **model-based unrolled optimization** (deep unrolling, in the spirit of USRNet/DPSR), alternating a data-consistency step (enforcing that re-degrading the estimate reproduces the input) with a learned denoising/prior step, for **3–5 unrolled iterations**.

*Justification:*
- **Mathematical validity:** the data-consistency step guarantees the output is physically compatible with the observed radiance — it cannot drift into pure hallucination, unlike a free-running GAN.
- **Hallucination reduction:** because the prior step only refines structure consistent with the known PSF, it cannot fabricate frequency content the sensor could not have captured at any plausible confidence.
- **Inference time:** 3–5 lightweight unrolled steps (small CNN denoiser, shared weights across iterations) is one to two orders of magnitude faster than diffusion-based SR, while retaining most of the accuracy benefit of model-based regularization.

Output: 100 m brightness temperature field, T_B^SR — this **is** the "Super-Resolved TIR" deliverable, and is evaluated directly against native 100 m Landsat TIRS-2 acquisitions (genuine ground truth exists here, unlike for RGB).

### Stage 2 — Land-Cover Class Posterior Estimation
A lightweight shared-backbone encoder (re-using features from Stage 1, to amortize compute) predicts a per-pixel **class probability vector** p(c | T_B^SR, texture features) over a small set of broad classes: open water, vegetation (further split high/low biomass if training data supports), bare soil/rock, built-up/impervious, cloud/shadow/no-data.

*Justification:* this step is well-grounded in established thermal remote sensing — diurnal thermal amplitude, apparent thermal inertia and mean brightness temperature have long been used (e.g., Apparent Thermal Inertia indices) for broad land-cover discrimination from thermal data alone. The class boundaries are therefore physically motivated, not arbitrary network outputs.

Supervision: weak/pseudo-labels from an existing global land-cover product (e.g., ESA WorldCover or equivalent), co-registered to the Landsat grid — no manual labeling required, and the label source is independent of the RGB target, which prevents label leakage into the color stage.

### Stage 3 — Class-Conditioned, Uncertainty-Calibrated Color Synthesis
For each pixel, given its class posterior, spatial context, and the SR'd thermal field, a **mixture-density decoder** predicts a distribution over RGB (e.g., a small Gaussian mixture per class, parameters modulated by local context) rather than a single deterministic value. The decoder outputs:
- mean RGB estimate,
- per-pixel predictive variance (aleatoric uncertainty — genuine ambiguity, e.g. mixed pixels),
- class-entropy-derived epistemic flag (model is being asked to color something its land-cover classifier itself is unsure about).

**Abstention mechanism:** where combined uncertainty exceeds a calibrated threshold, VARNA does **not** synthesize a color. The output reverts to a grayscale/false-color thermal overlay for that pixel, explicitly marked as "insufficient evidence for color assignment." This is the framework's principal anti-hallucination mechanism — instead of suppressing visible hallucination by making the network's confident-but-wrong outputs look smoother, VARNA makes *uncertainty itself* a first-class, visualized output.

A small adversarial/perceptual term may be added at low weight purely to remove block artifacts in the mixture-density output; it is **not** the primary fidelity signal, deliberately, because adversarial losses are the principal driver of confident fabrication (§1.4).

### Stage 4 — Structural Consistency Constraint (the physics-informed component)
Although thermal and visible processes are governed by different surface properties, they share the same **surface geometry** (boundaries, object edges occur at the same spatial locations in both modalities, even when the physical cause of contrast differs). VARNA enforces a **gradient-correlation constraint**: the spatial gradient of synthesized RGB luminance must be correlated with (not equal to) the spatial gradient of the SR'd brightness-temperature field, penalizing only cases where an edge appears in the color output with no corresponding structural evidence in the thermal field. This constrains hallucinated boundaries without forcing texture identity between modalities that physics does not support — a stricter equality constraint would be physically wrong (e.g. it would suppress legitimate color variation within thermally homogeneous regions, like painted vs. unpainted roofs at the same temperature).

### Stage 5 — Adaptive Tile Scheduling (latency optimisation)
A cheap pre-pass (tile-level entropy/variance of raw TIR, computable in microseconds) routes tiles into two compute tracks:
- **Low-information tiles** (uniform water, desert, large cloud-free homogeneous fields) → reduced unrolled-SR iterations (1–2) and a coarse class-conditioned color (since spatial detail to recover is genuinely low) — large latency savings where they cost nothing in accuracy.
- **High-information tiles** (urban/coastal/heterogeneous) → full 5-iteration SR and fine-grained class posterior.

*Justification:* this is not an arbitrary "make it faster" trick — it follows directly from the information-theoretic observation that homogeneous thermal regions carry little exploitable structure, so allocating equal compute to every tile is provably wasteful under an accuracy-per-millisecond objective, which the problem statement explicitly prioritizes.

### Expected failure cases (stated openly, as required)
- Mixed/sub-pixel land cover (urban-vegetation fringe) → high aleatoric uncertainty, frequent abstention — acceptable, not a bug.
- Novel materials absent from training class taxonomy (e.g. unusual industrial surfaces) → epistemic flag should trigger, but classifier miscalibration under true distribution shift remains a residual risk.
- Cloud/shadow contamination in TIR → degrades both calibration and SR; handled by a no-data class but cannot be fully eliminated.
- Acquisition time mismatch (day vs. night TIR) → diurnal thermal contrast differs sharply; without a known acquisition-time covariate fed into the class posterior, this is a confounder the model cannot fully resolve from radiance alone.

---

## SECTION 5 — Training Methodology

Landsat 9 acquires co-registered OLI (30 m, multispectral, including RGB-equivalent bands) and TIRS-2 (100 m thermal) in the same overpass. This gives **genuine paired ground truth** for both sub-problems without synthetic data:

- **SR target pairs:** real 100 m T_B (from native TIRS-2) degraded via the known PSF + downsampling to simulate 200 m input — self-supervised, no hallucinated targets.
- **Color target pairs:** real OLI-derived RGB, resampled/aggregated to 100 m, co-registered to the TIRS grid, used directly as supervision for Stage 3 — meaning the "RGB ground truth" used in training is physically real, not another network's guess.
- **Class labels:** independent weak supervision from a global land-cover product, as in Stage 2.
- Stratified sampling across land-cover classes is mandatory to prevent the dominant-class bias that produces over-confident, homogeneous coloring of underrepresented classes (a primary hallucination pathway in naturally imbalanced remote-sensing datasets).

Training proceeds in three phases: (1) Stage 1 SR network trained alone against degradation-consistency + brightness-temperature L1 + edge-gradient loss; (2) Stage 2 classifier trained against weak labels with spatial smoothness regularization; (3) Stage 3 color decoder trained against real OLI RGB using mixture-density negative log-likelihood + low-weight perceptual term + Stage 4 structural-consistency penalty, with Stages 1–2 frozen to prevent error coupling during early color training, then optionally fine-tuned end-to-end at low learning rate.

---

## SECTION 6 — Inference Pipeline

`raw TIR tile → triage (entropy/variance) → radiometric calibration (T_B) → unrolled SR (1–5 steps, route-dependent) → shared-backbone class posterior → mixture-density color decoder + uncertainty → structural-consistency check → abstention masking → output: {SR-TIR (100 m), RGB (100 m), confidence map}`

All stages share a single backbone feature extractor where possible (amortized convolution cost across Stage 1 and Stage 2), and the pipeline is implemented as a single fused inference graph (ONNX/TensorRT) to avoid intermediate I/O overhead.

---

## SECTION 7 — Loss Functions

- **L_SR** = L1(T_B^SR, T_B^true) + λ₁·L_degradation-consistency + λ₂·L_edge-gradient
- **L_class** = CrossEntropy(p(c), weak-label) + λ₃·L_spatial-smoothness (CRF-like pairwise term)
- **L_color** = NLL_mixture-density(RGB | class posterior, context) + λ₄·L_perceptual(low weight) + λ₅·L_structural-consistency (Stage 4 gradient correlation)
- **L_calibration** = penalty on Expected Calibration Error between predicted confidence and observed color accuracy on a held-out stratified validation set — included explicitly because an uncertainty output that is not calibrated is decorative, not scientific.

---

## SECTION 8 — Evaluation Methodology

**For SR-TIR (true ground truth exists):** PSNR, SSIM, and brightness-temperature RMSE against native 100 m TIRS-2.

**For RGB (no pixel-level ground truth is scientifically meaningful, as argued in §1.3):**
- FID for distributional plausibility (sanity check only, not primary metric).
- **Per-class color accuracy:** compare predicted mean RGB per land-cover class against the true class-conditional reflectance statistics from co-registered OLI — this is the metric that actually matches what the model can legitimately claim to predict.
- **Calibration metrics:** Expected Calibration Error, and sparsification plots (error vs. fraction of pixels rejected by the abstention mechanism) — directly operationalizes "minimize hallucination" as a measurable, reportable curve rather than a qualitative claim.
- **Visual inspection protocol:** structured comparison on held-out heterogeneous scenes (urban-rural fringe, coastlines) specifically because these are where hallucination is most likely, not on easy homogeneous scenes.

**For latency:** ms/tile broken down by triage track, reported as an accuracy-per-millisecond curve (e.g., per-class color accuracy vs. mean inference time), directly addressing the "low inference time" preference as a quantified trade-off rather than a single number.

---

## SECTION 9 — Deployment on ISRO Infrastructure

For million-tile batch processing on SAC ground-segment compute: INT8 quantization of the SR and classification backbones (validated against FP32 baseline to ensure brightness-temperature error stays within sensor noise floor), batched tile triage to maximize GPU occupancy on the low-compute track, and TensorRT/ONNX graph fusion across the shared backbone. **Diffusion-based variants, despite being in the suggested-model list, are explicitly excluded from the production inference path** — multi-step sampling is incompatible with the stated low-latency requirement at this scale; a diffusion model may still be retained offline as a teacher/distillation source for the mixture-density decoder, but is not deployed. This is a deliberate, justified departure from the suggested model list, consistent with the brief's instruction that suggested ideas are hypotheses, not mandates.

---

## SECTION 10 — Scientific Limitations

- The framework cannot and does not claim to recover true spectral reflectance from thermal radiance alone; this is an information-theoretic impossibility given the stated single-channel input, not an engineering shortfall.
- Color outputs are class-conditional statistical estimates, accurate at the level of land-cover category, not validated for individual-object material identification.
- Performance is bounded by the granularity and accuracy of the land-cover taxonomy used for weak supervision; novel or rare materials are systematically under-served.
- Acquisition-time (day/night) confounding is only partially mitigated; without an explicit time-of-acquisition covariate, some diurnal ambiguity is irreducible from radiance alone.
- Calibration quality depends on the representativeness of the stratified validation set; under genuine distribution shift (new geography, season, sensor), confidence estimates require re-calibration.

---

## SECTION 11 — Future Research Directions

- Incorporating **multi-temporal thermal pairs (day + night)** to estimate apparent thermal inertia directly, substantially improving land-cover class discriminability beyond single-image brightness temperature.
- Extending Stage 2 to a **temperature–emissivity separation (TES)-style retrieval** when multi-band thermal data is available, improving the physical grounding of the class posterior.
- Active learning loops that flag high-uncertainty regions for targeted ground-truth collection, progressively narrowing the abstention zone.
- Transfer of the VARNA architecture to other ISRO TIR sensors (e.g., future SAC thermal payloads) by re-calibrating Stage 0 alone, leaving Stages 1–4 largely sensor-agnostic.

---

### Summary of Major Design Trade-offs Considered and Rejected

| Rejected design | Reason |
|---|---|
| End-to-end GAN/diffusion TIR→RGB | Highest hallucination risk, no uncertainty output, diffusion too slow for deployment |
| Pixel-wise direct RGB regression | Conflates classification and generation; cannot express "don't know" |
| Hard equality constraint between thermal and RGB gradients | Physically incorrect — would suppress real color variation within thermally uniform regions |
| Uniform per-tile compute | Wastes latency budget on low-information tiles; fails the explicit low-inference-time priority |

The board's consensus is that the central scientific contribution is not a new architecture per se, but the **explicit reformulation of TIR-to-RGB as a class-mediated, uncertainty-calibrated estimation problem with a formal abstention mechanism** — a framing that makes hallucination measurable and bounded rather than merely "reduced" by a better generator.
