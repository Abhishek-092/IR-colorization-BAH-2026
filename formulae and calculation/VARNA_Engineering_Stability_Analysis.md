# VARNA — Engineering Realization Analysis of the Fixed Mathematical Formulation

**Scope constraint honored throughout:** the estimand p(RGB | T_B) = Σ_k π_k(T_B)·Π_c DiscLogistic(x_c; μ_{k,c}, s_{k,c}) and its NLL training objective are treated as fixed. Every item below is a modification to *how* this object is computed, initialized, regularized at the implementation level, or deployed — never a modification to *what* is being estimated.

---

## 1. Mixture collapse — two distinct phenomena, two distinct minimal fixes

This is the dominant risk class for any mixture-density network and it has two genuinely different failure shapes that are often conflated. Both are well documented in the mixture-density-network literature (Bishop 1994 onward) and both have established, minimal, formulation-preserving fixes.

**1a. Variance (degenerate-component) collapse.** During training, gradient descent on the NLL can drive one component's scale s_k toward zero around a single training example, since a discretized-logistic component with very small scale assigns almost all its mass to one bin — for that example, this is locally optimal (near-perfect fit), but it amounts to the component memorizing one pixel rather than learning a generalizable mode. Although the discretized-logistic likelihood is bounded above by 1 per bin (unlike a continuous Gaussian density, which is formally unbounded as variance → 0, so this failure is less catastrophic in magnitude than the classical continuous-MDN case) the practical effect is the same: a collapsed component contributes a near-delta-function that overfits and contributes a vanishing, uninformative gradient elsewhere.

*Smallest modification:* parameterize the scale as s_k = softplus(raw_s_k) + ε, with ε fixed at a small constant (order of one intensity level on the 0–255 scale, i.e. effectively the minimum meaningful resolution of the discretization itself, not an arbitrary regularization strength). This is a one-line reparameterization of the existing output, not a new loss term, and does not change the mixture's mathematical form — it only excludes a degenerate corner of the parameter space that was never a meaningful solution in the first place. Note ε itself is a new hyperparameter introduced by this fix; the smallest defensible choice is the bin width of the discretization, since any tighter scale is not separable from the discretization noise floor anyway, so this is not an arbitrary tuning knob but a value the formulation itself already implies.

**1b. Mode redundancy collapse.** Independently of 1a, multiple components can converge to nearly identical (μ_k, s_k) values, especially early in training from symmetric initialization, because all components start near the global marginal mean and receive nearly identical gradient signal — there is no symmetry-breaking force in the loss itself to push them apart. The network then behaves as if K were smaller than specified, silently wasting the capacity the original K-sweep (from the formulation review) assumed was available.

*Smallest modification:* break symmetry at initialization only, not during training — initialize each component's mean bias from a different quantile of the empirical marginal RGB distribution computed once from the training set (e.g., K components placed at the (k − 0.5)/K quantiles). This is a one-time, data-dependent initialization computation, not an additional loss term, regularizer, or architectural change, and it directly targets the actual cause of the redundancy (a degenerate saddle point at the symmetric initialization) rather than fighting its symptom with an added penalty.

**Engineering consequence for K-selection.** The earlier recommendation (formulation review, §7) to select K via a held-out-NLL sweep is only valid *after* both fixes above are applied. Without them, an NLL-vs-K curve that appears to plateau at small K may be reporting an optimization pathology (collapse) rather than a true statement about how many physical modes the data supports. The team should treat the K-sweep as contaminated until 1a/1b are in place, then re-run it — this ordering matters and should be written into the experiment log, not just the final report.

---

## 2. Component starvation (gradient routing pathology)

A subtler dynamic than outright collapse: gradients to each mixture component are weighted by that component's posterior responsibility (an implicit consequence of differentiating the log-sum-exp of the mixture). Components that start with low responsibility receive correspondingly small gradients, which keeps their responsibility low, which keeps their gradients small — a self-reinforcing "rich get richer" dynamic, structurally similar to the symmetry-breaking problem in §1b but persisting throughout training rather than only at initialization, and capable of suppressing a component even when it has been correctly initialized to a distinct, genuinely useful location.

*Smallest modification:* anneal a temperature on the mixture-weight softmax during the early part of training only — divide the pre-softmax logits for π_k by a temperature τ > 1 for the first portion of training, then linearly anneal τ → 1 by the end. This flattens the responsibility landscape early (preventing premature starvation) without changing the loss function's form at convergence — by the end of training the objective is exactly the original NLL with τ = 1, so the fixed point being optimized toward is unchanged; only the optimization trajectory toward it is smoothed. This is a training-schedule change, not a formulation change.

---

## 3. Numerical stability

Three distinct numerical risks, each with an established, narrowly-scoped fix rather than a redesign.

**3a. log-sum-exp underflow/overflow.** The mixture log-likelihood requires log Σ_k π_k(·)·L_k(·); computed naively, this overflows or underflows in floating point, especially under FP16. *Fix:* the standard log-sum-exp trick (subtract the maximum term before exponentiating, add it back after taking the log) — this is arithmetic bookkeeping, not a modeling choice, and must be implemented explicitly rather than assumed to "just work" inside an autodiff framework's default operator composition.

**3b. Catastrophic cancellation in the discretized-logistic CDF difference.** For a sharply peaked component (small s_k) evaluated away from its center, σ(a) − σ(b) subtracts two floating-point numbers that are nearly equal, losing significant digits. *Fix:* adopt the established stable formulation used in the discretized-logistic-mixture literature — express the difference via a numerically stable identity (e.g., using log1p/expm1-based forms) for interior bins, and use the raw CDF value directly (not a difference) at the two boundary bins (x = 0 and x = 255), where the "difference" degenerates to a one-sided tail probability. This is adopting a documented, off-the-shelf stable implementation, not deriving anything new — the formulation is unchanged; only the arithmetic path to evaluating it is corrected.

**3c. Precision boundary at deployment.** Both 3a and 3b must execute in FP32 regardless of the precision used for the shared backbone. *Scoped fix, more precise than the earlier blanket recommendation:* keep the entire chain from raw mixture-head outputs through log-sum-exp and the CDF-difference computation in FP32; only the convolutional backbone may run in FP16/INT8. This boundary should be enforced as an explicit module split (e.g., a dedicated "decode" submodule with its own forced precision), not left to the export tool's default casting behavior, since the failure mode here is silent — a precision mismatch will not crash the pipeline, it will quietly degrade the calibration quality the entire formulation exists to provide.

---

## 4. Spatial consistency at decode time

The formulation estimates p(RGB | T_B) **independently per pixel**. This is mathematically correct (nothing in the derivation requires or assumes spatial coupling), but it has a real decode-time consequence: the dominant-component selection k* = argmax_k π_k is a discrete decision, and at pixels where two components have nearly equal weight, small input variation (sensor noise, adjacent-pixel differences within an otherwise homogeneous region) can flip which component is selected, producing visible salt-and-pepper inconsistency in the rendered RGB output even though the underlying continuous distributions at neighboring pixels are nearly identical and individually correct.

*Smallest modification:* this is purely a **decode-time, non-trainable, zero-parameter post-processing step** — apply a small fixed spatial filter (e.g., a 3×3 box or Gaussian average) to the responsibility maps π_k(T_B) across neighboring pixels before taking the argmax, not to the underlying network outputs or training procedure. The per-pixel distribution p(RGB | T_B) estimated by the network is untouched; only the discrete decision of *which* mode to report as the dominant one is smoothed using its immediate spatial context. This does not alter the formulation, the loss, or the trained parameters — it is applied after inference, exactly analogous to a non-maximum-suppression step in detection pipelines, and can be disabled entirely without affecting the model's validity if a reviewer specifically wants to see the raw per-pixel output.

---

## 5. Convergence behavior and hyperparameter sensitivity

**Scale parameterization range.** An unconstrained log-scale parameterization (s_k = exp(raw)) is prone to exploding early in training if raw drifts to large positive values. The softplus-based floor already adopted in §1a additionally bounds the practical growth rate better than a raw exponential, but an explicit upper clamp on s_k (e.g., capped at the dynamic range of the channel, 255) is a trivial additional safeguard worth including for the same reason as the floor — it removes a degenerate, non-physical corner of parameter space without touching the formulation.

**Interaction between K and the variance floor ε.** Increasing K gives the mixture more components to potentially collapse (§1a), making the floor more frequently active; setting ε too large, conversely, prevents the model from ever expressing genuinely high-confidence, low-ambiguity pixels (e.g., a uniform water tile, where the true posterior is legitimately very tight). This is the one place in this analysis where a stability fix has a real, non-negligible cost, and it should be reported as such rather than presented as a free correction: the team should verify, on a known-homogeneous validation region (e.g., open water), that predicted scale at the floor still produces visually sharp, low-variance output, confirming ε is set near the discretization's own noise floor and not meaningfully above it.

**Order of hyperparameter tuning.** Tune in this order, since later stages assume earlier ones are stable: (1) variance floor ε and component-mean initialization (§1), (2) softmax annealing schedule (§2), (3) K via held-out NLL (only after 1–2 are fixed, per §1's closing point). Tuning K before 1–2 risks selecting a K that compensates for an optimization pathology rather than reflecting the data's true multimodality.

---

## 6. Memory usage and computational complexity

The mixture head outputs 7K channels per pixel (1 weight + 3 means + 3 log-scales per component) versus 3 channels for a plain regression head. For a practical K in the 6–8 range, this is 42–56 channels at the final layer only — a small, linear, one-time cost confined to the output head, not a multiplicative cost on the shared backbone (which remains unchanged from the deterministic-regression case). The log-sum-exp backward pass requires storing K intermediate per-channel likelihood values for the gradient computation, which is linear in K, not exponential, and negligible relative to the backbone's own activation memory at typical small-model sizes (§4.1 of the implementation plan, ~3M backbone parameters). Compute cost of the likelihood evaluation itself is O(H × W × K × 3) per tile — linear in resolution and in K, dominated in practice by the backbone's convolutional FLOPs, not by the mixture decode arithmetic. No change to either memory or complexity scaling is required; this section is included to confirm, with the actual numbers, that the fixes proposed above (which add a small constant amount of extra arithmetic per pixel) do not alter this favorable scaling.

---

## 7. Deployment characteristics

**Operator support.** log-sum-exp and sigmoid-difference operations are not always efficiently fused by ONNX Runtime / TensorRT graph optimizers in the same way standard convolution and activation layers are. *Smallest modification:* split the exported graph at the natural mathematical boundary already established by §3c — export the shared backbone and the raw mixture-head outputs (π_k, μ_k, s_k logits) through the optimized TensorRT/ONNX path at whatever precision is chosen for the backbone, and perform the decode arithmetic (softmax, log-sum-exp, CDF difference, argmax, variance decomposition) as a small, separate, FP32 post-processing step outside the optimized engine. Given the negligible cost established in §6, this split costs essentially nothing in latency while removing the single largest source of export-time numerical risk.

**Quantization validation scope.** The earlier recommendation to validate INT8 calibration against the brightness-temperature noise floor (implementation plan, §6.4) should be extended specifically to this formulation: validate INT8/FP16 backbone outputs not only against regression accuracy but against **parity of the NLL and sparsification curves** computed from the quantized model versus the FP32 training-time model. A quantization scheme can preserve mean-color accuracy while still distorting the *shape* of the predicted distribution (e.g., flattening sharp components), silently degrading exactly the calibration property that is this formulation's central contribution — average-case accuracy checks alone would not catch this.

**Boundary-case validation.** ONNX numerical-parity checks (already part of the implementation plan) should specifically include held-out pixels with near-degenerate predicted components (s_k near the floor ε), since this is precisely the numerically fragile region identified in §3b — a parity check run only on typical, well-separated-mode pixels could pass while the tails silently diverge under reduced precision.

---

## 8. Items examined and found not to require modification

For completeness, and to avoid proposing fixes where the existing formulation already behaves acceptably under standard practice:

- **Per-pixel training signal density.** Each tile supplies H×W spatially correlated supervision points rather than strictly i.i.d. samples. This is standard for any dense-prediction CNN trained with a per-pixel loss and is not a pathology specific to the mixture formulation; no modification proposed.
- **Backbone capacity.** The mixture head's modest channel-count increase (§6) does not warrant any change to backbone depth or width; growing the backbone to "support" the mixture head would be an unjustified complexity increase with no identified failure mode driving it.

---

## Summary table

| Category | Weakness identified | Smallest modification | Touches formulation? |
|---|---|---|---|
| Mixture collapse (variance) | Component scale collapses to near-zero, overfitting single pixels | Softplus + floor ε on scale parameterization | No — reparameterization only |
| Mixture collapse (redundancy) | Components converge to identical parameters from symmetric init | Quantile-based component-mean initialization | No — initialization only |
| Component starvation | Gradient-routing "rich get richer" dynamic suppresses some components throughout training | Annealed softmax temperature, τ→1 by end of training | No — training schedule only |
| Numerical stability | log-sum-exp overflow/underflow; CDF-difference cancellation | Standard log-sum-exp trick; stable discretized-logistic identity; boundary-bin special case | No — arithmetic implementation only |
| Precision boundary | Silent calibration degradation under FP16/INT8 | Force FP32 on the decode submodule only | No — execution precision only |
| Spatial consistency | Discrete mode-selection flicker in homogeneous regions | Fixed, non-trainable spatial smoothing of π_k before argmax, at decode time only | No — post-processing only |
| Convergence / K-tuning order | K-sweep contaminated by uncorrected collapse | Tune stability fixes before K; cap scale parameter range | No — tuning protocol only |
| Deployment | Unsupported/poorly-fused decode ops; quantization can distort distribution shape undetected | Split export at backbone/decode boundary; validate NLL and sparsification parity, not just mean-accuracy parity, under quantization | No — export and validation protocol only |

Every modification in this analysis is a reparameterization, an initialization scheme, a training schedule, an arithmetic identity, an execution-precision boundary, or a post-processing/validation step — none alters π_k(T_B), μ_{k,c}(T_B), s_{k,c}(T_B), the discretized-logistic likelihood, or the NLL training objective fixed at the start of this review.
