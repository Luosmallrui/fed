# ACL-oriented paper plan

## Central question

At the same **serialized bidirectional communication budget**, does fixing the
LoRA input factor to a pretrained spectral subspace improve federated PEFT over
a matched random subspace and established communication-efficient baselines?

The defensible primary claim is a controlled empirical claim about
accuracy–wire trade-offs. It should not be framed as a universal convergence or
systems-speed claim.

## Main comparisons

- FedAvg-LoRA: full two-factor LoRA, natural dense recurrent wire.
- FFA-LoRA: fixed PEFT-initialized A, dense B update.
- LoRA-XS: fixed SVD factors and trainable rank-by-rank core, natural wire.
- FLASC-style: full LoRA trained locally with Top-k upload/downlink under the
  same optimizer and aggregation protocol; report its reference masked-server
  metric and dense-server diagnostic separately.
- Random-A+EF: row-orthonormal random fixed A and bidirectional error feedback.
- Spectral-A+EF: matched to Random-A+EF except for the pretrained right-singular
  basis.
- Supplementary matched-wire control: FedAvg-LoRA+Top-k+EF trains both LoRA
  factors exactly like FedAvg-LoRA but applies the same bidirectional Top-k and
  error-feedback transport at exact 10%/20% wire. Re-run dense FedAvg-LoRA in
  the supplementary source revision so its paired compression-cost comparison
  has one source hash.

Optional `Random-A dense` and `Spectral-A dense` diagnostics use the same
row-orthonormal fixed-factor parameterization and natural dense recurrent wire.
Their paired difference isolates basis choice without Top-k or error feedback;
they are not members of the formal six-method matrix.

Dense methods are natural-wire references; only the three sparse methods are
assigned 10% and 20% serialized caps. Do not duplicate a dense run and label it
as two budgets.

## Required evidence

1. Raw scores for every task, seed, method, and budget.
2. Mean ± sample SD over seeds 401/402/403.
3. Paired Spectral-A − Random-A deltas for each task and budget.
4. Descriptive Spectral-A − FLASC deltas at equal wire, explicitly marked as
   reference-style masked versus communicated-global metrics rather than a
   same-state causal comparison.
5. Exact delivered bytes, per-packet cap utilization, and natural dense-wire
   ratios.
6. MNLI matched and mismatched scores separately.
7. Runtime decomposition: data preparation, SVD, local training, compression,
   evaluation, resume count, and peak GPU memory.
8. Technical gates: source/model/data/partition/layout hashes, frozen backbone
   and basis, codec round-trip, cap compliance, and cross-round EF ledger.
9. The primary scalar communication-efficiency measure is cumulative delivered
   bytes and speedup to reach 95% of the same-seed dense FedAvg-LoRA final
   accuracy. Per-task accuracy--wire Pareto membership is a complementary
   trade-off view, not a scalar winner. FLASC remains separately marked because
   its reference primary metric uses a different model state.
10. For FedAvg-LoRA+Top-k+EF, paired final-round accuracy change, retention,
    and wire saving versus dense FedAvg-LoRA at exact 10% and 20% caps.

With only three seeds, report paired descriptive evidence and practical-effect
thresholds; do not claim statistical significance. The preregistered thresholds
are 0.3 percentage points for SST-2/MNLI and 1.0 point for RTE.

## Ablations needed for module attribution

- Random-A+Top-k versus Random-A+EF: joint bidirectional EF effect.
- Spectral-A+Top-k versus Spectral-A+EF: joint bidirectional EF effect under the
  proposed basis.
- Random-A dense versus Spectral-A dense: matched uncompressed basis diagnostic
  without recurrent sparsification or error feedback.

These diagnostics do not separate uplink EF from downlink EF. A stronger paper
would add client-only and server-only EF, but the current report must mark that
causal split as unresolved.

## Hyperparameter policy

- Tune only on partition seed 311 and the pooled train-only holdout.
- Compare at least the common `(adapter LR, head LR)=(3e-4, 3e-4)` setting with
  `(1e-3, 1e-3)` for every method family.
- Random-A and Spectral-A must always share both learning rates.
- Freeze all choices before viewing seeds 401/402/403.
- Report the fixed final round; best-round accuracy is secondary diagnostics.

## Suggested paper structure

1. Introduction: serialized wire, not parameter density, is the systems
   constraint; basis choice is the scientific question.
2. Method: fixed A, spectral construction, bidirectional EF, exact codec.
3. Experimental protocol: shared data/optimizer/head, three frozen partitions,
   true 10%/20% caps.
4. Main results: accuracy–wire table and Pareto plot.
5. Mechanism analysis: spectral-vs-random pairs, EF ablations, residual/support
   dynamics, and task/budget interaction.
6. Systems analysis: delivered traffic, setup fallback, runtime, and memory.
7. Limitations: three seeds, validation rather than GLUE test server, controlled
   FLASC reproduction and metric-state mismatch, and no separate
   uplink/downlink EF attribution.

## Migration from the current `main.tex`

The existing manuscript describes Qwen, Banking77, four clients, NF4, and old
results. It should be treated as a source of motivation only. Results and method
sections must be rewritten from the validated RoBERTa/GLUE JSON outputs; no old
number should be copied into the new tables.
