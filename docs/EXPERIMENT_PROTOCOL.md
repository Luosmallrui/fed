# Experiment protocol

## Question

Does a shared pretrained right-singular subspace improve bidirectionally
compressed federated LoRA at a fixed serialized wire budget?

## Main matrix

Backbone: RoBERTa-base. Tasks: SST-2, RTE, and MNLI. Partition seeds:
401/402/403. Ten cross-silo clients participate in every round. Training data
are split with label-Dirichlet alpha 0.3; 10% of each client shard is retained
for development and client diagnostics. Official GLUE validation data are used
only by the locked final campaign. Older debug calibrations that inspected
official validation are retained as provenance but excluded from confirmatory
comparisons.

Dense references run once per task and seed:

1. `fedavg_lora`: train and synchronize both LoRA factors.
2. `ffa_lora`: freeze a standard random A and synchronize B.
3. `lora_xs`: freeze top-SVD left/right bases and synchronize an r-by-r core.

Sparse methods run at 10% and 20% of the serialized full-LoRA message cap:

4. `flasc`: global Top-k downlink of adapter state and independent Top-k
   upload of the locally dense-trained adapter delta; no residual.
5. `random_a_ef`: fixed row-orthonormal random A, B-only training, client
   upload and server downlink error feedback.
6. `spectral_a_ef`: identical to `random_a_ef`, except A is the top right
   singular subspace of the frozen pretrained weight.

The resulting main matrix contains 81 unique runs. `random_a_topk`,
`spectral_a_topk`, `random_a_dense`, and `spectral_a_dense` are diagnostic
methods and are not included unless explicitly requested. The two dense
fixed-factor diagnostics differ only in whether their row-orthonormal frozen A
is random or the pretrained right-singular subspace; both communicate B and
the classifier head densely at their natural recurrent wire.

The preregistered matched-wire supplement adds `full_lora_ef`, reported as
**FedAvg-LoRA+Top-k+EF**. Both LoRA factors remain trainable exactly as in
dense FedAvg-LoRA, while the complete A+B update uses the same bidirectional
Top-k/error-feedback path at exact 10% and 20% serialized caps. The supplement
also re-runs dense FedAvg-LoRA under the amended source, for 27 runs in total
(9 dense references plus 18 compressed runs). This keeps each paired
compression-cost comparison under one source hash. All optimizer and schedule
choices are inherited from the frozen main experiment; formal seeds are not
used for further tuning.

## Shared choices

- Rank 8, alpha 16, query/value projections only.
- FedAvg-LoRA and FFA-LoRA use PEFT's default Kaiming-uniform A
  initialization; random/spectral fixed-factor methods use matched
  row-orthonormal bases.
- A single linear `<s>` classification head replaces RoBERTa's large native
  two-layer head. It is trained, synchronized densely, and counted in wire.
- The pretrained encoder is frozen.
- Client AdamW state is reset for every client participation.
- Adapter and classifier-head learning rates are separate optimizer groups.
  Within both matched Random-A/Spectral-A pairs (EF and dense), adapter and
  classifier-head learning rates are identical.
- Client updates are weighted by complete client training-shard size.
- Batches are sampled on GPU by deterministic, cycling permutations.
- Model state and every update are consumed only after an actual
  FP16 serialization/deserialization round trip.
- The final schedules are fixed before seeds 401/402/403 are run: SST-2 uses
  30 rounds × 8 local steps × batch 128; RTE uses 50 × 8 × 16; MNLI uses
  50 × 16 × 128. MNLI is evaluated every five rounds and always on the final
  round; the other tasks are evaluated every round.

## Wire protocol

The full rank-8 LoRA adapter plus the task head, encoded as FP16, defines 100%
of one directional recurrent message. Sparse messages use a fixed header,
uint32 flattened indices, and FP16 values. The solver chooses the maximum K
whose encoded adapter plus dense head does not exceed the 10% or 20% cap.

Reported traffic separates:

- uplink payload;
- server-emitted downlink payload;
- delivered downlink payload (broadcast size times client count);
- one-time schema payload.

The pretrained backbone is assumed to be provisioned before federated
training. Random and spectral fixed bases are reconstructed from the shared
base model after canonicalizing SVD signs and are accepted only after a
SHA-256 handshake. Recurrent-wire tables assume successful reconstruction;
the report separately lists handshake bytes and the FP16 fallback cost if a
basis must be distributed. One-time SVD compute is also reported.

## Interpretation

- Spectral basis effect: paired `spectral_a_ef - random_a_ef` at identical
  task, partition seed, and wire cap.
- Uncompressed basis effect: paired `spectral_a_dense - random_a_dense` at
  identical task and partition seed, isolating basis choice from sparsification
  and error feedback.
- Compression reference: report proposed method versus FLASC at identical wire
  cap descriptively. FLASC uses its reference-style masked-server metric while
  fixed-A methods evaluate a communicated-global state, so this delta is not a
  same-state causal comparison.
- Fixed-factor cost: FFA-LoRA versus dense FedAvg-LoRA.
- Extreme parameter-efficiency reference: LoRA-XS versus all methods using its
  natural dense recurrent wire; rank is not silently increased to fill a cap.
- Full-LoRA compression cost: paired `full_lora_ef - fedavg_lora` at the same
  task and partition seed, reported separately at exact 10% and 20% wire.
- EF causality requires the optional no-EF diagnostic methods.

The primary scalar communication-efficiency measure is `C@95%-Dense`, the
cumulative delivered bytes at the first recorded evaluation that reaches 95%
of the same-seed dense FedAvg-LoRA final accuracy. The corresponding speedup is
dense `C@95` divided by method `C@95`. A per-task accuracy--wire Pareto frontier
is a complementary trade-off view. Endpoint accuracy change, retention, and
wire saving are supporting measures; `accuracy / bytes` is not treated as a
primary ranking statistic.

Three seeds support paired descriptive evidence, not a claim of statistical
significance. Failed and incomplete runs remain visible in the report.

For the FLASC-style baseline, the primary score follows the reference
implementation's masked-server convention (`global_after_update ×
round_start_mask`). Its dense-server score is stored separately. The masked
score is not described as the literal state already present at a client,
because its values include the just-aggregated update.
