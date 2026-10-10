# Legacy: Stable-Subspace Federated PEFT (stability study, Sep 2026)

> Archived README of the earlier stability-protocol phase (screen / stability v1-v4, DCEF, CAB). Paths and commands refer to that phase; the current entry point is the top-level README.


Reproducible federated RoBERTa experiments for communication-efficient PEFT.
The code compares dense LoRA, FFA-LoRA, FLASC, LoRA-XS, and fixed-factor
LoRA with bidirectional error-feedback sparsification. A preregistered
supplement also compares dense FedAvg-LoRA with full-A+B
FedAvg-LoRA+Top-k+EF at exact 10% and 20% wire budgets.

The post-hoc Generation-4 task-suite revision adds GLUE MRPC and QQP. Both
report accuracy and positive-class F1, with their mean used as the frozen
task-level primary score. Historical RTE results remain archived as negative
evidence but RTE is not part of the revised V4 stability gate.

## Layout

- `src/fedpeft/`: model adapters, data, compression, FL engine, metrics, reports
- `configs/roberta_screen.yaml`: experiment protocol
- `configs/roberta_full_lora_ef.yaml`: 27-run matched-source FedAvg wire
  supplement (9 dense references plus 18 compressed runs)
- `scripts/`: preflight, one-run, matrix, and report entry points
- `tests/`: CPU unit tests for partitioning, codecs, adapters, and aggregation

## Server quick start

```bash
cd /data/home/202422011138/bishe
export PYTHONPATH="$PWD/src:$PWD/.compat:$PWD/.deps"
export PYTHONPYCACHEPREFIX="$PWD/artifacts/pycache"
python3 scripts/preflight.py --config configs/roberta_screen.yaml
python3 scripts/run_matrix.py --config configs/roberta_screen.yaml
python3 scripts/build_report.py --config configs/roberta_screen.yaml
```

After the main matrix is complete and archived, the matched-wire supplement is
run with the same frozen seeds, schedules, and FedAvg learning rates:

```bash
python3 scripts/preflight.py --config configs/roberta_full_lora_ef.yaml
python3 scripts/run_matrix.py --config configs/roberta_full_lora_ef.yaml
python3 scripts/build_report.py --config configs/roberta_full_lora_ef.yaml
```

The matched dense-basis diagnostic is opt-in and does not change the formal
81-run matrix:

```bash
python3 scripts/run_matrix.py --config configs/roberta_screen.yaml \
  --include-diagnostics --methods random_a_dense,spectral_a_dense
```

Every run writes an atomic `summary.json`, per-round `round-NNN.json` files, and a
resumable checkpoint under `docs/results/`. The generated analysis Markdown is
stored beside the corresponding result set. The matrix runner skips completed runs.

The `flasc` entry preserves FLASC's global Top-k downlink, dense local adapter
training, and independent Top-k upload, while using the same FedAvg server rule
and AdamW client optimizer as the other methods for a controlled comparison.
Reports therefore label it as a **FLASC-style controlled baseline**, not as an
exact reproduction of the paper's optimizer configuration.

The frozen original campaign used SST-2, RTE, and MNLI; those outputs remain
immutable historical evidence.  New stability work uses Generation 4 and does
not launch RTE.  Its development matrix uses SST-2/MRPC/QQP, while MNLI enters
the preregistered Stage-B and confirmation suites.  MRPC and QQP report both
accuracy and label-1 binary F1 and use their arithmetic mean as the task-level
primary score.  Development configurations evaluate the pooled 10% train-only
holdout; confirmation evaluates official GLUE validation splits.  Final
results always use the fixed last round, never the best round. Communication
efficiency is reported as bytes to 95% of the paired dense FedAvg final primary
score and its speedup, with a per-task primary-score--wire Pareto frontier as a
complementary view.

Generation-4 implementation smoke and Stage A are launched only after their
respective preflight checks pass:

```bash
python3 scripts/preflight.py --config configs/roberta_stability_v4_smoke.yaml
python3 scripts/run_matrix.py --config configs/roberta_stability_v4_smoke.yaml
python3 scripts/preflight.py --config configs/roberta_stability_v4_stage_a.yaml
python3 scripts/run_matrix.py --config configs/roberta_stability_v4_stage_a.yaml
```

Delay-consistent error feedback (DCEF) adds `full_lora_dcef`, its client-only
ablation `full_lora_dcef_client`, and fixed-A variants `random_a_dcef` /
`spectral_a_dcef`. Each reuses the model construction of its EF counterpart and
differs only in the synchronisation rule; see
`docs/proposals/DCEF_IMPLEMENTATION.md`. The development matrix uses fresh
seeds 361-363 and train-holdout evaluation:

```bash
python3 scripts/preflight.py --config configs/roberta_dcef_smoke.yaml
python3 scripts/run_matrix.py --config configs/roberta_dcef_smoke.yaml --include-diagnostics
python3 scripts/preflight.py --config configs/roberta_dcef_dev.yaml
python3 scripts/run_matrix.py --config configs/roberta_dcef_dev.yaml
python3 scripts/build_dcef_report.py --config configs/roberta_dcef_dev.yaml
```

The communication-aware fixed basis (CAB) adds `cab_a_dcef` and its kappa = 0
ablation `cab0_a_dcef`: a fixed A chosen in closed form from one-time client
gradient factorisations, synchronised with DCEF, with all setup bytes counted.
Its development config is gated on gamma calibrated from the DCEF stage; see
`docs/proposals/CAB_IMPLEMENTATION.md`.

All generated experiment material is constrained to `docs/results/`:
per-round JSON, checkpoints, launcher logs, CSV tables, and Markdown reports.
Reusable tokenized data, partition manifests, spectral bases, and artifact
fingerprints live under `artifacts/` and are not treated as results.
