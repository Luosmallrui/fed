# DCEF / CAB campaign report

Config: `/data/home/202412012250/bishe_cab/configs/roberta_cab_ef_dev.yaml`; evaluation policy: `train_holdout`; seeds: [361, 362, 363].

Runs: 116/117 technically valid; 0 missing; 1 invalid.

Invalid runs:

- sst2 random_a_ef seed 362 wire 0.1: EF_LEDGER

## Stability counts

| Task | Method | Wire | n | C95 reached | Final ≥ target | Strict stable | Mean final | Mean Speedup95 | Mean γ̄·d/K |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mnli | cab0_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.5211 |  |  |
| mnli | cab0_a_ef | 0.1 | 3 | 0 | 0 | 0 | 0.6020 |  |  |
| mnli | cab0_a_ef | 0.2 | 3 | 0 | 0 | 0 | 0.6475 |  |  |
| mnli | cab_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7556 |  |  |
| mnli | cab_a_ef | 0.1 | 3 | 3 | 3 | 3 | 0.7906 | 4.8975 |  |
| mnli | cab_a_ef | 0.2 | 3 | 1 | 1 | 1 | 0.7182 | 3.4814 |  |
| mnli | fedavg_lora | dense | 3 | 3 | 3 | 2 | 0.8192 | 1.0000 |  |
| mnli | random_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7697 |  |  |
| mnli | random_a_ef | 0.1 | 3 | 2 | 2 | 2 | 0.7801 | 4.4435 |  |
| mnli | random_a_ef | 0.2 | 3 | 3 | 3 | 3 | 0.7904 | 2.6851 |  |
| mnli | spectral_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7663 |  |  |
| mnli | spectral_a_ef | 0.1 | 3 | 2 | 2 | 2 | 0.7823 | 4.4435 |  |
| mnli | spectral_a_ef | 0.2 | 3 | 3 | 3 | 2 | 0.7897 | 2.8967 |  |
| rte | cab0_a_ef | 0.05 | 3 | 3 | 3 | 0 | 0.5436 | 3.0908 |  |
| rte | cab0_a_ef | 0.1 | 3 | 3 | 3 | 1 | 0.5824 | 2.3906 |  |
| rte | cab0_a_ef | 0.2 | 3 | 3 | 3 | 0 | 0.5758 | 2.2405 |  |
| rte | cab_a_ef | 0.05 | 3 | 3 | 3 | 2 | 0.5986 | 4.8749 |  |
| rte | cab_a_ef | 0.1 | 3 | 3 | 3 | 2 | 0.6430 | 4.7083 |  |
| rte | cab_a_ef | 0.2 | 3 | 3 | 3 | 1 | 0.5933 | 3.0242 |  |
| rte | fedavg_lora | dense | 3 | 3 | 3 | 0 | 0.5490 | 1.0000 |  |
| rte | random_a_ef | 0.05 | 3 | 3 | 2 | 0 | 0.5261 | 12.9108 |  |
| rte | random_a_ef | 0.1 | 3 | 3 | 1 | 0 | 0.5194 | 5.6155 |  |
| rte | random_a_ef | 0.2 | 3 | 3 | 2 | 1 | 0.5489 | 3.1241 |  |
| rte | spectral_a_ef | 0.05 | 3 | 2 | 1 | 1 | 0.4846 | 11.8984 |  |
| rte | spectral_a_ef | 0.1 | 3 | 2 | 1 | 1 | 0.4886 | 6.0386 |  |
| rte | spectral_a_ef | 0.2 | 3 | 2 | 2 | 1 | 0.5006 | 3.0422 |  |
| sst2 | cab0_a_ef | 0.05 | 3 | 2 | 2 | 2 | 0.8125 | 4.7345 |  |
| sst2 | cab0_a_ef | 0.1 | 3 | 2 | 2 | 1 | 0.8486 | 3.1672 |  |
| sst2 | cab0_a_ef | 0.2 | 3 | 2 | 2 | 0 | 0.8525 | 2.4930 |  |
| sst2 | cab_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7995 |  |  |
| sst2 | cab_a_ef | 0.1 | 3 | 2 | 2 | 1 | 0.8099 | 3.2409 |  |
| sst2 | cab_a_ef | 0.2 | 3 | 2 | 2 | 1 | 0.8620 | 2.5140 |  |
| sst2 | fedavg_lora | dense | 3 | 3 | 3 | 1 | 0.9027 | 1.0000 |  |
| sst2 | random_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.8054 |  |  |
| sst2 | random_a_ef | 0.1 | 2 | 1 | 1 | 1 | 0.8306 | 2.5924 |  |
| sst2 | random_a_ef | 0.2 | 3 | 2 | 1 | 1 | 0.8486 | 1.8679 |  |
| sst2 | spectral_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.6537 |  |  |
| sst2 | spectral_a_ef | 0.1 | 3 | 1 | 1 | 1 | 0.7457 | 4.3985 |  |
| sst2 | spectral_a_ef | 0.2 | 3 | 2 | 2 | 2 | 0.7662 | 1.9346 |  |

## Paired differences (left − right, final primary score)

| Comparison | Task | Wire | n | Δ final mean ± SD | Win / Tie / Loss | Strict (left / right) | Post-hit violations (left / right) |
|---|---|---|---:|---:|---:|---:|---:|
| cab0_a_ef-random_a_ef | mnli | 0.05 | 3 | -0.2486 ± 0.0958 | 0 / 0 / 3 | 0 / 0 | 0 / 0 |
| cab0_a_ef-random_a_ef | mnli | 0.1 | 3 | -0.1781 ± 0.0120 | 0 / 0 / 3 | 0 / 2 | 0 / 0 |
| cab0_a_ef-random_a_ef | mnli | 0.2 | 3 | -0.1429 ± 0.0143 | 0 / 0 / 3 | 0 / 3 | 0 / 0 |
| cab0_a_ef-random_a_ef | rte | 0.05 | 3 | +0.0174 ± 0.0291 | 2 / 0 / 1 | 0 / 0 | 29 / 61 |
| cab0_a_ef-random_a_ef | rte | 0.1 | 3 | +0.0630 ± 0.0769 | 2 / 1 / 0 | 1 / 0 | 7 / 56 |
| cab0_a_ef-random_a_ef | rte | 0.2 | 3 | +0.0269 ± 0.0308 | 2 / 1 / 0 | 0 / 1 | 18 / 42 |
| cab0_a_ef-random_a_ef | sst2 | 0.05 | 3 | +0.0072 ± 0.0849 | 2 / 0 / 1 | 2 / 0 | 0 / 0 |
| cab0_a_ef-random_a_ef | sst2 | 0.1 | 2 | +0.0061 ± 0.0031 | 2 / 0 / 0 | 0 / 1 | 0 / 0 |
| cab0_a_ef-random_a_ef | sst2 | 0.2 | 3 | +0.0040 ± 0.0186 | 1 / 0 / 2 | 0 / 1 | 4 / 1 |
| cab_a_ef-cab0_a_ef | mnli | 0.05 | 3 | +0.2346 ± 0.0954 | 3 / 0 / 0 | 0 / 0 | 0 / 0 |
| cab_a_ef-cab0_a_ef | mnli | 0.1 | 3 | +0.1886 ± 0.0198 | 3 / 0 / 0 | 3 / 0 | 0 / 0 |
| cab_a_ef-cab0_a_ef | mnli | 0.2 | 3 | +0.0707 ± 0.0750 | 2 / 1 / 0 | 1 / 0 | 0 / 0 |
| cab_a_ef-cab0_a_ef | rte | 0.05 | 3 | +0.0551 ± 0.0308 | 3 / 0 / 0 | 2 / 0 | 22 / 29 |
| cab_a_ef-cab0_a_ef | rte | 0.1 | 3 | +0.0605 ± 0.0911 | 2 / 0 / 1 | 2 / 1 | 22 / 7 |
| cab_a_ef-cab0_a_ef | rte | 0.2 | 3 | +0.0175 ± 0.0343 | 1 / 2 / 0 | 1 / 0 | 26 / 18 |
| cab_a_ef-cab0_a_ef | sst2 | 0.05 | 3 | -0.0130 ± 0.1466 | 1 / 0 / 2 | 0 / 2 | 0 / 0 |
| cab_a_ef-cab0_a_ef | sst2 | 0.1 | 3 | -0.0387 ± 0.1562 | 2 / 0 / 1 | 1 / 1 | 0 / 0 |
| cab_a_ef-cab0_a_ef | sst2 | 0.2 | 3 | +0.0095 ± 0.0735 | 1 / 0 / 2 | 1 / 0 | 2 / 4 |
| cab_a_ef-random_a_ef | mnli | 0.05 | 3 | -0.0141 ± 0.0074 | 0 / 0 / 3 | 0 / 0 | 0 / 0 |
| cab_a_ef-random_a_ef | mnli | 0.1 | 3 | +0.0105 ± 0.0114 | 2 / 1 / 0 | 3 / 2 | 0 / 0 |
| cab_a_ef-random_a_ef | mnli | 0.2 | 3 | -0.0722 ± 0.0847 | 1 / 0 / 2 | 1 / 3 | 0 / 0 |
| cab_a_ef-random_a_ef | rte | 0.05 | 3 | +0.0725 ± 0.0122 | 3 / 0 / 0 | 2 / 0 | 22 / 61 |
| cab_a_ef-random_a_ef | rte | 0.1 | 3 | +0.1235 ± 0.0170 | 3 / 0 / 0 | 2 / 0 | 22 / 56 |
| cab_a_ef-random_a_ef | rte | 0.2 | 3 | +0.0443 ± 0.0630 | 2 / 1 / 0 | 1 / 1 | 26 / 42 |
| cab_a_ef-random_a_ef | sst2 | 0.05 | 3 | -0.0058 ± 0.0617 | 1 / 0 / 2 | 0 / 0 | 0 / 0 |
| cab_a_ef-random_a_ef | sst2 | 0.1 | 2 | +0.0539 ± 0.0653 | 2 / 0 / 0 | 1 / 1 | 0 / 0 |
| cab_a_ef-random_a_ef | sst2 | 0.2 | 3 | +0.0134 ± 0.0583 | 1 / 0 / 2 | 1 / 1 | 2 / 1 |
| cab_a_ef-spectral_a_ef | mnli | 0.05 | 3 | -0.0107 ± 0.0068 | 0 / 0 / 3 | 0 / 0 | 0 / 0 |
| cab_a_ef-spectral_a_ef | mnli | 0.1 | 3 | +0.0083 ± 0.0182 | 1 / 2 / 0 | 3 / 2 | 0 / 0 |
| cab_a_ef-spectral_a_ef | mnli | 0.2 | 3 | -0.0715 ± 0.0850 | 1 / 0 / 2 | 1 / 2 | 0 / 1 |
| cab_a_ef-spectral_a_ef | rte | 0.05 | 3 | +0.1141 ± 0.0383 | 3 / 0 / 0 | 2 / 1 | 22 / 34 |
| cab_a_ef-spectral_a_ef | rte | 0.1 | 3 | +0.1544 ± 0.0444 | 3 / 0 / 0 | 2 / 1 | 22 / 25 |
| cab_a_ef-spectral_a_ef | rte | 0.2 | 3 | +0.0927 ± 0.1037 | 3 / 0 / 0 | 1 / 1 | 26 / 20 |
| cab_a_ef-spectral_a_ef | sst2 | 0.05 | 3 | +0.1459 ± 0.1349 | 2 / 1 / 0 | 0 / 0 | 0 / 0 |
| cab_a_ef-spectral_a_ef | sst2 | 0.1 | 3 | +0.0641 ± 0.0365 | 3 / 0 / 0 | 1 / 1 | 0 / 0 |
| cab_a_ef-spectral_a_ef | sst2 | 0.2 | 3 | +0.0957 ± 0.1461 | 3 / 0 / 0 | 1 / 2 | 2 / 0 |
| spectral_a_ef-random_a_ef | mnli | 0.05 | 3 | -0.0034 ± 0.0026 | 0 / 2 / 1 | 0 / 0 | 0 / 0 |
| spectral_a_ef-random_a_ef | mnli | 0.1 | 3 | +0.0023 ± 0.0072 | 2 / 0 / 1 | 2 / 2 | 0 / 0 |
| spectral_a_ef-random_a_ef | mnli | 0.2 | 3 | -0.0007 ± 0.0107 | 2 / 0 / 1 | 2 / 3 | 1 / 0 |
| spectral_a_ef-random_a_ef | rte | 0.05 | 3 | -0.0416 ± 0.0335 | 0 / 1 / 2 | 1 / 0 | 34 / 61 |
| spectral_a_ef-random_a_ef | rte | 0.1 | 3 | -0.0309 ± 0.0348 | 0 / 1 / 2 | 1 / 0 | 25 / 56 |
| spectral_a_ef-random_a_ef | rte | 0.2 | 3 | -0.0483 ± 0.0465 | 0 / 1 / 2 | 1 / 1 | 20 / 42 |
| spectral_a_ef-random_a_ef | sst2 | 0.05 | 3 | -0.1517 ± 0.1857 | 1 / 0 / 2 | 0 / 0 | 0 / 0 |
| spectral_a_ef-random_a_ef | sst2 | 0.1 | 2 | +0.0096 ± 0.0831 | 1 / 0 / 1 | 1 / 1 | 0 / 0 |
| spectral_a_ef-random_a_ef | sst2 | 0.2 | 3 | -0.0823 ± 0.1884 | 1 / 0 / 2 | 2 / 1 | 0 / 1 |

## CAB basis construction (zero-training diagnostics)

Objective = predicted stationary excess error J(P)/tr(S) under exact statistics and the run's kappa (lower is better); overlaps are ||A1 A2^T||_F^2 / r.

| Task | Method | Wire | n | κ | J selected / J exact | overlap ↔ spectral | J selected | J spectral | J random | setup / total bytes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mnli | cab0_a_ef | 0.05 | 3 | 0.0000 | 1.0861 | 0.0367 | 0.0094 | 0.9878 | 0.9894 | 0.229 |
| mnli | cab0_a_ef | 0.1 | 3 | 0.0000 | 1.0861 | 0.0367 | 0.0094 | 0.9878 | 0.9894 | 0.130 |
| mnli | cab0_a_ef | 0.2 | 3 | 0.0000 | 1.0861 | 0.0367 | 0.0094 | 0.9878 | 0.9894 | 0.069 |
| mnli | cab_a_ef | 0.05 | 3 | 0.4680 | 1.0024 | 0.0257 | 0.9950 | 1.0341 | 1.0514 | 0.229 |
| mnli | cab_a_ef | 0.1 | 3 | 0.1630 | 1.0021 | 0.0374 | 0.9624 | 1.0040 | 1.0111 | 0.130 |
| mnli | cab_a_ef | 0.2 | 3 | 0.0664 | 1.0020 | 0.0376 | 0.7408 | 0.9944 | 0.9983 | 0.069 |
| rte | cab0_a_ef | 0.05 | 3 | 0.0000 | 1.0945 | 0.0351 | 0.0239 | 0.9886 | 0.9895 | 0.230 |
| rte | cab0_a_ef | 0.1 | 3 | 0.0000 | 1.0945 | 0.0351 | 0.0239 | 0.9886 | 0.9895 | 0.130 |
| rte | cab0_a_ef | 0.2 | 3 | 0.0000 | 1.0945 | 0.0351 | 0.0239 | 0.9886 | 0.9895 | 0.069 |
| rte | cab_a_ef | 0.05 | 3 | 0.6183 | 1.0091 | 0.0255 | 0.9974 | 1.2298 | 1.3455 | 0.230 |
| rte | cab_a_ef | 0.1 | 3 | 0.2029 | 1.0054 | 0.0315 | 0.9836 | 1.0739 | 1.1154 | 0.130 |
| rte | cab_a_ef | 0.2 | 3 | 0.0816 | 1.0040 | 0.0332 | 0.9687 | 1.0236 | 1.0412 | 0.069 |
| sst2 | cab0_a_ef | 0.05 | 3 | 0.0000 | 1.0786 | 0.0242 | 0.0056 | 0.9900 | 0.9895 | 0.332 |
| sst2 | cab0_a_ef | 0.1 | 3 | 0.0000 | 1.0786 | 0.0242 | 0.0056 | 0.9900 | 0.9895 | 0.199 |
| sst2 | cab0_a_ef | 0.2 | 3 | 0.0000 | 1.0786 | 0.0242 | 0.0056 | 0.9900 | 0.9895 | 0.111 |
| sst2 | cab_a_ef | 0.05 | 3 | 0.6221 | 1.0024 | 0.0236 | 0.9948 | 1.0567 | 1.0779 | 0.332 |
| sst2 | cab_a_ef | 0.1 | 3 | 0.2068 | 1.0013 | 0.0246 | 0.9181 | 1.0119 | 1.0188 | 0.199 |
| sst2 | cab_a_ef | 0.2 | 3 | 0.0833 | 1.0011 | 0.0254 | 0.7452 | 0.9988 | 1.0013 | 0.111 |

C95 for CAB includes its one-time setup bytes (`c95_bytes`); `c95_training_bytes` in DCEF_RUNS.csv excludes them.

Delivered bytes differ between matched EF and DCEF runs only by the DCEF control packet and the at most three sparse entries it displaces; see `delta_total_bytes` in DCEF_PAIRS.csv.
Development-stage evidence on train-holdout scores; not a confirmation result.
