# DCEF / CAB campaign report

Config: `/data/home/202412012250/bishe_qnli/configs/roberta_qnli_dev.yaml`; evaluation policy: `train_holdout`; seeds: [42].

Runs: 19/19 technically valid; 0 missing; 0 invalid.

## Stability counts

| Task | Method | Wire | n | C95 reached | Final ≥ target | Strict stable | Mean final | Mean Speedup95 | Mean γ̄·d/K |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qnli | cab0_a_ef | 0.05 | 1 | 1 | 0 | 0 | 0.5810 | 3.6434 |  |
| qnli | cab0_a_ef | 0.1 | 1 | 0 | 0 | 0 | 0.6192 |  |  |
| qnli | cab0_a_ef | 0.2 | 1 | 0 | 0 | 0 | 0.6396 |  |  |
| qnli | cab0wh_a_ef | 0.05 | 1 | 0 | 0 | 0 | 0.7002 |  |  |
| qnli | cab0wh_a_ef | 0.1 | 1 | 1 | 0 | 0 | 0.7419 | 1.7596 |  |
| qnli | cab0wh_a_ef | 0.2 | 1 | 1 | 1 | 0 | 0.7680 | 1.4964 |  |
| qnli | fedavg_lora | dense | 1 | 1 | 1 | 0 | 0.7966 | 1.0000 |  |
| qnli | gev0_a_ef | 0.05 | 1 | 1 | 1 | 1 | 0.7726 | 2.9115 |  |
| qnli | gev0_a_ef | 0.1 | 1 | 1 | 1 | 0 | 0.7766 | 1.9292 |  |
| qnli | gev0_a_ef | 0.2 | 1 | 1 | 1 | 1 | 0.7895 | 1.1519 |  |
| qnli | random_a_ef | 0.05 | 1 | 0 | 0 | 0 | 0.6156 |  |  |
| qnli | random_a_ef | 0.1 | 1 | 0 | 0 | 0 | 0.7471 |  |  |
| qnli | random_a_ef | 0.2 | 1 | 1 | 1 | 0 | 0.7600 | 1.4289 |  |
| qnli | randomwh_a_ef | 0.05 | 1 | 0 | 0 | 0 | 0.6815 |  |  |
| qnli | randomwh_a_ef | 0.1 | 1 | 0 | 0 | 0 | 0.7523 |  |  |
| qnli | randomwh_a_ef | 0.2 | 1 | 1 | 1 | 0 | 0.7697 | 1.4591 |  |
| qnli | spectral_a_ef | 0.05 | 1 | 0 | 0 | 0 | 0.7216 |  |  |
| qnli | spectral_a_ef | 0.1 | 1 | 0 | 0 | 0 | 0.7050 |  |  |
| qnli | spectral_a_ef | 0.2 | 1 | 0 | 0 | 0 | 0.7170 |  |  |

## Paired differences (left − right, final primary score)

| Comparison | Task | Wire | n | Δ final mean ± SD | Win / Tie / Loss | Strict (left / right) | Post-hit violations (left / right) |
|---|---|---|---:|---:|---:|---:|---:|
| cab0_a_ef-random_a_ef | qnli | 0.05 | 1 | -0.0347 | 0 / 0 / 1 | 0 / 0 | 4 / 0 |
| cab0_a_ef-random_a_ef | qnli | 0.1 | 1 | -0.1278 | 0 / 0 / 1 | 0 / 0 | 0 / 0 |
| cab0_a_ef-random_a_ef | qnli | 0.2 | 1 | -0.1204 | 0 / 0 / 1 | 0 / 0 | 0 / 2 |
| cab0wh_a_ef-cab0_a_ef | qnli | 0.05 | 1 | +0.1192 | 1 / 0 / 0 | 0 / 0 | 0 / 4 |
| cab0wh_a_ef-cab0_a_ef | qnli | 0.1 | 1 | +0.1227 | 1 / 0 / 0 | 0 / 0 | 1 / 0 |
| cab0wh_a_ef-cab0_a_ef | qnli | 0.2 | 1 | +0.1284 | 1 / 0 / 0 | 0 / 0 | 1 / 0 |
| cab0wh_a_ef-gev0_a_ef | qnli | 0.05 | 1 | -0.0724 | 0 / 0 / 1 | 0 / 1 | 0 / 0 |
| cab0wh_a_ef-gev0_a_ef | qnli | 0.1 | 1 | -0.0347 | 0 / 0 / 1 | 0 / 0 | 1 / 0 |
| cab0wh_a_ef-gev0_a_ef | qnli | 0.2 | 1 | -0.0215 | 0 / 0 / 1 | 0 / 1 | 1 / 0 |
| gev0_a_ef-cab0_a_ef | qnli | 0.05 | 1 | +0.1916 | 1 / 0 / 0 | 1 / 0 | 0 / 4 |
| gev0_a_ef-cab0_a_ef | qnli | 0.1 | 1 | +0.1573 | 1 / 0 / 0 | 0 / 0 | 0 / 0 |
| gev0_a_ef-cab0_a_ef | qnli | 0.2 | 1 | +0.1499 | 1 / 0 / 0 | 1 / 0 | 0 / 0 |
| gev0_a_ef-random_a_ef | qnli | 0.05 | 1 | +0.1570 | 1 / 0 / 0 | 1 / 0 | 0 / 0 |
| gev0_a_ef-random_a_ef | qnli | 0.1 | 1 | +0.0295 | 1 / 0 / 0 | 0 / 0 | 0 / 0 |
| gev0_a_ef-random_a_ef | qnli | 0.2 | 1 | +0.0295 | 1 / 0 / 0 | 1 / 0 | 0 / 2 |
| gev0_a_ef-spectral_a_ef | qnli | 0.05 | 1 | +0.0510 | 1 / 0 / 0 | 1 / 0 | 0 / 0 |
| gev0_a_ef-spectral_a_ef | qnli | 0.1 | 1 | +0.0716 | 1 / 0 / 0 | 0 / 0 | 0 / 0 |
| gev0_a_ef-spectral_a_ef | qnli | 0.2 | 1 | +0.0725 | 1 / 0 / 0 | 1 / 0 | 0 / 0 |
| randomwh_a_ef-random_a_ef | qnli | 0.05 | 1 | +0.0659 | 1 / 0 / 0 | 0 / 0 | 0 / 0 |
| randomwh_a_ef-random_a_ef | qnli | 0.1 | 1 | +0.0053 | 1 / 0 / 0 | 0 / 0 | 0 / 0 |
| randomwh_a_ef-random_a_ef | qnli | 0.2 | 1 | +0.0097 | 1 / 0 / 0 | 0 / 0 | 2 / 2 |
| spectral_a_ef-random_a_ef | qnli | 0.05 | 1 | +0.1060 | 1 / 0 / 0 | 0 / 0 | 0 / 0 |
| spectral_a_ef-random_a_ef | qnli | 0.1 | 1 | -0.0421 | 0 / 0 / 1 | 0 / 0 | 0 / 0 |
| spectral_a_ef-random_a_ef | qnli | 0.2 | 1 | -0.0430 | 0 / 0 / 1 | 0 / 0 | 0 / 2 |

## CAB basis construction (zero-training diagnostics)

Objective = predicted stationary excess error J(P)/tr(S) under exact statistics and the run's kappa (lower is better); overlaps are ||A1 A2^T||_F^2 / r.

| Task | Method | Wire | n | κ | J selected / J exact | overlap ↔ spectral | J selected | J spectral | J random | setup / total bytes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qnli | cab0_a_ef | 0.05 | 1 | 0.0000 | 1.1023 | 0.0336 | 0.0125 | 0.9889 | 0.9895 | 0.332 |
| qnli | cab0_a_ef | 0.1 | 1 | 0.0000 | 1.1023 | 0.0336 | 0.0125 | 0.9889 | 0.9895 | 0.199 |
| qnli | cab0_a_ef | 0.2 | 1 | 0.0000 | 1.1023 | 0.0336 | 0.0125 | 0.9889 | 0.9895 | 0.111 |
| qnli | cab0wh_a_ef | 0.05 | 1 | 0.0000 | 1.1023 | 0.0336 | 0.0125 | 0.9889 | 0.9895 | 0.402 |
| qnli | cab0wh_a_ef | 0.1 | 1 | 0.0000 | 1.1023 | 0.0336 | 0.0125 | 0.9889 | 0.9895 | 0.252 |
| qnli | cab0wh_a_ef | 0.2 | 1 | 0.0000 | 1.1023 | 0.0336 | 0.0125 | 0.9889 | 0.9895 | 0.144 |
| qnli | gev0_a_ef | 0.05 | 1 | 0.0000 | 0.9990 | 0.0241 | 0.9707 | 0.9889 | 0.9895 | 0.402 |
| qnli | gev0_a_ef | 0.1 | 1 | 0.0000 | 0.9990 | 0.0241 | 0.9707 | 0.9889 | 0.9895 | 0.252 |
| qnli | gev0_a_ef | 0.2 | 1 | 0.0000 | 0.9990 | 0.0241 | 0.9707 | 0.9889 | 0.9895 | 0.144 |
| qnli | randomwh_a_ef | 0.05 | 1 | 0.0000 | 1.0000 | 0.0104 | 0.9895 | 0.9889 | 0.9895 | 0.255 |
| qnli | randomwh_a_ef | 0.1 | 1 | 0.0000 | 1.0000 | 0.0104 | 0.9895 | 0.9889 | 0.9895 | 0.146 |
| qnli | randomwh_a_ef | 0.2 | 1 | 0.0000 | 1.0000 | 0.0104 | 0.9895 | 0.9889 | 0.9895 | 0.079 |

C95 for CAB includes its one-time setup bytes (`c95_bytes`); `c95_training_bytes` in DCEF_RUNS.csv excludes them.

Delivered bytes differ between matched EF and DCEF runs only by the DCEF control packet and the at most three sparse entries it displaces; see `delta_total_bytes` in DCEF_PAIRS.csv.
Development-stage evidence on train-holdout scores; not a confirmation result.
