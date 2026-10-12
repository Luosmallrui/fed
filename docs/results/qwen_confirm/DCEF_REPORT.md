# DCEF / CAB campaign report

Config: `/data/home/202412012250/bishe_llm3/configs/qwen05b_confirm.yaml`; evaluation policy: `official_validation`; seeds: [1001, 1002, 1003].

Runs: 21/21 technically valid; 0 missing; 0 invalid.

## Stability counts

| Task | Method | Wire | n | C95 reached | Final ≥ target | Strict stable | Mean final | Mean Speedup95 | Mean γ̄·d/K |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mnli | cab0_a_ef | 0.05 | 3 | 1 | 0 | 0 | 0.6362 | 4.6333 |  |
| mnli | cab0_a_ef | 0.1 | 3 | 0 | 0 | 0 | 0.5880 |  |  |
| mnli | fedavg_lora | dense | 3 | 3 | 3 | 0 | 0.7642 | 1.0000 |  |
| mnli | gev0_a_ef | 0.05 | 3 | 3 | 3 | 0 | 0.7543 | 7.6688 |  |
| mnli | gev0_a_ef | 0.1 | 3 | 3 | 3 | 0 | 0.7578 | 6.3321 |  |
| mnli | random_a_ef | 0.05 | 3 | 3 | 3 | 0 | 0.7573 | 10.5600 |  |
| mnli | random_a_ef | 0.1 | 3 | 3 | 3 | 1 | 0.7567 | 5.9969 |  |

## Paired differences (left − right, final primary score)

| Comparison | Task | Wire | n | Δ final mean ± SD | Win / Tie / Loss | Strict (left / right) | Post-hit violations (left / right) |
|---|---|---|---:|---:|---:|---:|---:|
| cab0_a_ef-random_a_ef | mnli | 0.05 | 3 | -0.1211 ± 0.0537 | 0 / 0 / 3 | 0 / 0 | 3 / 2 |
| cab0_a_ef-random_a_ef | mnli | 0.1 | 3 | -0.1687 ± 0.1001 | 0 / 0 / 3 | 0 / 1 | 0 / 0 |
| gev0_a_ef-cab0_a_ef | mnli | 0.05 | 3 | +0.1181 ± 0.0544 | 3 / 0 / 0 | 0 / 0 | 2 / 3 |
| gev0_a_ef-cab0_a_ef | mnli | 0.1 | 3 | +0.1698 ± 0.1003 | 3 / 0 / 0 | 0 / 0 | 3 / 0 |
| gev0_a_ef-random_a_ef | mnli | 0.05 | 3 | -0.0029 ± 0.0129 | 1 / 0 / 2 | 0 / 0 | 2 / 2 |
| gev0_a_ef-random_a_ef | mnli | 0.1 | 3 | +0.0011 ± 0.0222 | 2 / 0 / 1 | 0 / 1 | 3 / 0 |

## CAB basis construction (zero-training diagnostics)

Objective = predicted stationary excess error J(P)/tr(S) under exact statistics and the run's kappa (lower is better); overlaps are ||A1 A2^T||_F^2 / r.

| Task | Method | Wire | n | κ | J selected / J exact | overlap ↔ spectral | J selected | J spectral | J random | setup / total bytes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mnli | cab0_a_ef | 0.05 | 3 | 0.0000 | 2.6378 | 0.0161 | 0.0058 | 0.9820 | 0.9914 | 0.140 |
| mnli | cab0_a_ef | 0.1 | 3 | 0.0000 | 2.6378 | 0.0161 | 0.0058 | 0.9820 | 0.9914 | 0.075 |
| mnli | gev0_a_ef | 0.05 | 3 | 0.0000 | 1.1105 | 0.0142 | 0.8965 | 0.9820 | 0.9913 | 0.146 |
| mnli | gev0_a_ef | 0.1 | 3 | 0.0000 | 1.1105 | 0.0142 | 0.8965 | 0.9820 | 0.9913 | 0.079 |

C95 for CAB includes its one-time setup bytes (`c95_bytes`); `c95_training_bytes` in DCEF_RUNS.csv excludes them.

Delivered bytes differ between matched EF and DCEF runs only by the DCEF control packet and the at most three sparse entries it displaces; see `delta_total_bytes` in DCEF_PAIRS.csv.
Confirmation-stage evidence on official validation scores with seeds fixed before any run of this stage; schedules, learning rates and the basis protocol were frozen in the development stage.
