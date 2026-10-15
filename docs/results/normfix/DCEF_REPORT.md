# DCEF / CAB campaign report

Config: `/data/home/202412012250/bishe_norm/configs/roberta_normfix_main.yaml`; evaluation policy: `official_validation`; seeds: [1001, 1002, 1003].

Runs: 81/81 technically valid; 0 missing; 0 invalid.

## Stability counts

| Task | Method | Wire | n | C95 reached | Final ≥ target | Strict stable | Mean final | Mean Speedup95 | Mean γ̄·d/K |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mnli | eva_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.3274 |  |  |
| mnli | eva_a_ef | 0.1 | 3 | 0 | 0 | 0 | 0.3804 |  |  |
| mnli | eva_a_ef | 0.2 | 3 | 0 | 0 | 0 | 0.3388 |  |  |
| mnli | evawh_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7628 |  |  |
| mnli | evawh_a_ef | 0.1 | 3 | 0 | 0 | 0 | 0.7803 |  |  |
| mnli | evawh_a_ef | 0.2 | 3 | 0 | 0 | 0 | 0.7892 |  |  |
| mnli | fedsb_norm | 0.05 | 3 | 0 | 0 | 0 | 0.5359 |  |  |
| mnli | fedsb_norm | 0.1 | 3 | 0 | 0 | 0 | 0.6042 |  |  |
| mnli | fedsb_norm | 0.2 | 3 | 0 | 0 | 0 | 0.6633 |  |  |
| qnli | eva_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.5929 |  |  |
| qnli | eva_a_ef | 0.1 | 3 | 0 | 0 | 0 | 0.6401 |  |  |
| qnli | eva_a_ef | 0.2 | 3 | 0 | 0 | 0 | 0.6037 |  |  |
| qnli | evawh_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.8029 |  |  |
| qnli | evawh_a_ef | 0.1 | 3 | 0 | 0 | 0 | 0.8465 |  |  |
| qnli | evawh_a_ef | 0.2 | 3 | 0 | 0 | 0 | 0.8378 |  |  |
| qnli | fedsb_norm | 0.05 | 3 | 0 | 0 | 0 | 0.6837 |  |  |
| qnli | fedsb_norm | 0.1 | 3 | 0 | 0 | 0 | 0.7472 |  |  |
| qnli | fedsb_norm | 0.2 | 3 | 0 | 0 | 0 | 0.7575 |  |  |
| sst2 | eva_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.8135 |  |  |
| sst2 | eva_a_ef | 0.1 | 3 | 0 | 0 | 0 | 0.8578 |  |  |
| sst2 | eva_a_ef | 0.2 | 3 | 0 | 0 | 0 | 0.8555 |  |  |
| sst2 | evawh_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.6479 |  |  |
| sst2 | evawh_a_ef | 0.1 | 3 | 0 | 0 | 0 | 0.6380 |  |  |
| sst2 | evawh_a_ef | 0.2 | 3 | 0 | 0 | 0 | 0.6995 |  |  |
| sst2 | fedsb_norm | 0.05 | 3 | 0 | 0 | 0 | 0.7267 |  |  |
| sst2 | fedsb_norm | 0.1 | 3 | 0 | 0 | 0 | 0.7397 |  |  |
| sst2 | fedsb_norm | 0.2 | 3 | 0 | 0 | 0 | 0.8417 |  |  |

## Paired differences (left − right, final primary score)

| Comparison | Task | Wire | n | Δ final mean ± SD | Win / Tie / Loss | Strict (left / right) | Post-hit violations (left / right) |
|---|---|---|---:|---:|---:|---:|---:|

## CAB basis construction (zero-training diagnostics)

Objective = predicted stationary excess error J(P)/tr(S) under exact statistics and the run's kappa (lower is better); overlaps are ||A1 A2^T||_F^2 / r.

| Task | Method | Wire | n | κ | J selected / J exact | overlap ↔ spectral | J selected | J spectral | J random | setup / total bytes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mnli | eva_a_ef | 0.05 | 3 | 0.0000 | 1.0093 | 0.0615 | 0.0406 | 0.9872 | 0.9895 | 0.134 |
| mnli | eva_a_ef | 0.1 | 3 | 0.0000 | 1.0093 | 0.0615 | 0.0406 | 0.9872 | 0.9895 | 0.072 |
| mnli | eva_a_ef | 0.2 | 3 | 0.0000 | 1.0093 | 0.0615 | 0.0406 | 0.9872 | 0.9895 | 0.037 |
| mnli | evawh_a_ef | 0.05 | 3 | 0.0000 | 1.0093 | 0.0615 | 0.0406 | 0.9872 | 0.9895 | 0.134 |
| mnli | evawh_a_ef | 0.1 | 3 | 0.0000 | 1.0093 | 0.0615 | 0.0406 | 0.9872 | 0.9895 | 0.072 |
| mnli | evawh_a_ef | 0.2 | 3 | 0.0000 | 1.0093 | 0.0615 | 0.0406 | 0.9872 | 0.9895 | 0.037 |
| qnli | eva_a_ef | 0.05 | 3 | 0.0000 | 1.0161 | 0.0607 | 0.0394 | 0.9875 | 0.9893 | 0.206 |
| qnli | eva_a_ef | 0.1 | 3 | 0.0000 | 1.0161 | 0.0607 | 0.0394 | 0.9875 | 0.9893 | 0.115 |
| qnli | eva_a_ef | 0.2 | 3 | 0.0000 | 1.0161 | 0.0607 | 0.0394 | 0.9875 | 0.9893 | 0.061 |
| qnli | evawh_a_ef | 0.05 | 3 | 0.0000 | 1.0161 | 0.0607 | 0.0394 | 0.9875 | 0.9893 | 0.206 |
| qnli | evawh_a_ef | 0.1 | 3 | 0.0000 | 1.0161 | 0.0607 | 0.0394 | 0.9875 | 0.9893 | 0.115 |
| qnli | evawh_a_ef | 0.2 | 3 | 0.0000 | 1.0161 | 0.0607 | 0.0394 | 0.9875 | 0.9893 | 0.061 |
| sst2 | eva_a_ef | 0.05 | 3 | 0.0000 | 1.0021 | 0.0347 | 0.0176 | 0.9893 | 0.9895 | 0.206 |
| sst2 | eva_a_ef | 0.1 | 3 | 0.0000 | 1.0021 | 0.0347 | 0.0176 | 0.9893 | 0.9895 | 0.115 |
| sst2 | eva_a_ef | 0.2 | 3 | 0.0000 | 1.0021 | 0.0347 | 0.0176 | 0.9893 | 0.9895 | 0.061 |
| sst2 | evawh_a_ef | 0.05 | 3 | 0.0000 | 1.0021 | 0.0347 | 0.0176 | 0.9893 | 0.9895 | 0.206 |
| sst2 | evawh_a_ef | 0.1 | 3 | 0.0000 | 1.0021 | 0.0347 | 0.0176 | 0.9893 | 0.9895 | 0.115 |
| sst2 | evawh_a_ef | 0.2 | 3 | 0.0000 | 1.0021 | 0.0347 | 0.0176 | 0.9893 | 0.9895 | 0.061 |

C95 for CAB includes its one-time setup bytes (`c95_bytes`); `c95_training_bytes` in DCEF_RUNS.csv excludes them.

Delivered bytes differ between matched EF and DCEF runs only by the DCEF control packet and the at most three sparse entries it displaces; see `delta_total_bytes` in DCEF_PAIRS.csv.
Confirmation-stage evidence on official validation scores with seeds fixed before any run of this stage; schedules, learning rates and the basis protocol were frozen in the development stage.
