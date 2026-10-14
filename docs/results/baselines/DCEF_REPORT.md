# DCEF / CAB campaign report

Config: `/data/home/202412012250/bishe_base/configs/roberta_baselines_main.yaml`; evaluation policy: `official_validation`; seeds: [1001, 1002, 1003].

Runs: 144/144 technically valid; 0 missing; 0 invalid.

## Stability counts

| Task | Method | Wire | n | C95 reached | Final ≥ target | Strict stable | Mean final | Mean Speedup95 | Mean γ̄·d/K |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mnli | fedex_dense | dense | 3 | 0 | 0 | 0 | 0.8098 |  |  |
| mnli | fedsb | 0.05 | 3 | 0 | 0 | 0 | 0.5287 |  |  |
| mnli | fedsb | 0.1 | 3 | 0 | 0 | 0 | 0.5983 |  |  |
| mnli | fedsb | 0.2 | 3 | 0 | 0 | 0 | 0.6730 |  |  |
| mnli | fedsvd_dense | dense | 3 | 0 | 0 | 0 | 0.7898 |  |  |
| mnli | fedsvd_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7666 |  |  |
| mnli | fedsvd_ef | 0.1 | 3 | 0 | 0 | 0 | 0.7791 |  |  |
| mnli | fedsvd_ef | 0.2 | 3 | 0 | 0 | 0 | 0.7824 |  |  |
| mnli | flora_dense | dense | 3 | 0 | 0 | 0 | 0.7808 |  |  |
| mnli | full_lora_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7651 |  |  |
| mnli | full_lora_ef | 0.1 | 3 | 0 | 0 | 0 | 0.7727 |  |  |
| mnli | full_lora_ef | 0.2 | 3 | 0 | 0 | 0 | 0.8046 |  |  |
| mnli | rolora_dense | dense | 3 | 0 | 0 | 0 | 0.7959 |  |  |
| mnli | rolora_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7413 |  |  |
| mnli | rolora_ef | 0.1 | 3 | 0 | 0 | 0 | 0.7654 |  |  |
| mnli | rolora_ef | 0.2 | 3 | 0 | 0 | 0 | 0.7822 |  |  |
| qnli | fedex_dense | dense | 3 | 0 | 0 | 0 | 0.8066 |  |  |
| qnli | fedsb | 0.05 | 3 | 0 | 0 | 0 | 0.6730 |  |  |
| qnli | fedsb | 0.1 | 3 | 0 | 0 | 0 | 0.7058 |  |  |
| qnli | fedsb | 0.2 | 3 | 0 | 0 | 0 | 0.7101 |  |  |
| qnli | fedsvd_dense | dense | 3 | 0 | 0 | 0 | 0.7929 |  |  |
| qnli | fedsvd_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7855 |  |  |
| qnli | fedsvd_ef | 0.1 | 3 | 0 | 0 | 0 | 0.7975 |  |  |
| qnli | fedsvd_ef | 0.2 | 3 | 0 | 0 | 0 | 0.8194 |  |  |
| qnli | flora_dense | dense | 3 | 0 | 0 | 0 | 0.7611 |  |  |
| qnli | full_lora_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7681 |  |  |
| qnli | full_lora_ef | 0.1 | 3 | 0 | 0 | 0 | 0.8007 |  |  |
| qnli | full_lora_ef | 0.2 | 3 | 0 | 0 | 0 | 0.8254 |  |  |
| qnli | rolora_dense | dense | 3 | 0 | 0 | 0 | 0.8218 |  |  |
| qnli | rolora_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7797 |  |  |
| qnli | rolora_ef | 0.1 | 3 | 0 | 0 | 0 | 0.7852 |  |  |
| qnli | rolora_ef | 0.2 | 3 | 0 | 0 | 0 | 0.7989 |  |  |
| sst2 | fedex_dense | dense | 3 | 0 | 0 | 0 | 0.8857 |  |  |
| sst2 | fedsb | 0.05 | 3 | 0 | 0 | 0 | 0.7114 |  |  |
| sst2 | fedsb | 0.1 | 3 | 0 | 0 | 0 | 0.7611 |  |  |
| sst2 | fedsb | 0.2 | 3 | 0 | 0 | 0 | 0.8085 |  |  |
| sst2 | fedsvd_dense | dense | 3 | 0 | 0 | 0 | 0.8777 |  |  |
| sst2 | fedsvd_ef | 0.05 | 3 | 0 | 0 | 0 | 0.8727 |  |  |
| sst2 | fedsvd_ef | 0.1 | 3 | 0 | 0 | 0 | 0.8823 |  |  |
| sst2 | fedsvd_ef | 0.2 | 3 | 0 | 0 | 0 | 0.8773 |  |  |
| sst2 | flora_dense | dense | 3 | 0 | 0 | 0 | 0.8876 |  |  |
| sst2 | full_lora_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7653 |  |  |
| sst2 | full_lora_ef | 0.1 | 3 | 0 | 0 | 0 | 0.8402 |  |  |
| sst2 | full_lora_ef | 0.2 | 3 | 0 | 0 | 0 | 0.8956 |  |  |
| sst2 | rolora_dense | dense | 3 | 0 | 0 | 0 | 0.8777 |  |  |
| sst2 | rolora_ef | 0.05 | 3 | 0 | 0 | 0 | 0.8547 |  |  |
| sst2 | rolora_ef | 0.1 | 3 | 0 | 0 | 0 | 0.8135 |  |  |
| sst2 | rolora_ef | 0.2 | 3 | 0 | 0 | 0 | 0.8704 |  |  |

## Paired differences (left − right, final primary score)

| Comparison | Task | Wire | n | Δ final mean ± SD | Win / Tie / Loss | Strict (left / right) | Post-hit violations (left / right) |
|---|---|---|---:|---:|---:|---:|---:|

Delivered bytes differ between matched EF and DCEF runs only by the DCEF control packet and the at most three sparse entries it displaces; see `delta_total_bytes` in DCEF_PAIRS.csv.
Confirmation-stage evidence on official validation scores with seeds fixed before any run of this stage; schedules, learning rates and the basis protocol were frozen in the development stage.
