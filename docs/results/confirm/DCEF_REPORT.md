# DCEF / CAB campaign report

Config: `/data/home/202412012250/bishe_confirm/configs/roberta_confirm_main.yaml`; evaluation policy: `official_validation`; seeds: [1001, 1002, 1003].

Runs: 162/162 technically valid; 0 missing; 0 invalid.

## Stability counts

| Task | Method | Wire | n | C95 reached | Final ≥ target | Strict stable | Mean final | Mean Speedup95 | Mean γ̄·d/K |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mnli | cab0_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.4724 |  |  |
| mnli | cab0_a_ef | 0.1 | 3 | 0 | 0 | 0 | 0.6125 |  |  |
| mnli | cab0_a_ef | 0.2 | 3 | 0 | 0 | 0 | 0.4525 |  |  |
| mnli | cab0wh_a_ef | 0.05 | 3 | 3 | 3 | 3 | 0.7903 | 8.5748 |  |
| mnli | cab0wh_a_ef | 0.1 | 3 | 3 | 3 | 2 | 0.7968 | 6.1388 |  |
| mnli | cab0wh_a_ef | 0.2 | 3 | 3 | 3 | 3 | 0.8011 | 4.3948 |  |
| mnli | fedavg_lora | dense | 3 | 3 | 3 | 1 | 0.8099 | 1.0000 |  |
| mnli | gev0_a_dense | dense | 3 | 3 | 3 | 2 | 0.8114 | 2.8473 |  |
| mnli | gev0_a_ef | 0.05 | 3 | 3 | 3 | 3 | 0.7934 | 10.1832 |  |
| mnli | gev0_a_ef | 0.1 | 3 | 3 | 3 | 3 | 0.8057 | 7.4381 |  |
| mnli | gev0_a_ef | 0.2 | 3 | 3 | 3 | 3 | 0.8090 | 5.5265 |  |
| mnli | random_a_dense | dense | 3 | 2 | 2 | 1 | 0.7774 | 1.1183 |  |
| mnli | random_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7489 |  |  |
| mnli | random_a_ef | 0.1 | 3 | 1 | 1 | 1 | 0.7659 | 4.9986 |  |
| mnli | random_a_ef | 0.2 | 3 | 2 | 2 | 2 | 0.7773 | 2.9909 |  |
| mnli | spectral_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7510 |  |  |
| mnli | spectral_a_ef | 0.1 | 3 | 1 | 1 | 1 | 0.7698 | 4.9986 |  |
| mnli | spectral_a_ef | 0.2 | 3 | 3 | 2 | 2 | 0.7782 | 2.8009 |  |
| qnli | cab0_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.6503 |  |  |
| qnli | cab0_a_ef | 0.1 | 3 | 1 | 1 | 1 | 0.7700 | 2.5732 |  |
| qnli | cab0_a_ef | 0.2 | 3 | 2 | 0 | 0 | 0.7748 | 1.2720 |  |
| qnli | cab0wh_a_ef | 0.05 | 3 | 3 | 3 | 2 | 0.8290 | 7.1402 |  |
| qnli | cab0wh_a_ef | 0.1 | 3 | 3 | 3 | 2 | 0.8431 | 5.4853 |  |
| qnli | cab0wh_a_ef | 0.2 | 3 | 3 | 3 | 2 | 0.8515 | 4.3574 |  |
| qnli | fedavg_lora | dense | 3 | 3 | 3 | 0 | 0.8268 | 1.0000 |  |
| qnli | gev0_a_dense | dense | 3 | 3 | 3 | 1 | 0.8393 | 2.1437 |  |
| qnli | gev0_a_ef | 0.05 | 3 | 3 | 3 | 1 | 0.8347 | 7.1402 |  |
| qnli | gev0_a_ef | 0.1 | 3 | 3 | 3 | 3 | 0.8423 | 4.9462 |  |
| qnli | gev0_a_ef | 0.2 | 3 | 3 | 3 | 2 | 0.8487 | 4.4660 |  |
| qnli | random_a_dense | dense | 3 | 3 | 2 | 0 | 0.7967 | 1.3818 |  |
| qnli | random_a_ef | 0.05 | 3 | 1 | 1 | 0 | 0.7803 | 7.4926 |  |
| qnli | random_a_ef | 0.1 | 3 | 3 | 2 | 1 | 0.8007 | 4.2837 |  |
| qnli | random_a_ef | 0.2 | 3 | 3 | 3 | 1 | 0.8044 | 2.5460 |  |
| qnli | spectral_a_ef | 0.05 | 3 | 2 | 2 | 2 | 0.8007 | 5.8286 |  |
| qnli | spectral_a_ef | 0.1 | 3 | 3 | 3 | 2 | 0.8121 | 4.4425 |  |
| qnli | spectral_a_ef | 0.2 | 3 | 3 | 3 | 2 | 0.8255 | 2.7774 |  |
| sst2 | cab0_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.8108 |  |  |
| sst2 | cab0_a_ef | 0.1 | 3 | 2 | 2 | 2 | 0.8559 | 3.0010 |  |
| sst2 | cab0_a_ef | 0.2 | 3 | 2 | 2 | 1 | 0.8758 | 2.4440 |  |
| sst2 | cab0wh_a_ef | 0.05 | 3 | 2 | 1 | 0 | 0.8544 | 6.5162 |  |
| sst2 | cab0wh_a_ef | 0.1 | 3 | 3 | 3 | 2 | 0.8811 | 3.1670 |  |
| sst2 | cab0wh_a_ef | 0.2 | 3 | 3 | 3 | 2 | 0.8895 | 2.2149 |  |
| sst2 | fedavg_lora | dense | 3 | 3 | 3 | 0 | 0.9056 | 1.0000 |  |
| sst2 | gev0_a_dense | dense | 3 | 3 | 3 | 0 | 0.8976 | 1.5887 |  |
| sst2 | gev0_a_ef | 0.05 | 3 | 3 | 1 | 1 | 0.8326 | 5.7454 |  |
| sst2 | gev0_a_ef | 0.1 | 3 | 3 | 3 | 3 | 0.8723 | 2.9827 |  |
| sst2 | gev0_a_ef | 0.2 | 3 | 3 | 3 | 1 | 0.8857 | 3.2346 |  |
| sst2 | random_a_dense | dense | 3 | 2 | 1 | 1 | 0.8601 | 0.9055 |  |
| sst2 | random_a_ef | 0.05 | 3 | 0 | 0 | 0 | 0.7083 |  |  |
| sst2 | random_a_ef | 0.1 | 3 | 0 | 0 | 0 | 0.8211 |  |  |
| sst2 | random_a_ef | 0.2 | 3 | 1 | 1 | 1 | 0.8494 | 1.3639 |  |
| sst2 | spectral_a_ef | 0.05 | 3 | 1 | 1 | 1 | 0.6277 | 4.9967 |  |
| sst2 | spectral_a_ef | 0.1 | 3 | 1 | 1 | 1 | 0.6758 | 3.1810 |  |
| sst2 | spectral_a_ef | 0.2 | 3 | 1 | 1 | 1 | 0.7859 | 1.8422 |  |

## Paired differences (left − right, final primary score)

| Comparison | Task | Wire | n | Δ final mean ± SD | Win / Tie / Loss | Strict (left / right) | Post-hit violations (left / right) |
|---|---|---|---:|---:|---:|---:|---:|
| cab0_a_ef-random_a_ef | mnli | 0.05 | 3 | -0.2765 ± 0.1059 | 0 / 0 / 3 | 0 / 0 | 0 / 0 |
| cab0_a_ef-random_a_ef | mnli | 0.1 | 3 | -0.1534 ± 0.0305 | 0 / 0 / 3 | 0 / 1 | 0 / 0 |
| cab0_a_ef-random_a_ef | mnli | 0.2 | 3 | -0.3248 ± 0.1624 | 0 / 0 / 3 | 0 / 2 | 0 / 0 |
| cab0_a_ef-random_a_ef | qnli | 0.05 | 3 | -0.1300 ± 0.0823 | 0 / 0 / 3 | 0 / 0 | 0 / 0 |
| cab0_a_ef-random_a_ef | qnli | 0.1 | 3 | -0.0307 ± 0.0328 | 1 / 0 / 2 | 1 / 1 | 0 / 5 |
| cab0_a_ef-random_a_ef | qnli | 0.2 | 3 | -0.0297 ± 0.0300 | 0 / 0 / 3 | 0 / 1 | 2 / 3 |
| cab0_a_ef-random_a_ef | sst2 | 0.05 | 3 | +0.1024 ± 0.1348 | 3 / 0 / 0 | 0 / 0 | 0 / 0 |
| cab0_a_ef-random_a_ef | sst2 | 0.1 | 3 | +0.0348 ± 0.0193 | 3 / 0 / 0 | 2 / 0 | 0 / 0 |
| cab0_a_ef-random_a_ef | sst2 | 0.2 | 3 | +0.0264 ± 0.0159 | 3 / 0 / 0 | 1 / 1 | 3 / 0 |
| cab0wh_a_ef-cab0_a_ef | mnli | 0.05 | 3 | +0.3179 ± 0.1094 | 3 / 0 / 0 | 3 / 0 | 0 / 0 |
| cab0wh_a_ef-cab0_a_ef | mnli | 0.1 | 3 | +0.1843 ± 0.0306 | 3 / 0 / 0 | 2 / 0 | 1 / 0 |
| cab0wh_a_ef-cab0_a_ef | mnli | 0.2 | 3 | +0.3486 ± 0.1687 | 3 / 0 / 0 | 3 / 0 | 0 / 0 |
| cab0wh_a_ef-cab0_a_ef | qnli | 0.05 | 3 | +0.1788 ± 0.0674 | 3 / 0 / 0 | 2 / 0 | 3 / 0 |
| cab0wh_a_ef-cab0_a_ef | qnli | 0.1 | 3 | +0.0732 ± 0.0228 | 3 / 0 / 0 | 2 / 1 | 0 / 0 |
| cab0wh_a_ef-cab0_a_ef | qnli | 0.2 | 3 | +0.0768 ± 0.0495 | 3 / 0 / 0 | 2 / 0 | 0 / 2 |
| cab0wh_a_ef-cab0_a_ef | sst2 | 0.05 | 3 | +0.0436 ± 0.0626 | 3 / 0 / 0 | 0 / 0 | 13 / 0 |
| cab0wh_a_ef-cab0_a_ef | sst2 | 0.1 | 3 | +0.0252 ± 0.0289 | 3 / 0 / 0 | 2 / 2 | 7 / 0 |
| cab0wh_a_ef-cab0_a_ef | sst2 | 0.2 | 3 | +0.0138 ± 0.0325 | 1 / 1 / 1 | 2 / 1 | 5 / 3 |
| cab0wh_a_ef-gev0_a_ef | mnli | 0.05 | 3 | -0.0030 ± 0.0096 | 1 / 0 / 2 | 3 / 3 | 0 / 0 |
| cab0wh_a_ef-gev0_a_ef | mnli | 0.1 | 3 | -0.0089 ± 0.0062 | 0 / 1 / 2 | 2 / 3 | 1 / 0 |
| cab0wh_a_ef-gev0_a_ef | mnli | 0.2 | 3 | -0.0079 ± 0.0016 | 0 / 0 / 3 | 3 / 3 | 0 / 0 |
| cab0wh_a_ef-gev0_a_ef | qnli | 0.05 | 3 | -0.0057 ± 0.0041 | 0 / 1 / 2 | 2 / 1 | 3 / 4 |
| cab0wh_a_ef-gev0_a_ef | qnli | 0.1 | 3 | +0.0009 ± 0.0080 | 1 / 1 / 1 | 2 / 3 | 0 / 0 |
| cab0wh_a_ef-gev0_a_ef | qnli | 0.2 | 3 | +0.0029 ± 0.0096 | 1 / 1 / 1 | 2 / 2 | 0 / 0 |
| cab0wh_a_ef-gev0_a_ef | sst2 | 0.05 | 3 | +0.0218 ± 0.0399 | 2 / 0 / 1 | 0 / 1 | 13 / 26 |
| cab0wh_a_ef-gev0_a_ef | sst2 | 0.1 | 3 | +0.0088 ± 0.0086 | 2 / 1 / 0 | 2 / 3 | 7 / 0 |
| cab0wh_a_ef-gev0_a_ef | sst2 | 0.2 | 3 | +0.0038 ± 0.0083 | 2 / 0 / 1 | 2 / 1 | 5 / 18 |
| cab0wh_a_ef-random_a_ef | mnli | 0.05 | 3 | +0.0414 ± 0.0091 | 3 / 0 / 0 | 3 / 0 | 0 / 0 |
| cab0wh_a_ef-random_a_ef | mnli | 0.1 | 3 | +0.0310 ± 0.0139 | 3 / 0 / 0 | 2 / 1 | 1 / 0 |
| cab0wh_a_ef-random_a_ef | mnli | 0.2 | 3 | +0.0238 ± 0.0094 | 3 / 0 / 0 | 3 / 2 | 0 / 0 |
| cab0wh_a_ef-random_a_ef | qnli | 0.05 | 3 | +0.0488 ± 0.0261 | 3 / 0 / 0 | 2 / 0 | 3 / 0 |
| cab0wh_a_ef-random_a_ef | qnli | 0.1 | 3 | +0.0425 ± 0.0109 | 3 / 0 / 0 | 2 / 1 | 0 / 5 |
| cab0wh_a_ef-random_a_ef | qnli | 0.2 | 3 | +0.0471 ± 0.0199 | 3 / 0 / 0 | 2 / 1 | 0 / 3 |
| cab0wh_a_ef-random_a_ef | sst2 | 0.05 | 3 | +0.1460 ± 0.1277 | 3 / 0 / 0 | 0 / 0 | 13 / 0 |
| cab0wh_a_ef-random_a_ef | sst2 | 0.1 | 3 | +0.0600 ± 0.0444 | 3 / 0 / 0 | 2 / 0 | 7 / 0 |
| cab0wh_a_ef-random_a_ef | sst2 | 0.2 | 3 | +0.0401 ± 0.0259 | 3 / 0 / 0 | 2 / 1 | 5 / 0 |
| gev0_a_dense-fedavg_lora | mnli | dense | 3 | +0.0014 ± 0.0088 | 2 / 0 / 1 | 2 / 1 | 0 / 0 |
| gev0_a_dense-fedavg_lora | qnli | dense | 3 | +0.0126 ± 0.0259 | 2 / 0 / 1 | 1 / 0 | 2 / 1 |
| gev0_a_dense-fedavg_lora | sst2 | dense | 3 | -0.0080 ± 0.0053 | 0 / 1 / 2 | 0 / 0 | 5 / 7 |
| gev0_a_dense-random_a_dense | mnli | dense | 3 | +0.0340 ± 0.0164 | 3 / 0 / 0 | 2 / 1 | 0 / 0 |
| gev0_a_dense-random_a_dense | qnli | dense | 3 | +0.0427 ± 0.0194 | 3 / 0 / 0 | 1 / 0 | 2 / 4 |
| gev0_a_dense-random_a_dense | sst2 | dense | 3 | +0.0375 ± 0.0221 | 3 / 0 / 0 | 0 / 1 | 5 / 3 |
| gev0_a_ef-cab0_a_ef | mnli | 0.05 | 3 | +0.3209 ± 0.1155 | 3 / 0 / 0 | 3 / 0 | 0 / 0 |
| gev0_a_ef-cab0_a_ef | mnli | 0.1 | 3 | +0.1932 ± 0.0275 | 3 / 0 / 0 | 3 / 0 | 0 / 0 |
| gev0_a_ef-cab0_a_ef | mnli | 0.2 | 3 | +0.3564 ± 0.1702 | 3 / 0 / 0 | 3 / 0 | 0 / 0 |
| gev0_a_ef-cab0_a_ef | qnli | 0.05 | 3 | +0.1845 ± 0.0642 | 3 / 0 / 0 | 1 / 0 | 4 / 0 |
| gev0_a_ef-cab0_a_ef | qnli | 0.1 | 3 | +0.0723 ± 0.0185 | 3 / 0 / 0 | 3 / 1 | 0 / 0 |
| gev0_a_ef-cab0_a_ef | qnli | 0.2 | 3 | +0.0739 ± 0.0399 | 3 / 0 / 0 | 2 / 0 | 0 / 2 |
| gev0_a_ef-cab0_a_ef | sst2 | 0.05 | 3 | +0.0218 ± 0.0374 | 2 / 0 / 1 | 1 / 0 | 26 / 0 |
| gev0_a_ef-cab0_a_ef | sst2 | 0.1 | 3 | +0.0164 ± 0.0216 | 2 / 1 / 0 | 3 / 2 | 0 / 0 |
| gev0_a_ef-cab0_a_ef | sst2 | 0.2 | 3 | +0.0099 ± 0.0281 | 1 / 0 / 2 | 1 / 1 | 18 / 3 |
| gev0_a_ef-cab0wh_a_ef | mnli | 0.05 | 3 | +0.0030 ± 0.0096 | 2 / 0 / 1 | 3 / 3 | 0 / 0 |
| gev0_a_ef-cab0wh_a_ef | mnli | 0.1 | 3 | +0.0089 ± 0.0062 | 2 / 1 / 0 | 3 / 2 | 0 / 1 |
| gev0_a_ef-cab0wh_a_ef | mnli | 0.2 | 3 | +0.0079 ± 0.0016 | 3 / 0 / 0 | 3 / 3 | 0 / 0 |
| gev0_a_ef-cab0wh_a_ef | qnli | 0.05 | 3 | +0.0057 ± 0.0041 | 2 / 1 / 0 | 1 / 2 | 4 / 3 |
| gev0_a_ef-cab0wh_a_ef | qnli | 0.1 | 3 | -0.0009 ± 0.0080 | 1 / 1 / 1 | 3 / 2 | 0 / 0 |
| gev0_a_ef-cab0wh_a_ef | qnli | 0.2 | 3 | -0.0029 ± 0.0096 | 1 / 1 / 1 | 2 / 2 | 0 / 0 |
| gev0_a_ef-cab0wh_a_ef | sst2 | 0.05 | 3 | -0.0218 ± 0.0399 | 1 / 0 / 2 | 1 / 0 | 26 / 13 |
| gev0_a_ef-cab0wh_a_ef | sst2 | 0.1 | 3 | -0.0088 ± 0.0086 | 0 / 1 / 2 | 3 / 2 | 0 / 7 |
| gev0_a_ef-cab0wh_a_ef | sst2 | 0.2 | 3 | -0.0038 ± 0.0083 | 1 / 0 / 2 | 1 / 2 | 18 / 5 |
| gev0_a_ef-random_a_ef | mnli | 0.05 | 3 | +0.0444 ± 0.0186 | 3 / 0 / 0 | 3 / 0 | 0 / 0 |
| gev0_a_ef-random_a_ef | mnli | 0.1 | 3 | +0.0398 ± 0.0185 | 3 / 0 / 0 | 3 / 1 | 0 / 0 |
| gev0_a_ef-random_a_ef | mnli | 0.2 | 3 | +0.0317 ± 0.0106 | 3 / 0 / 0 | 3 / 2 | 0 / 0 |
| gev0_a_ef-random_a_ef | qnli | 0.05 | 3 | +0.0544 ± 0.0256 | 3 / 0 / 0 | 1 / 0 | 4 / 0 |
| gev0_a_ef-random_a_ef | qnli | 0.1 | 3 | +0.0416 ± 0.0186 | 3 / 0 / 0 | 3 / 1 | 0 / 5 |
| gev0_a_ef-random_a_ef | qnli | 0.2 | 3 | +0.0442 ± 0.0107 | 3 / 0 / 0 | 2 / 1 | 0 / 3 |
| gev0_a_ef-random_a_ef | sst2 | 0.05 | 3 | +0.1242 ± 0.1054 | 3 / 0 / 0 | 1 / 0 | 26 / 0 |
| gev0_a_ef-random_a_ef | sst2 | 0.1 | 3 | +0.0512 ± 0.0359 | 3 / 0 / 0 | 3 / 0 | 0 / 0 |
| gev0_a_ef-random_a_ef | sst2 | 0.2 | 3 | +0.0363 ± 0.0259 | 3 / 0 / 0 | 1 / 1 | 18 / 0 |
| gev0_a_ef-spectral_a_ef | mnli | 0.05 | 3 | +0.0423 ± 0.0132 | 3 / 0 / 0 | 3 / 0 | 0 / 0 |
| gev0_a_ef-spectral_a_ef | mnli | 0.1 | 3 | +0.0359 ± 0.0147 | 3 / 0 / 0 | 3 / 1 | 0 / 0 |
| gev0_a_ef-spectral_a_ef | mnli | 0.2 | 3 | +0.0308 ± 0.0098 | 3 / 0 / 0 | 3 / 2 | 0 / 1 |
| gev0_a_ef-spectral_a_ef | qnli | 0.05 | 3 | +0.0340 ± 0.0405 | 2 / 1 / 0 | 1 / 2 | 4 / 0 |
| gev0_a_ef-spectral_a_ef | qnli | 0.1 | 3 | +0.0301 ± 0.0132 | 3 / 0 / 0 | 3 / 2 | 0 / 0 |
| gev0_a_ef-spectral_a_ef | qnli | 0.2 | 3 | +0.0232 ± 0.0154 | 3 / 0 / 0 | 2 / 2 | 0 / 0 |
| gev0_a_ef-spectral_a_ef | sst2 | 0.05 | 3 | +0.2049 ± 0.2395 | 2 / 0 / 1 | 1 / 1 | 26 / 0 |
| gev0_a_ef-spectral_a_ef | sst2 | 0.1 | 3 | +0.1965 ± 0.2116 | 2 / 0 / 1 | 3 / 1 | 0 / 0 |
| gev0_a_ef-spectral_a_ef | sst2 | 0.2 | 3 | +0.0998 ± 0.1636 | 2 / 0 / 1 | 1 / 1 | 18 / 0 |
| random_a_dense-fedavg_lora | mnli | dense | 3 | -0.0326 ± 0.0136 | 0 / 0 / 3 | 1 / 1 | 0 / 0 |
| random_a_dense-fedavg_lora | qnli | dense | 3 | -0.0301 ± 0.0193 | 0 / 0 / 3 | 0 / 0 | 4 / 1 |
| random_a_dense-fedavg_lora | sst2 | dense | 3 | -0.0455 ± 0.0205 | 0 / 0 / 3 | 1 / 0 | 3 / 7 |
| spectral_a_ef-random_a_ef | mnli | 0.05 | 3 | +0.0021 ± 0.0056 | 1 / 2 / 0 | 0 / 0 | 0 / 0 |
| spectral_a_ef-random_a_ef | mnli | 0.1 | 3 | +0.0040 ± 0.0043 | 1 / 2 / 0 | 1 / 1 | 0 / 0 |
| spectral_a_ef-random_a_ef | mnli | 0.2 | 3 | +0.0009 ± 0.0017 | 0 / 3 / 0 | 2 / 2 | 1 / 0 |
| spectral_a_ef-random_a_ef | qnli | 0.05 | 3 | +0.0204 ± 0.0508 | 2 / 0 / 1 | 2 / 0 | 0 / 0 |
| spectral_a_ef-random_a_ef | qnli | 0.1 | 3 | +0.0115 ± 0.0107 | 2 / 1 / 0 | 2 / 1 | 0 / 5 |
| spectral_a_ef-random_a_ef | qnli | 0.2 | 3 | +0.0211 ± 0.0069 | 3 / 0 / 0 | 2 / 1 | 0 / 3 |
| spectral_a_ef-random_a_ef | sst2 | 0.05 | 3 | -0.0807 ± 0.2529 | 1 / 0 / 2 | 1 / 0 | 0 / 0 |
| spectral_a_ef-random_a_ef | sst2 | 0.1 | 3 | -0.1453 ± 0.2402 | 1 / 0 / 2 | 1 / 0 | 0 / 0 |
| spectral_a_ef-random_a_ef | sst2 | 0.2 | 3 | -0.0635 ± 0.1695 | 1 / 0 / 2 | 1 / 1 | 0 / 0 |

## CAB basis construction (zero-training diagnostics)

Objective = predicted stationary excess error J(P)/tr(S) under exact statistics and the run's kappa (lower is better); overlaps are ||A1 A2^T||_F^2 / r.

| Task | Method | Wire | n | κ | J selected / J exact | overlap ↔ spectral | J selected | J spectral | J random | setup / total bytes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mnli | cab0_a_ef | 0.05 | 3 | 0.0000 | 1.4210 | 0.0358 | 0.0182 | 0.9872 | 0.9895 | 0.165 |
| mnli | cab0_a_ef | 0.1 | 3 | 0.0000 | 1.4210 | 0.0358 | 0.0182 | 0.9872 | 0.9895 | 0.090 |
| mnli | cab0_a_ef | 0.2 | 3 | 0.0000 | 1.4210 | 0.0358 | 0.0182 | 0.9872 | 0.9895 | 0.047 |
| mnli | cab0wh_a_ef | 0.05 | 3 | 0.0000 | 1.4210 | 0.0358 | 0.0182 | 0.9872 | 0.9895 | 0.203 |
| mnli | cab0wh_a_ef | 0.1 | 3 | 0.0000 | 1.4210 | 0.0358 | 0.0182 | 0.9872 | 0.9895 | 0.113 |
| mnli | cab0wh_a_ef | 0.2 | 3 | 0.0000 | 1.4210 | 0.0358 | 0.0182 | 0.9872 | 0.9895 | 0.060 |
| mnli | gev0_a_dense | dense | 3 | 0.0000 | 0.9940 | 0.0256 | 0.9667 | 0.9872 | 0.9895 | 0.025 |
| mnli | gev0_a_ef | 0.05 | 3 | 0.0000 | 0.9940 | 0.0256 | 0.9667 | 0.9872 | 0.9895 | 0.203 |
| mnli | gev0_a_ef | 0.1 | 3 | 0.0000 | 0.9940 | 0.0256 | 0.9667 | 0.9872 | 0.9895 | 0.113 |
| mnli | gev0_a_ef | 0.2 | 3 | 0.0000 | 0.9940 | 0.0256 | 0.9667 | 0.9872 | 0.9895 | 0.060 |
| qnli | cab0_a_ef | 0.05 | 3 | 0.0000 | 1.4858 | 0.0364 | 0.0115 | 0.9875 | 0.9893 | 0.249 |
| qnli | cab0_a_ef | 0.1 | 3 | 0.0000 | 1.4858 | 0.0364 | 0.0115 | 0.9875 | 0.9893 | 0.142 |
| qnli | cab0_a_ef | 0.2 | 3 | 0.0000 | 1.4858 | 0.0364 | 0.0115 | 0.9875 | 0.9893 | 0.077 |
| qnli | cab0wh_a_ef | 0.05 | 3 | 0.0000 | 1.4858 | 0.0364 | 0.0115 | 0.9875 | 0.9893 | 0.298 |
| qnli | cab0wh_a_ef | 0.1 | 3 | 0.0000 | 1.4858 | 0.0364 | 0.0115 | 0.9875 | 0.9893 | 0.175 |
| qnli | cab0wh_a_ef | 0.2 | 3 | 0.0000 | 1.4858 | 0.0364 | 0.0115 | 0.9875 | 0.9893 | 0.096 |
| qnli | gev0_a_dense | dense | 3 | 0.0000 | 0.9925 | 0.0283 | 0.9638 | 0.9875 | 0.9893 | 0.041 |
| qnli | gev0_a_ef | 0.05 | 3 | 0.0000 | 0.9925 | 0.0283 | 0.9638 | 0.9875 | 0.9893 | 0.298 |
| qnli | gev0_a_ef | 0.1 | 3 | 0.0000 | 0.9925 | 0.0283 | 0.9638 | 0.9875 | 0.9893 | 0.175 |
| qnli | gev0_a_ef | 0.2 | 3 | 0.0000 | 0.9925 | 0.0283 | 0.9638 | 0.9875 | 0.9893 | 0.096 |
| sst2 | cab0_a_ef | 0.05 | 3 | 0.0000 | 1.6386 | 0.0225 | 0.0105 | 0.9893 | 0.9895 | 0.249 |
| sst2 | cab0_a_ef | 0.1 | 3 | 0.0000 | 1.6386 | 0.0225 | 0.0105 | 0.9893 | 0.9895 | 0.142 |
| sst2 | cab0_a_ef | 0.2 | 3 | 0.0000 | 1.6386 | 0.0225 | 0.0105 | 0.9893 | 0.9895 | 0.077 |
| sst2 | cab0wh_a_ef | 0.05 | 3 | 0.0000 | 1.6386 | 0.0225 | 0.0105 | 0.9893 | 0.9895 | 0.298 |
| sst2 | cab0wh_a_ef | 0.1 | 3 | 0.0000 | 1.6386 | 0.0225 | 0.0105 | 0.9893 | 0.9895 | 0.175 |
| sst2 | cab0wh_a_ef | 0.2 | 3 | 0.0000 | 1.6386 | 0.0225 | 0.0105 | 0.9893 | 0.9895 | 0.096 |
| sst2 | gev0_a_dense | dense | 3 | 0.0000 | 0.9971 | 0.0229 | 0.9733 | 0.9893 | 0.9895 | 0.041 |
| sst2 | gev0_a_ef | 0.05 | 3 | 0.0000 | 0.9971 | 0.0229 | 0.9733 | 0.9893 | 0.9895 | 0.298 |
| sst2 | gev0_a_ef | 0.1 | 3 | 0.0000 | 0.9971 | 0.0229 | 0.9733 | 0.9893 | 0.9895 | 0.175 |
| sst2 | gev0_a_ef | 0.2 | 3 | 0.0000 | 0.9971 | 0.0229 | 0.9733 | 0.9893 | 0.9895 | 0.096 |

C95 for CAB includes its one-time setup bytes (`c95_bytes`); `c95_training_bytes` in DCEF_RUNS.csv excludes them.

Delivered bytes differ between matched EF and DCEF runs only by the DCEF control packet and the at most three sparse entries it displaces; see `delta_total_bytes` in DCEF_PAIRS.csv.
Confirmation-stage evidence on official validation scores with seeds fixed before any run of this stage; schedules, learning rates and the basis protocol were frozen in the development stage.
