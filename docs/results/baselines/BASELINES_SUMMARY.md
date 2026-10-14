# gev0 vs published federated LoRA baselines

Official validation, seeds [1001, 1002, 1003].  Cells: mean ± SD of the final score (seeds reaching C95, median C95 bytes including one-time setup).  C95 = 95% of the same seed's dense FedIT final score.

## MNLI

### Uncompressed

| Method | final (C95) | MB / round |
|---|---|---:|
| FedIT (FedAvg-LoRA) | 0.810 ± 0.004 (3/3, 297.3 MB) | 11.89 |
| FFA-LoRA (orthonormal A) | 0.777 ± 0.011 (2/3, 239.7 MB) | 5.99 |
| FLoRA | 0.781 ± 0.005 (3/3, 2274.1 MB) | 64.97 |
| FedEx-LoRA | 0.810 ± 0.002 (3/3, 974.6 MB) | 64.97 |
| RoLoRA | 0.796 ± 0.003 (3/3, 239.7 MB) | 5.99 |
| FedSVD | 0.790 ± 0.011 (3/3, 179.8 MB) | 5.99 |
| gev0 (ours) | 0.811 ± 0.007 (3/3, 97.4 MB) | 5.99 |

### Communication budget (fraction of a dense full-LoRA message)

| Method | 5% | 10% | 20% |
|---|---|---|---|
| FFA-LoRA + Top-k/EF | 0.749 ± 0.017 (0/3) | 0.766 ± 0.015 (1/3, 47.6 MB) | 0.777 ± 0.009 (2/3, 89.2 MB) |
| FedIT + Top-k/EF | 0.765 ± 0.011 (1/3, 29.8 MB) | 0.773 ± 0.044 (2/3, 47.6 MB) | 0.805 ± 0.004 (3/3, 71.4 MB) |
| RoLoRA + Top-k/EF | 0.741 ± 0.015 (0/3) | 0.765 ± 0.022 (2/3, 53.5 MB) | 0.782 ± 0.015 (2/3, 95.2 MB) |
| FedSVD + Top-k/EF | 0.767 ± 0.017 (1/3, 23.8 MB) | 0.779 ± 0.016 (3/3, 53.5 MB) | 0.782 ± 0.010 (3/3, 95.1 MB) |
| Fed-SB (rank matched to budget) | 0.529 ± 0.063 (0/3) | 0.598 ± 0.082 (0/3) | 0.673 ± 0.096 (0/3) |
| gev0 + Top-k/EF (ours) | 0.793 ± 0.007 (3/3, 28.4 MB) | 0.806 ± 0.007 (3/3, 37.3 MB) | 0.809 ± 0.004 (3/3, 55.1 MB) |

Fed-SB rank per budget: 5% → r=22, 10% → r=33, 20% → r=48

## QNLI

### Uncompressed

| Method | final (C95) | MB / round |
|---|---|---:|
| FedIT (FedAvg-LoRA) | 0.827 ± 0.069 (3/3, 106.8 MB) | 11.86 |
| FFA-LoRA (orthonormal A) | 0.797 ± 0.070 (3/3, 71.5 MB) | 5.96 |
| FLoRA | 0.761 ± 0.052 (1/3, 1558.7 MB) | 64.94 |
| FedEx-LoRA | 0.807 ± 0.114 (3/3, 779.4 MB) | 64.94 |
| RoLoRA | 0.822 ± 0.047 (3/3, 71.6 MB) | 5.96 |
| FedSVD | 0.793 ± 0.059 (3/3, 71.5 MB) | 5.96 |
| gev0 (ours) | 0.839 ± 0.051 (3/3, 43.3 MB) | 5.96 |

### Communication budget (fraction of a dense full-LoRA message)

| Method | 5% | 10% | 20% |
|---|---|---|---|
| FFA-LoRA + Top-k/EF | 0.780 ± 0.054 (1/3, 14.2 MB) | 0.801 ± 0.050 (3/3, 24.9 MB) | 0.804 ± 0.046 (3/3, 42.7 MB) |
| FedIT + Top-k/EF | 0.768 ± 0.040 (2/3, 12.5 MB) | 0.801 ± 0.079 (3/3, 24.9 MB) | 0.825 ± 0.035 (3/3, 42.7 MB) |
| RoLoRA + Top-k/EF | 0.780 ± 0.072 (1/3, 16.0 MB) | 0.785 ± 0.073 (2/3, 28.5 MB) | 0.799 ± 0.068 (3/3, 42.7 MB) |
| FedSVD + Top-k/EF | 0.786 ± 0.074 (2/3, 13.4 MB) | 0.797 ± 0.062 (3/3, 28.5 MB) | 0.819 ± 0.034 (3/3, 35.6 MB) |
| Fed-SB (rank matched to budget) | 0.673 ± 0.126 (0/3) | 0.706 ± 0.033 (0/3) | 0.710 ± 0.116 (1/3, 93.0 MB) |
| gev0 + Top-k/EF (ours) | 0.835 ± 0.028 (3/3, 14.7 MB) | 0.842 ± 0.032 (3/3, 21.8 MB) | 0.849 ± 0.038 (3/3, 21.8 MB) |

Fed-SB rank per budget: 5% → r=23, 10% → r=34, 20% → r=49

## SST2

### Uncompressed

| Method | final (C95) | MB / round |
|---|---|---:|
| FedIT (FedAvg-LoRA) | 0.906 ± 0.022 (3/3, 83.1 MB) | 11.86 |
| FFA-LoRA (orthonormal A) | 0.860 ± 0.040 (2/3, 122.2 MB) | 5.96 |
| FLoRA | 0.888 ± 0.026 (3/3, 1234.0 MB) | 64.94 |
| FedEx-LoRA | 0.886 ± 0.030 (3/3, 519.6 MB) | 64.94 |
| RoLoRA | 0.878 ± 0.048 (3/3, 83.5 MB) | 5.96 |
| FedSVD | 0.878 ± 0.041 (3/3, 41.7 MB) | 5.96 |
| gev0 (ours) | 0.898 ± 0.026 (3/3, 55.3 MB) | 5.96 |

### Communication budget (fraction of a dense full-LoRA message)

| Method | 5% | 10% | 20% |
|---|---|---|---|
| FFA-LoRA + Top-k/EF | 0.708 ± 0.131 (0/3) | 0.821 ± 0.065 (0/3) | 0.849 ± 0.056 (1/3, 52.2 MB) |
| FedIT + Top-k/EF | 0.765 ± 0.119 (1/3, 17.2 MB) | 0.840 ± 0.050 (2/3, 26.7 MB) | 0.896 ± 0.014 (3/3, 49.8 MB) |
| RoLoRA + Top-k/EF | 0.855 ± 0.019 (2/3, 14.3 MB) | 0.813 ± 0.110 (2/3, 20.8 MB) | 0.870 ± 0.033 (3/3, 61.7 MB) |
| FedSVD + Top-k/EF | 0.873 ± 0.053 (2/3, 8.6 MB) | 0.882 ± 0.050 (2/3, 15.4 MB) | 0.877 ± 0.060 (3/3, 33.2 MB) |
| Fed-SB (rank matched to budget) | 0.711 ± 0.181 (0/3) | 0.761 ± 0.121 (1/3, 55.6 MB) | 0.808 ± 0.098 (1/3, 76.4 MB) |
| gev0 + Top-k/EF (ours) | 0.833 ± 0.042 (3/3, 20.6 MB) | 0.872 ± 0.030 (3/3, 36.0 MB) | 0.886 ± 0.032 (3/3, 40.8 MB) |

Fed-SB rank per budget: 5% → r=23, 10% → r=34, 20% → r=49

## Paired comparisons (same task, seed and budget)

Win / tie / loss with |Δ| ≤ 0.003 as a tie; p = exact two-sided sign test over non-tied pairs.

| gev0 − baseline | Task | n | mean Δ final | mean Δ last-3 | W / T / L | p |
|---|---|---:|---:|---:|---:|---:|
| gev0 (ours) − FedIT (FedAvg-LoRA) | mnli | 3 | +0.0014 | -0.0003 | 2 / 0 / 1 | 1 |
| gev0 (ours) − FedIT (FedAvg-LoRA) | qnli | 3 | +0.0126 | +0.0096 | 2 / 0 / 1 | 1 |
| gev0 (ours) − FedIT (FedAvg-LoRA) | sst2 | 3 | -0.0080 | -0.0124 | 0 / 1 / 2 | 0.5 |
| gev0 (ours) − FedIT (FedAvg-LoRA) | all | 9 | +0.0020 | -0.0010 | 4 / 1 / 4 | 1 |
| gev0 (ours) − FFA-LoRA (orthonormal A) | mnli | 3 | +0.0340 | +0.0332 | 3 / 0 / 0 | 0.25 |
| gev0 (ours) − FFA-LoRA (orthonormal A) | qnli | 3 | +0.0427 | +0.0399 | 3 / 0 / 0 | 0.25 |
| gev0 (ours) − FFA-LoRA (orthonormal A) | sst2 | 3 | +0.0375 | +0.0399 | 3 / 0 / 0 | 0.25 |
| gev0 (ours) − FFA-LoRA (orthonormal A) | all | 9 | +0.0380 | +0.0377 | 9 / 0 / 0 | 0.0039 |
| gev0 (ours) − FLoRA | mnli | 3 | +0.0305 | +0.0325 | 3 / 0 / 0 | 0.25 |
| gev0 (ours) − FLoRA | qnli | 3 | +0.0783 | +0.0997 | 3 / 0 / 0 | 0.25 |
| gev0 (ours) − FLoRA | sst2 | 3 | +0.0099 | +0.0353 | 1 / 0 / 2 | 1 |
| gev0 (ours) − FLoRA | all | 9 | +0.0396 | +0.0558 | 7 / 0 / 2 | 0.18 |
| gev0 (ours) − FedEx-LoRA | mnli | 3 | +0.0016 | +0.0007 | 2 / 0 / 1 | 1 |
| gev0 (ours) − FedEx-LoRA | qnli | 3 | +0.0327 | +0.0070 | 2 / 0 / 1 | 1 |
| gev0 (ours) − FedEx-LoRA | sst2 | 3 | +0.0119 | +0.0040 | 1 / 1 / 1 | 1 |
| gev0 (ours) − FedEx-LoRA | all | 9 | +0.0154 | +0.0039 | 5 / 1 / 3 | 0.73 |
| gev0 (ours) − RoLoRA | mnli | 3 | +0.0155 | +0.0236 | 3 / 0 / 0 | 0.25 |
| gev0 (ours) − RoLoRA | qnli | 3 | +0.0176 | +0.0367 | 2 / 1 / 0 | 0.5 |
| gev0 (ours) − RoLoRA | sst2 | 3 | +0.0199 | +0.0159 | 3 / 0 / 0 | 0.25 |
| gev0 (ours) − RoLoRA | all | 9 | +0.0176 | +0.0254 | 8 / 1 / 0 | 0.0078 |
| gev0 (ours) − FedSVD | mnli | 3 | +0.0216 | +0.0247 | 2 / 1 / 0 | 0.5 |
| gev0 (ours) − FedSVD | qnli | 3 | +0.0464 | +0.0367 | 3 / 0 / 0 | 0.25 |
| gev0 (ours) − FedSVD | sst2 | 3 | +0.0199 | +0.0130 | 2 / 0 / 1 | 1 |
| gev0 (ours) − FedSVD | all | 9 | +0.0293 | +0.0248 | 7 / 1 / 1 | 0.07 |
| gev0 + Top-k/EF (ours) − FFA-LoRA + Top-k/EF | mnli | 9 | +0.0386 | +0.0405 | 9 / 0 / 0 | 0.0039 |
| gev0 + Top-k/EF (ours) − FFA-LoRA + Top-k/EF | qnli | 9 | +0.0468 | +0.0435 | 9 / 0 / 0 | 0.0039 |
| gev0 + Top-k/EF (ours) − FFA-LoRA + Top-k/EF | sst2 | 9 | +0.0706 | +0.0804 | 9 / 0 / 0 | 0.0039 |
| gev0 + Top-k/EF (ours) − FFA-LoRA + Top-k/EF | all | 27 | +0.0520 | +0.0548 | 27 / 0 / 0 | 1.5e-08 |
| gev0 + Top-k/EF (ours) − FedIT + Top-k/EF | mnli | 9 | +0.0219 | +0.0277 | 7 / 2 / 0 | 0.016 |
| gev0 + Top-k/EF (ours) − FedIT + Top-k/EF | qnli | 9 | +0.0438 | +0.0429 | 7 / 2 / 0 | 0.016 |
| gev0 + Top-k/EF (ours) − FedIT + Top-k/EF | sst2 | 9 | +0.0298 | +0.0299 | 5 / 1 / 3 | 0.73 |
| gev0 + Top-k/EF (ours) − FedIT + Top-k/EF | all | 27 | +0.0318 | +0.0335 | 19 / 5 / 3 | 0.00086 |
| gev0 + Top-k/EF (ours) − RoLoRA + Top-k/EF | mnli | 9 | +0.0397 | +0.0483 | 9 / 0 / 0 | 0.0039 |
| gev0 + Top-k/EF (ours) − RoLoRA + Top-k/EF | qnli | 9 | +0.0540 | +0.0656 | 9 / 0 / 0 | 0.0039 |
| gev0 + Top-k/EF (ours) − RoLoRA + Top-k/EF | sst2 | 9 | +0.0173 | +0.0220 | 6 / 0 / 3 | 0.51 |
| gev0 + Top-k/EF (ours) − RoLoRA + Top-k/EF | all | 27 | +0.0370 | +0.0453 | 24 / 0 / 3 | 4.9e-05 |
| gev0 + Top-k/EF (ours) − FedSVD + Top-k/EF | mnli | 9 | +0.0266 | +0.0279 | 9 / 0 / 0 | 0.0039 |
| gev0 + Top-k/EF (ours) − FedSVD + Top-k/EF | qnli | 9 | +0.0411 | +0.0305 | 9 / 0 / 0 | 0.0039 |
| gev0 + Top-k/EF (ours) − FedSVD + Top-k/EF | sst2 | 9 | -0.0139 | -0.0140 | 2 / 1 / 6 | 0.29 |
| gev0 + Top-k/EF (ours) − FedSVD + Top-k/EF | all | 27 | +0.0179 | +0.0148 | 20 / 1 / 6 | 0.0094 |
| gev0 + Top-k/EF (ours) − Fed-SB (rank matched to budget) | mnli | 9 | +0.2027 | +0.2129 | 9 / 0 / 0 | 0.0039 |
| gev0 + Top-k/EF (ours) − Fed-SB (rank matched to budget) | qnli | 9 | +0.1456 | +0.1458 | 9 / 0 / 0 | 0.0039 |
| gev0 + Top-k/EF (ours) − Fed-SB (rank matched to budget) | sst2 | 9 | +0.1032 | +0.1108 | 7 / 0 / 2 | 0.18 |
| gev0 + Top-k/EF (ours) − Fed-SB (rank matched to budget) | all | 27 | +0.1505 | +0.1565 | 25 / 0 / 2 | 5.6e-06 |
