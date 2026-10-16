# Curvature normalization as a fix for data-driven frozen bases

Official validation, seeds [1001, 1002, 1003]; bidirectional Top-k + EF.  Cells: mean ± SD of the final score (seeds reaching C95).

Runs from /data/home/202412012250/bishe_norm/docs/results/normfix_retuned_mnli, /data/home/202412012250/bishe_norm/docs/results/normfix_retuned_qnli replace 18 runs of the other roots.

## MNLI

| Basis | 5% | 10% | 20% |
|---|---|---|---|
| Random (FFA-LoRA) | 0.749 ± 0.017 (0/3) | 0.766 ± 0.015 (1/3) | 0.777 ± 0.009 (2/3) |
| Grad-energy | 0.472 ± 0.119 (0/3) | 0.613 ± 0.022 (0/3) | 0.453 ± 0.171 (0/3) |
| Grad-energy + normalization | 0.790 ± 0.010 (3/3) | 0.797 ± 0.009 (3/3) | 0.801 ± 0.004 (3/3) |
| EVA | 0.327 ± 0.011 (0/3) | 0.380 ± 0.108 (0/3) | 0.339 ± 0.014 (0/3) |
| EVA + normalization | 0.763 ± 0.013 (1/3) | 0.780 ± 0.010 (3/3) | 0.789 ± 0.005 (3/3) |
| Fed-SB | 0.529 ± 0.063 (0/3) | 0.598 ± 0.082 (0/3) | 0.673 ± 0.096 (0/3) |
| Fed-SB + normalization | 0.515 ± 0.126 (0/3) | 0.614 ± 0.099 (0/3) | 0.653 ± 0.046 (0/3) |
| Ours | 0.793 ± 0.007 (3/3) | 0.806 ± 0.007 (3/3) | 0.809 ± 0.004 (3/3) |

## QNLI

| Basis | 5% | 10% | 20% |
|---|---|---|---|
| Random (FFA-LoRA) | 0.780 ± 0.054 (1/3) | 0.801 ± 0.050 (3/3) | 0.804 ± 0.046 (3/3) |
| Grad-energy | 0.650 ± 0.054 (0/3) | 0.770 ± 0.020 (1/3) | 0.775 ± 0.076 (2/3) |
| Grad-energy + normalization | 0.829 ± 0.028 (3/3) | 0.843 ± 0.039 (3/3) | 0.852 ± 0.029 (3/3) |
| EVA | 0.593 ± 0.106 (0/3) | 0.640 ± 0.167 (0/3) | 0.604 ± 0.189 (0/3) |
| EVA + normalization | 0.803 ± 0.061 (3/3) | 0.846 ± 0.025 (3/3) | 0.838 ± 0.040 (3/3) |
| Fed-SB | 0.673 ± 0.126 (0/3) | 0.706 ± 0.033 (0/3) | 0.710 ± 0.116 (1/3) |
| Fed-SB + normalization | 0.601 ± 0.172 (0/3) | 0.704 ± 0.170 (1/3) | 0.782 ± 0.070 (1/3) |
| Ours | 0.835 ± 0.028 (3/3) | 0.842 ± 0.032 (3/3) | 0.849 ± 0.038 (3/3) |

## SST2

| Basis | 5% | 10% | 20% |
|---|---|---|---|
| Random (FFA-LoRA) | 0.708 ± 0.131 (0/3) | 0.821 ± 0.065 (0/3) | 0.849 ± 0.056 (1/3) |
| Grad-energy | 0.811 ± 0.064 (0/3) | 0.856 ± 0.051 (2/3) | 0.876 ± 0.060 (2/3) |
| Grad-energy + normalization | 0.854 ± 0.003 (2/3) | 0.881 ± 0.022 (3/3) | 0.890 ± 0.030 (3/3) |
| EVA | 0.813 ± 0.017 (0/3) | 0.858 ± 0.042 (3/3) | 0.856 ± 0.039 (3/3) |
| EVA + normalization | 0.648 ± 0.029 (0/3) | 0.638 ± 0.112 (0/3) | 0.700 ± 0.164 (0/3) |
| Fed-SB | 0.711 ± 0.181 (0/3) | 0.761 ± 0.121 (1/3) | 0.808 ± 0.098 (1/3) |
| Fed-SB + normalization | 0.727 ± 0.189 (1/3) | 0.740 ± 0.200 (1/3) | 0.842 ± 0.033 (1/3) |
| Ours | 0.833 ± 0.042 (3/3) | 0.872 ± 0.030 (3/3) | 0.886 ± 0.032 (3/3) |

## Paired comparisons (same task, seed and budget)

| Left − right | Task | n | mean Δ | W / T / L | p |
|---|---|---:|---:|---:|---:|
| Grad-energy + normalization − Grad-energy | mnli | 9 | +0.2836 | 9 / 0 / 0 | 0.0039 |
| Grad-energy + normalization − Grad-energy | qnli | 9 | +0.1096 | 9 / 0 / 0 | 0.0039 |
| Grad-energy + normalization − Grad-energy | sst2 | 9 | +0.0275 | 7 / 1 / 1 | 0.07 |
| Grad-energy + normalization − Grad-energy | all | 27 | +0.1402 | 25 / 1 / 1 | 8e-07 |
| EVA + normalization − EVA | mnli | 9 | +0.4286 | 9 / 0 / 0 | 0.0039 |
| EVA + normalization − EVA | qnli | 9 | +0.2168 | 9 / 0 / 0 | 0.0039 |
| EVA + normalization − EVA | sst2 | 9 | -0.1804 | 0 / 0 / 9 | 0.0039 |
| EVA + normalization − EVA | all | 27 | +0.1550 | 18 / 0 / 9 | 0.12 |
| Fed-SB + normalization − Fed-SB | mnli | 9 | -0.0060 | 4 / 0 / 5 | 1 |
| Fed-SB + normalization − Fed-SB | qnli | 9 | -0.0007 | 6 / 0 / 3 | 0.51 |
| Fed-SB + normalization − Fed-SB | sst2 | 9 | +0.0090 | 4 / 1 / 4 | 1 |
| Fed-SB + normalization − Fed-SB | all | 27 | +0.0008 | 14 / 1 / 12 | 0.85 |
| Grad-energy + normalization − Random (FFA-LoRA) | mnli | 9 | +0.0321 | 9 / 0 / 0 | 0.0039 |
| Grad-energy + normalization − Random (FFA-LoRA) | qnli | 9 | +0.0461 | 9 / 0 / 0 | 0.0039 |
| Grad-energy + normalization − Random (FFA-LoRA) | sst2 | 9 | +0.0821 | 9 / 0 / 0 | 0.0039 |
| Grad-energy + normalization − Random (FFA-LoRA) | all | 27 | +0.0534 | 27 / 0 / 0 | 1.5e-08 |
| EVA + normalization − Random (FFA-LoRA) | mnli | 9 | +0.0134 | 9 / 0 / 0 | 0.0039 |
| EVA + normalization − Random (FFA-LoRA) | qnli | 9 | +0.0339 | 9 / 0 / 0 | 0.0039 |
| EVA + normalization − Random (FFA-LoRA) | sst2 | 9 | -0.1311 | 2 / 0 / 7 | 0.18 |
| EVA + normalization − Random (FFA-LoRA) | all | 27 | -0.0279 | 20 / 0 / 7 | 0.019 |
| Fed-SB + normalization − Random (FFA-LoRA) | mnli | 9 | -0.1701 | 0 / 0 / 9 | 0.0039 |
| Fed-SB + normalization − Random (FFA-LoRA) | qnli | 9 | -0.0996 | 1 / 1 / 7 | 0.07 |
| Fed-SB + normalization − Random (FFA-LoRA) | sst2 | 9 | -0.0236 | 3 / 0 / 6 | 0.51 |
| Fed-SB + normalization − Random (FFA-LoRA) | all | 27 | -0.0977 | 4 / 1 / 22 | 0.00053 |
| Ours − EVA + normalization | mnli | 9 | +0.0252 | 9 / 0 / 0 | 0.0039 |
| Ours − EVA + normalization | qnli | 9 | +0.0129 | 7 / 0 / 2 | 0.18 |
| Ours − EVA + normalization | sst2 | 9 | +0.2017 | 9 / 0 / 0 | 0.0039 |
| Ours − EVA + normalization | all | 27 | +0.0799 | 25 / 0 / 2 | 5.6e-06 |
| Ours − Fed-SB + normalization | mnli | 9 | +0.2087 | 9 / 0 / 0 | 0.0039 |
| Ours − Fed-SB + normalization | qnli | 9 | +0.1463 | 9 / 0 / 0 | 0.0039 |
| Ours − Fed-SB + normalization | sst2 | 9 | +0.0942 | 7 / 0 / 2 | 0.18 |
| Ours − Fed-SB + normalization | all | 27 | +0.1497 | 25 / 0 / 2 | 5.6e-06 |
