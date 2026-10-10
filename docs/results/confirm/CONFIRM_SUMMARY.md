# Confirmation stage summary

Official validation, seeds [1001, 1002, 1003], phase final.  Cells: mean ± SD of the final score (seeds reaching C95, median C95 bytes including the one-time basis setup).

## MNLI

| Method | dense | 5% | 10% | 20% |
|---|---|---|---|---|
| fedavg_lora | 0.810 ± 0.004 (3/3, 297.3 MB) | | | |
| random_a_dense | 0.777 ± 0.011 (2/3, 239.7 MB) | | | |
| gev0_a_dense | 0.811 ± 0.007 (3/3, 97.4 MB) | | | |
| random_a_ef | | 0.749 ± 0.017 (0/3) | 0.766 ± 0.015 (1/3, 47.6 MB) | 0.777 ± 0.009 (2/3, 89.2 MB) |
| spectral_a_ef | | 0.751 ± 0.013 (0/3) | 0.770 ± 0.013 (1/3, 47.6 MB) | 0.778 ± 0.009 (3/3, 95.1 MB) |
| cab0_a_ef | | 0.472 ± 0.119 (0/3) | 0.613 ± 0.022 (0/3) | 0.453 ± 0.171 (0/3) |
| cab0wh_a_ef | | 0.790 ± 0.010 (3/3, 31.4 MB) | 0.797 ± 0.009 (3/3, 43.2 MB) | 0.801 ± 0.004 (3/3, 67.0 MB) |
| gev0_a_ef | | 0.793 ± 0.007 (3/3, 28.4 MB) | 0.806 ± 0.007 (3/3, 37.3 MB) | 0.809 ± 0.004 (3/3, 55.1 MB) |
| eva_a_ef | |  | 0.380 ± 0.108 (0/3) |  |

## QNLI

| Method | dense | 5% | 10% | 20% |
|---|---|---|---|---|
| fedavg_lora | 0.827 ± 0.069 (3/3, 106.8 MB) | | | |
| random_a_dense | 0.797 ± 0.070 (3/3, 71.5 MB) | | | |
| gev0_a_dense | 0.839 ± 0.051 (3/3, 43.3 MB) | | | |
| random_a_ef | | 0.780 ± 0.054 (1/3, 14.2 MB) | 0.801 ± 0.050 (3/3, 24.9 MB) | 0.804 ± 0.046 (3/3, 42.7 MB) |
| spectral_a_ef | | 0.801 ± 0.047 (2/3, 15.1 MB) | 0.812 ± 0.041 (3/3, 21.4 MB) | 0.825 ± 0.046 (3/3, 35.6 MB) |
| cab0_a_ef | | 0.650 ± 0.054 (0/3) | 0.770 ± 0.020 (1/3, 41.5 MB) | 0.775 ± 0.076 (2/3, 70.0 MB) |
| cab0wh_a_ef | | 0.829 ± 0.028 (3/3, 14.7 MB) | 0.843 ± 0.039 (3/3, 18.3 MB) | 0.852 ± 0.029 (3/3, 21.8 MB) |
| gev0_a_ef | | 0.835 ± 0.028 (3/3, 14.7 MB) | 0.842 ± 0.032 (3/3, 21.8 MB) | 0.849 ± 0.038 (3/3, 21.8 MB) |
| eva_a_ef | |  | 0.640 ± 0.167 (0/3) |  |

## SST2

| Method | dense | 5% | 10% | 20% |
|---|---|---|---|---|
| fedavg_lora | 0.906 ± 0.022 (3/3, 83.1 MB) | | | |
| random_a_dense | 0.860 ± 0.040 (2/3, 122.2 MB) | | | |
| gev0_a_dense | 0.898 ± 0.026 (3/3, 55.3 MB) | | | |
| random_a_ef | | 0.708 ± 0.131 (0/3) | 0.821 ± 0.065 (0/3) | 0.849 ± 0.056 (1/3, 52.2 MB) |
| spectral_a_ef | | 0.628 ± 0.205 (1/3, 16.6 MB) | 0.676 ± 0.185 (1/3, 26.1 MB) | 0.786 ± 0.145 (1/3, 45.1 MB) |
| cab0_a_ef | | 0.811 ± 0.064 (0/3) | 0.856 ± 0.051 (2/3, 37.3 MB) | 0.876 ± 0.060 (2/3, 45.1 MB) |
| cab0wh_a_ef | | 0.854 ± 0.003 (2/3, 18.0 MB) | 0.881 ± 0.022 (3/3, 33.7 MB) | 0.890 ± 0.030 (3/3, 50.3 MB) |
| gev0_a_ef | | 0.833 ± 0.042 (3/3, 20.6 MB) | 0.872 ± 0.030 (3/3, 36.0 MB) | 0.886 ± 0.032 (3/3, 40.8 MB) |
| eva_a_ef | |  | 0.858 ± 0.042 (3/3, 37.8 MB) |  |

## Paired comparisons (same task, seed and budget)

Win / tie / loss with |Δ| ≤ 0.003 as a tie; p = exact two-sided sign test over non-tied pairs.

| Left − right | Task | n | mean Δ | W / T / L | sign-test p |
|---|---|---:|---:|---:|---:|
| gev0_a_ef − random_a_ef | mnli | 9 | +0.0386 | 9 / 0 / 0 | 0.0039 |
| gev0_a_ef − random_a_ef | qnli | 9 | +0.0468 | 9 / 0 / 0 | 0.0039 |
| gev0_a_ef − random_a_ef | sst2 | 9 | +0.0706 | 9 / 0 / 0 | 0.0039 |
| gev0_a_ef − random_a_ef | all | 27 | +0.0520 | 27 / 0 / 0 | 1.5e-08 |
| gev0_a_ef − spectral_a_ef | mnli | 9 | +0.0363 | 9 / 0 / 0 | 0.0039 |
| gev0_a_ef − spectral_a_ef | qnli | 9 | +0.0291 | 8 / 1 / 0 | 0.0078 |
| gev0_a_ef − spectral_a_ef | sst2 | 9 | +0.1670 | 6 / 0 / 3 | 0.51 |
| gev0_a_ef − spectral_a_ef | all | 27 | +0.0775 | 23 / 1 / 3 | 8.8e-05 |
| gev0_a_ef − cab0wh_a_ef | mnli | 9 | +0.0066 | 7 / 1 / 1 | 0.07 |
| gev0_a_ef − cab0wh_a_ef | qnli | 9 | +0.0007 | 4 / 3 / 2 | 0.69 |
| gev0_a_ef − cab0wh_a_ef | sst2 | 9 | -0.0115 | 2 / 1 / 6 | 0.29 |
| gev0_a_ef − cab0wh_a_ef | all | 27 | -0.0014 | 13 / 5 / 9 | 0.52 |
| cab0wh_a_ef − random_a_ef | mnli | 9 | +0.0321 | 9 / 0 / 0 | 0.0039 |
| cab0wh_a_ef − random_a_ef | qnli | 9 | +0.0461 | 9 / 0 / 0 | 0.0039 |
| cab0wh_a_ef − random_a_ef | sst2 | 9 | +0.0821 | 9 / 0 / 0 | 0.0039 |
| cab0wh_a_ef − random_a_ef | all | 27 | +0.0534 | 27 / 0 / 0 | 1.5e-08 |
| cab0_a_ef − random_a_ef | mnli | 9 | -0.2515 | 0 / 0 / 9 | 0.0039 |
| cab0_a_ef − random_a_ef | qnli | 9 | -0.0635 | 1 / 0 / 8 | 0.039 |
| cab0_a_ef − random_a_ef | sst2 | 9 | +0.0545 | 9 / 0 / 0 | 0.0039 |
| cab0_a_ef − random_a_ef | all | 27 | -0.0868 | 10 / 0 / 17 | 0.25 |
| eva_a_ef − random_a_ef | mnli | 3 | -0.3855 | 0 / 0 / 3 | 0.25 |
| eva_a_ef − random_a_ef | qnli | 3 | -0.1605 | 0 / 1 / 2 | 0.5 |
| eva_a_ef − random_a_ef | sst2 | 3 | +0.0367 | 3 / 0 / 0 | 0.25 |
| eva_a_ef − random_a_ef | all | 9 | -0.1698 | 3 / 1 / 5 | 0.73 |
| gev0_a_dense − random_a_dense | mnli | 3 | +0.0340 | 3 / 0 / 0 | 0.25 |
| gev0_a_dense − random_a_dense | qnli | 3 | +0.0427 | 3 / 0 / 0 | 0.25 |
| gev0_a_dense − random_a_dense | sst2 | 3 | +0.0375 | 3 / 0 / 0 | 0.25 |
| gev0_a_dense − random_a_dense | all | 9 | +0.0380 | 9 / 0 / 0 | 0.0039 |
| gev0_a_dense − fedavg_lora | mnli | 3 | +0.0014 | 2 / 0 / 1 | 1 |
| gev0_a_dense − fedavg_lora | qnli | 3 | +0.0126 | 2 / 0 / 1 | 1 |
| gev0_a_dense − fedavg_lora | sst2 | 3 | -0.0080 | 0 / 1 / 2 | 0.5 |
| gev0_a_dense − fedavg_lora | all | 9 | +0.0020 | 4 / 1 / 4 | 1 |
