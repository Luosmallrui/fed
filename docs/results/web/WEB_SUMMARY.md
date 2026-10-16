# Web-service tasks

Seeds [1001, 1002, 1003]; final score x100, mean ± SD (seeds reaching C95; median C95 MB).

## banking77

Uncompressed FedIT: 90.0

| Method | 5% | 10% |
|---|---|---|
| FFA-LoRA | 84.1 ± 0.7 (0/3; never) | 86.2 ± 0.6 (3/3; 106.5 MB) |
| Grad-energy | 19.3 ± 7.7 (0/3; never) | 25.6 ± 7.6 (0/3; never) |
| FedIT | 83.8 ± 1.2 (0/3; never) | 86.5 ± 0.4 (3/3; 106.5 MB) |
| RoLoRA | 81.2 ± 1.9 (0/3; never) | 83.4 ± 1.1 (0/3; never) |
| FedSVD | 85.8 ± 0.9 (2/3; 79.9 MB) | 87.3 ± 0.5 (3/3; 92.3 MB) |
| Fed-SB | 76.0 ± 4.0 (0/3; never) | 80.2 ± 1.0 (0/3; never) |
| FoCuS | 85.3 ± 1.2 (1/3; 90.4 MB) | 87.8 ± 1.0 (3/3; 99.8 MB) |

## moderation

Uncompressed FedIT: 73.5

| Method | 5% | 10% |
|---|---|---|
| FFA-LoRA | 66.4 ± 8.3 (0/3; never) | 68.6 ± 7.7 (2/3; 31.7 MB) |
| Grad-energy | 51.6 ± 11.6 (0/3; never) | 51.3 ± 11.0 (0/3; never) |
| FedIT | 65.1 ± 11.1 (2/3; 12.7 MB) | 68.8 ± 8.9 (2/3; 20.5 MB) |
| RoLoRA | 67.6 ± 7.0 (0/3; never) | 71.5 ± 4.7 (3/3; 37.3 MB) |
| FedSVD | 65.0 ± 14.1 (2/3; 12.7 MB) | 68.9 ± 10.9 (2/3; 20.5 MB) |
| Fed-SB | 56.9 ± 10.5 (0/3; never) | 63.3 ± 15.9 (2/3; 55.6 MB) |
| FoCuS | 68.7 ± 7.1 (1/3; 13.4 MB) | 70.5 ± 5.2 (3/3; 41.1 MB) |

## FoCuS − method (same task, seed and budget)

| Method | Task | n | mean Δ (points) | W / T / L | p |
|---|---|---:|---:|---:|---:|
| FFA-LoRA | banking77 | 6 | +1.4 | 6 / 0 / 0 | 0.031 |
| FFA-LoRA | moderation | 6 | +2.1 | 6 / 0 / 0 | 0.031 |
| FFA-LoRA | all | 12 | +1.8 | 12 / 0 / 0 | 0.00049 |
| Grad-energy | banking77 | 6 | +64.1 | 6 / 0 / 0 | 0.031 |
| Grad-energy | moderation | 6 | +18.1 | 6 / 0 / 0 | 0.031 |
| Grad-energy | all | 12 | +41.1 | 12 / 0 / 0 | 0.00049 |
| FedIT | banking77 | 6 | +1.4 | 6 / 0 / 0 | 0.031 |
| FedIT | moderation | 6 | +2.6 | 3 / 0 / 3 | 1 |
| FedIT | all | 12 | +2.0 | 9 / 0 / 3 | 0.15 |
| RoLoRA | banking77 | 6 | +4.2 | 6 / 0 / 0 | 0.031 |
| RoLoRA | moderation | 6 | +0.0 | 3 / 1 / 2 | 1 |
| RoLoRA | all | 12 | +2.1 | 9 / 1 / 2 | 0.065 |
| FedSVD | banking77 | 6 | -0.0 | 2 / 2 / 2 | 1 |
| FedSVD | moderation | 6 | +2.6 | 2 / 1 / 3 | 1 |
| FedSVD | all | 12 | +1.3 | 4 / 3 / 5 | 1 |
| Fed-SB | banking77 | 6 | +8.5 | 6 / 0 / 0 | 0.031 |
| Fed-SB | moderation | 6 | +9.5 | 6 / 0 / 0 | 0.031 |
| Fed-SB | all | 12 | +9.0 | 12 / 0 / 0 | 0.00049 |
