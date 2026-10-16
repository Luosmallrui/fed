# Web-service tasks

Seeds [1001, 1002, 1003]; final score x100, mean ± SD (seeds reaching C95; median C95 MB).

## banking77

Uncompressed FedIT: 90.8

| Method | 5% | 10% |
|---|---|---|
| FFA-LoRA |  | 87.0 ± 0.2 (3/3; 532.4 MB) |
| Grad-energy |  | 58.8 ± 1.7 (0/3; never) |
| FedIT |  | 87.4 ± 0.4 (3/3; 461.5 MB) |
| RoLoRA |  | 84.3 ± 1.0 (0/3; never) |
| FedSVD |  | 89.6 ± 0.2 (3/3; 319.5 MB) |
| Fed-SB |  | 81.9 ± 0.6 (0/3; never) |
| FoCuS |  | 89.0 ± 0.3 (3/3; 463.7 MB) |

## moderation



| Method | 5% | 10% |
|---|---|---|
| FFA-LoRA |  |  |
| Grad-energy |  |  |
| FedIT |  |  |
| RoLoRA |  |  |
| FedSVD |  |  |
| Fed-SB |  |  |
| FoCuS |  |  |

## FoCuS − method (same task, seed and budget)

| Method | Task | n | mean Δ (points) | W / T / L | p |
|---|---|---:|---:|---:|---:|
| FFA-LoRA | banking77 | 3 | +2.0 | 3 / 0 / 0 | 0.25 |
| FFA-LoRA | all | 3 | +2.0 | 3 / 0 / 0 | 0.25 |
| Grad-energy | banking77 | 3 | +30.2 | 3 / 0 / 0 | 0.25 |
| Grad-energy | all | 3 | +30.2 | 3 / 0 / 0 | 0.25 |
| FedIT | banking77 | 3 | +1.6 | 3 / 0 / 0 | 0.25 |
| FedIT | all | 3 | +1.6 | 3 / 0 / 0 | 0.25 |
| RoLoRA | banking77 | 3 | +4.7 | 3 / 0 / 0 | 0.25 |
| RoLoRA | all | 3 | +4.7 | 3 / 0 / 0 | 0.25 |
| FedSVD | banking77 | 3 | -0.6 | 0 / 1 / 2 | 0.5 |
| FedSVD | all | 3 | -0.6 | 0 / 1 / 2 | 0.5 |
| Fed-SB | banking77 | 3 | +7.0 | 3 / 0 / 0 | 0.25 |
| Fed-SB | all | 3 | +7.0 | 3 / 0 / 0 | 0.25 |
