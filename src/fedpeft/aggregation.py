from __future__ import annotations

import torch


def weighted_average(vectors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    if not vectors or len(vectors) != len(weights):
        raise ValueError("vectors and weights must be non-empty and aligned")
    if any(vector.shape != vectors[0].shape for vector in vectors):
        raise ValueError("all vectors must have the same shape")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    stacked = torch.stack([vector.float() for vector in vectors], dim=0)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=stacked.device) / total
    shape = (len(weights),) + (1,) * (stacked.ndim - 1)
    return (stacked * weight_tensor.reshape(shape)).sum(dim=0)

