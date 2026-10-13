"""Published federated LoRA baselines on the shared FL engine (same data, clients, wire accounting).

flora_dense   FLoRA (Wang et al., NeurIPS 2024).  Clients train a fresh LoRA every round; the
              server stacks the client factors, so sum_k p_k B_k A_k is applied exactly: it is
              merged into the frozen weights and the next round restarts from a new LoRA.  The
              downlink carries every client's factors.
fedex_dense   FedEx-LoRA (Singhal et al., ACL 2025).  A and B are averaged as in FedIT and the
              residual sum_k p_k B_k A_k - Bbar Abar is merged into the frozen weights.  The
              residual is sent in its cheaper exact form (all client factors, or the dense matrix).
rolora_*      RoLoRA (Chen et al., NeurIPS 2025).  Rounds alternate between training and sending
              only B (first) and only A; the other factor is shared, so averaging is exact.
fedsvd_*      FedSVD (Lee et al., NeurIPS 2025).  FFA-LoRA with a Kaiming A; after every
              aggregation BA is refactorised, A <- right singular vectors, B <- U S.  Clients do it
              locally from the shared B, so it costs no bytes.  rowspace(BA) lies in rowspace(A):
              the adapted subspace is fixed at initialisation and only rotated.
fedsb         Fed-SB (Singhal et al., TMLR 2026).  Fixed B = U_r, A = V_r^T from the SVD of the
              LoRA-SB first-step estimate -lr * sign(sum of client gradients); clients train the
              r x r matrix R (initialised to S_r) and send it dense.  r is the largest rank whose
              payload fits the run's wire budget.  Only the basis broadcast is charged, not the
              client gradient upload, so its setup cost is a lower bound.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .adapters import FixedALinear, LoRALinear, LoRAXSLinear, TensorLayout, _module_parent
from .wire import encode_dense


FLORA_METHODS = {"flora_dense"}
FEDEX_METHODS = {"fedex_dense"}
MERGE_METHODS = FLORA_METHODS | FEDEX_METHODS
ROLORA_METHODS = {"rolora_dense", "rolora_ef"}
FEDSVD_METHODS = {"fedsvd_dense", "fedsvd_ef"}
FEDSB_METHODS = {"fedsb"}
BASELINE_DENSE_METHODS = {"flora_dense", "fedex_dense", "rolora_dense", "fedsvd_dense"}
BASELINE_EF_METHODS = {"rolora_ef", "fedsvd_ef"}
BASELINE_METHODS = BASELINE_DENSE_METHODS | BASELINE_EF_METHODS | FEDSB_METHODS
# Adapter construction shared with an existing method.
BASELINE_CONSTRUCTION = {
    "flora_dense": "fedavg_lora",
    "fedex_dense": "fedavg_lora",
    "rolora_dense": "fedavg_lora",
    "rolora_ef": "fedavg_lora",
    "fedsvd_dense": "ffa_lora",
    "fedsvd_ef": "ffa_lora",
    "fedsb": "fedsb",
}
FEDSB_CACHE_VERSION = 1


def _slices(layout: TensorLayout) -> dict[str, tuple[int, int, tuple[int, ...]]]:
    offsets: dict[str, tuple[int, int, tuple[int, ...]]] = {}
    start = 0
    for name, shape, count in zip(layout.names, layout.shapes, layout.numels):
        offsets[name] = (start, start + count, shape)
        start += count
    return offsets


def _factor_modules(layout: TensorLayout, factor: str) -> list[str]:
    suffix = "." + factor
    return [name[: -len(suffix)] for name in layout.names if name.endswith(suffix)]


def rolora_factor(round_index: int) -> str:
    """Factor trained and sent in a 0-based round: B first (A gets no gradient while B = 0)."""
    return "lora_B" if round_index % 2 == 0 else "lora_A"


def factor_mask(layout: TensorLayout, factor: str, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(layout.total_numel, dtype=torch.bool, device=device)
    for name, (start, stop, _) in _slices(layout).items():
        if name.endswith("." + factor):
            mask[start:stop] = True
    if not bool(mask.any()):
        raise RuntimeError(f"layout has no {factor} parameters")
    return mask


def set_trainable_factor(model: nn.Module, factor: str | None) -> None:
    """RoLoRA: only one factor trains in a round; None restores both."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(factor in (None, "lora_A"))
            module.lora_B.requires_grad_(factor in (None, "lora_B"))


def _client_factors(
    layout: TensorLayout, vector: torch.Tensor
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    offsets = _slices(layout)
    factors = {}
    for module_name in _factor_modules(layout, "lora_A"):
        a0, a1, ashape = offsets[module_name + ".lora_A"]
        b0, b1, bshape = offsets[module_name + ".lora_B"]
        factors[module_name] = (vector[a0:a1].view(ashape), vector[b0:b1].view(bshape))
    return factors


@torch.no_grad()
def merge_client_products(
    model: nn.Module,
    layout: TensorLayout,
    client_adapters: list[torch.Tensor],
    client_weights: list[float],
    shared_adapter: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Add scaling * (sum_k p_k B_k A_k - B_shared A_shared) to each frozen weight.

    FLoRA passes shared_adapter=None (the whole product is merged, the next LoRA starts
    fresh); FedEx-LoRA passes the new averaged adapter, so the model equals the exact
    average of the client models.  Returns the per-module weight increments.
    """
    total = float(sum(client_weights))
    probabilities = [float(weight) / total for weight in client_weights]
    modules = dict(model.named_modules())
    per_client = [_client_factors(layout, vector) for vector in client_adapters]
    shared = None if shared_adapter is None else _client_factors(layout, shared_adapter)
    increments: dict[str, torch.Tensor] = {}
    for module_name in _factor_modules(layout, "lora_A"):
        module = modules[module_name]
        if not isinstance(module, LoRALinear):
            raise RuntimeError(f"{module_name} is not a LoRA module")
        product = torch.zeros_like(module.base.weight, dtype=torch.float64)
        for probability, factors in zip(probabilities, per_client):
            a, b = factors[module_name]
            product += probability * (b.double() @ a.double())
        if shared is not None:
            a, b = shared[module_name]
            product -= b.double() @ a.double()
        increment = (module.scaling * product).float()
        module.base.weight += increment.to(module.base.weight.dtype)
        increments[module_name] = increment
    return increments


def stacked_downlink_bytes(
    layout: TensorLayout,
    client_adapters: list[torch.Tensor],
    client_weights: list[float],
    shared_adapter: torch.Tensor | None,
) -> tuple[int, str]:
    """Cheapest exact downlink of the merged product: every client's factors and weights, or
    (FedEx-LoRA only) the averaged adapter plus the dense residual matrices."""
    stacked = encode_dense(torch.cat(client_adapters)).nbytes
    stacked += encode_dense(torch.tensor(client_weights, dtype=torch.float32)).nbytes
    if shared_adapter is None:
        return stacked, "stacked_client_factors"
    dense_numel = sum(
        int(a.shape[1]) * int(b.shape[0]) for a, b in _client_factors(layout, shared_adapter).values()
    )
    dense = encode_dense(shared_adapter).nbytes + 13 + 2 * dense_numel
    if dense < stacked:
        return dense, "averaged_adapter_plus_dense_residual"
    return stacked, "stacked_client_factors"


@torch.no_grad()
def flora_fresh_adapter(model: nn.Module, layout: TensorLayout, basis_seed: int, round_index: int) -> torch.Tensor:
    """Fresh LoRA for a 0-based round >= 1: Kaiming A (PEFT default) from a round seed, B = 0."""
    modules = dict(model.named_modules())
    offsets = _slices(layout)
    vector = torch.zeros(layout.total_numel, dtype=torch.float32, device=next(model.parameters()).device)
    for index, module_name in enumerate(_factor_modules(layout, "lora_A")):
        module = modules[module_name]
        a0, a1, ashape = offsets[module_name + ".lora_A"]
        generator = torch.Generator(device=vector.device).manual_seed(
            int(basis_seed) + 7919 * int(round_index) + index
        )
        value = torch.empty(ashape, dtype=torch.float32, device=vector.device)
        nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
        vector[a0:a1] = value.reshape(-1)
        if module.lora_B.shape != offsets[module_name + ".lora_B"][2]:
            raise RuntimeError("layout and module shapes disagree")
    return vector


@torch.no_grad()
def fedsvd_reparameterize(model: nn.Module, layout: TensorLayout, vectors: list[torch.Tensor]) -> float:
    """Refactorise every B A as (U S)(V^T) in place: fixed_A <- V^T, and B-coordinate vectors
    (global adapter, EF residuals) <- v T with T = A_old A_new^T, which is exact because the
    new rows span the old row space.  Returns the max |B A - B' A'| / |B A| over modules."""
    modules = dict(model.named_modules())
    offsets = _slices(layout)
    worst = 0.0
    for module_name in _factor_modules(layout, "lora_B"):
        module = modules[module_name]
        if not isinstance(module, FixedALinear):
            raise RuntimeError(f"{module_name} is not a fixed-A module")
        b0, b1, bshape = offsets[module_name + ".lora_B"]
        a = module.fixed_A.double()
        q, upper = torch.linalg.qr(a.T, mode="reduced")
        b = vectors[0][b0:b1].view(bshape).double()
        _, _, right = torch.linalg.svd(b @ upper.T, full_matrices=False)
        new_a = right @ q.T
        transform = a @ new_a.T
        before = b @ a
        module.fixed_A.copy_(new_a.float())
        for vector in vectors:
            vector[b0:b1] = (vector[b0:b1].view(bshape).double() @ transform).float().reshape(-1)
        after = vectors[0][b0:b1].view(bshape).double() @ module.fixed_A.double()
        scale = float(torch.linalg.norm(before))
        if scale > 0:
            worst = max(worst, float(torch.linalg.norm(after - before)) / scale)
    return worst


def fixed_a_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: module.fixed_A.detach().cpu().clone()
        for name, module in model.named_modules()
        if isinstance(module, FixedALinear)
    }


@torch.no_grad()
def load_fixed_a_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    modules = dict(model.named_modules())
    for name, value in state.items():
        modules[name].fixed_A.copy_(value.to(modules[name].fixed_A.device))


def merged_weight_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """FLoRA / FedEx-LoRA checkpoint: the frozen weights themselves (bit-exact on resume,
    unlike re-adding accumulated increments in a different order)."""
    return {
        name: module.base.weight.detach().cpu().clone()
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }


@torch.no_grad()
def load_merged_weight_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    modules = dict(model.named_modules())
    for name, value in state.items():
        modules[name].base.weight.copy_(value.to(modules[name].base.weight.device))


def fedsb_rank(model: nn.Module, adapter_cap: int) -> int:
    """Largest common r with a dense FP16 payload of sum_m r^2 values within adapter_cap."""
    modules = [module for module in model.modules() if isinstance(module, LoRAXSLinear)]
    if not modules:
        raise RuntimeError("Fed-SB requires LoRA-XS-style target modules")
    rank = int(math.isqrt(max(adapter_cap - 13, 0) // (2 * len(modules))))
    rank = min(rank, min(min(m.base.in_features, m.base.out_features) for m in modules))
    if rank < 1:
        raise ValueError(f"wire cap {adapter_cap} B is too small for Fed-SB")
    return rank


@torch.no_grad()
def install_fedsb(
    model: nn.Module,
    gradient_sums: list[torch.Tensor],
    rank: int,
    learning_rate: float,
) -> tuple[list[dict[str, torch.Tensor]], int]:
    """Replace each placeholder with B=U_r, R=S_r, A=V_r^T of -lr*sign(G) (LoRA-SB, s = 1).

    Returns the installed bases (for caching) and the broadcast payload per client.
    """
    targets = sorted(
        ((name, module) for name, module in model.named_modules() if isinstance(module, LoRAXSLinear)),
        key=lambda item: item[0],
    )
    if len(targets) != len(gradient_sums):
        raise RuntimeError("one gradient per Fed-SB module is required")
    bases = []
    for (_, module), gradient in zip(targets, gradient_sums):
        update = -float(learning_rate) * torch.sign(gradient.double())
        left, singular, right = torch.linalg.svd(update, full_matrices=False)
        bases.append(
            {
                "left": left[:, :rank].float().cpu(),
                "right": right[:rank].float().cpu(),
                "singular": singular[:rank].float().cpu(),
            }
        )
    load_fedsb_bases(model, bases)
    payload = []
    for basis in bases:
        payload.extend([basis["left"].reshape(-1), basis["right"].reshape(-1), basis["singular"]])
    return bases, encode_dense(torch.cat(payload)).nbytes


@torch.no_grad()
def load_fedsb_bases(model: nn.Module, bases: list[dict[str, torch.Tensor]]) -> None:
    targets = sorted(
        ((name, module) for name, module in model.named_modules() if isinstance(module, LoRAXSLinear)),
        key=lambda item: item[0],
    )
    for (name, module), basis in zip(targets, bases):
        device = module.base.weight.device
        rank = int(basis["singular"].numel())
        replacement = LoRAXSLinear(
            module.base,
            basis["left"].to(device),
            torch.ones(rank, device=device),
            basis["right"].to(device),
            rank,
            float(rank),  # alpha = r: scaling s = 1 as in LoRA-SB
            0.0,
            0.0,
            0,
        )
        replacement.lora_R.copy_(torch.diag(basis["singular"].to(device)))
        parent, child = _module_parent(model, name)
        setattr(parent, child, replacement)


def summary_record(method: str, **values: Any) -> dict[str, Any]:
    return {"method": method, **values}
