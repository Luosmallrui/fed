"""Communication-aware fixed basis (CAB).

For B-only federated LoRA with a shared row-orthonormal A (P = A^T A), the
stationary excess error at the fixed final round under sparse synchronisation is
(see docs/proposals/CLOSED_FORM_NOVELTY.md, Theorem 2)

    J(P) = ||Dbar (I - P)||_F^2 + kappa * tr(P Sigma_het),

the consensus update that P cannot represent plus the noise floor that client
disagreement leaves when clients transmit different coordinates.  With an
unbiased omega-compressor (omega = d / K - 1), local contraction gamma and
N_eff = 1 / sum p_i^2 effective clients,

    kappa = gamma (omega / N_eff) / (2 - gamma (1 + omega / N_eff)).

By Ky Fan's theorem the minimiser over rank-r projections is

    A* = TopEig_r(M),   M = Gbar^T Gbar - kappa * N_eff * sum_i p_i^2 (G_i - Gbar)^T (G_i - Gbar),

with the initial per-client weight gradients G_i standing in for the update
directions (both terms share one scale, so the eigenvectors do not depend on
it).  kappa = 0 gives the consensus-gradient basis.

Protocol, all of it charged to delivered bytes:
  * every client uploads one dense FP16 packet with a rank-k truncated SVD of
    each target-layer gradient, as balanced factors U sqrt(S) and sqrt(S) V^T,
    plus one power-of-two scale exponent;
  * the server rebuilds the low-rank gradients, forms M, takes the top-r
    eigenvectors per layer and broadcasts them once as FP16;
  * every peer re-orthonormalises the decoded rows identically.
At equal bytes this deterministic per-client truncation reached a far smaller
objective gap than a shared Gaussian left sketch on synthetic 768 x 768
gradients (1.004-1.07x vs 1.10-1.25x the exact optimum).  The exact M is also
formed in simulation, for diagnostics only.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import nn

from .wire import decode_packet, encode_dense


# Activation-metric variants (docs/proposals/RESEARCH_DIRECTIONS.md, direction 3).
# The Euclidean objective keeps only the first-order term -eta ||G A^T||^2 of a
# B-only step.  Its curvature, A Sigma_x A^T (x) H_out under a Gauss-Newton /
# K-FAC model, is what RoBERTa's anisotropic inputs make 74x larger along the
# shared mean direction.  Maximising the optimal one-step decrease
#     1/2 tr(M A^T (A Sigma_x A^T)^{-1} A)
# over r-dimensional row spaces (H_out ~ I) gives the top-r generalized
# eigenvectors of (M, Sigma_x), normalised to A Sigma_x A^T = c^2 I with
# c^2 = tr(Sigma_x) / d so E||A x||^2 matches a row-orthonormal random A.
#   gev_a_ef / gev0_a_ef      generalized eigenvectors, CAB kappa / kappa = 0
#   cab0wh_a_ef               ablation: Euclidean kappa = 0 subspace, whitened within
#   randomwh_a_ef             ablation: random subspace, whitened within
#   gev0x_a_ef                oracle: gev0 from the exact (uncompressed) statistics,
#                             simulation-only upper bound; bytes charged as gev0
#   gev0_a_dense              gev0 basis with uncompressed (dense) B synchronisation,
#                             the counterpart of random_a_dense (FFA-LoRA style)
#   eva_a_ef                  baseline: EVA-style A = top-r eigenvectors of the input
#                             second moment (row-orthonormal, no gradients)
#   evawh_a_ef                EVA subspace, whitened within (A Sigma_x A^T = c^2 I): tests the
#                             normalization as a fix for a published data-driven basis
GEV_METHODS = {"gev_a_ef", "gev0_a_ef", "gev0x_a_ef", "gev0_a_dense"}
ORACLE_METHODS = {"gev0x_a_ef"}
WHITENED_SUBSPACE_METHODS = {"cab0wh_a_ef", "randomwh_a_ef", "evawh_a_ef"}
EVA_METHODS = {"eva_a_ef"}
# Rows deliberately not orthonormal (A Sigma_x A^T = c^2 I).
WHITENED_METHODS = GEV_METHODS | WHITENED_SUBSPACE_METHODS
# Methods whose clients upload an input second-moment sketch.
ACTIVATION_METHODS = WHITENED_METHODS | EVA_METHODS
# Basis built without any gradient upload; their gradients are simulation-only diagnostics.
GRADIENT_FREE_METHODS = {"randomwh_a_ef", "eva_a_ef", "evawh_a_ef"}
# CAB-protocol bases synchronised without compression (no wire ratio, kappa = 0).
CAB_DENSE_METHODS = {"gev0_a_dense"}
PAYLOAD_DTYPES = {"float16", "int8"}
ACTIVATION_RECIPE = "cab_activation_sketch_fp16_v1"
ACTIVATION_DEFAULT_RANK = 16
ACTIVATION_DEFAULT_RIDGE = 1e-3

# *_dcef: synchronised with DCEF; *_ef: synchronised with standard bidirectional EF.
CAB_METHODS = {"cab_a_dcef", "cab0_a_dcef", "cab_a_ef", "cab0_a_ef", *ACTIVATION_METHODS}
CAB_KAPPA_ZERO_METHODS = {
    "cab0_a_dcef",
    "cab0_a_ef",
    "gev0_a_ef",
    "gev0x_a_ef",
    "gev0_a_dense",
    *WHITENED_SUBSPACE_METHODS,
    *EVA_METHODS,
}
CAB_EF_METHODS = {"cab_a_ef", "cab0_a_ef", *(ACTIVATION_METHODS - CAB_DENSE_METHODS)}
CAB_RECIPE = "cab_client_svd_fp16_v1"
CAB_CACHE_VERSION = 3


def effective_clients(weights: Iterable[float]) -> float:
    values = [float(value) for value in weights]
    total = sum(values)
    if total <= 0 or any(value < 0 for value in values):
        raise ValueError("client weights must be non-negative with a positive sum")
    return total * total / sum(value * value for value in values)


def kappa_from_budget(gamma: float, adapter_numel: int, sparse_k: int, n_eff: float) -> float:
    """Closed-form noise-floor weight of the stationary excess error."""
    gamma = float(gamma)
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must lie in (0, 1], got {gamma}")
    if not 0 < int(sparse_k) <= int(adapter_numel):
        raise ValueError("sparse_k must lie in [1, adapter_numel]")
    if n_eff < 1.0:
        raise ValueError("n_eff must be at least one")
    omega = int(adapter_numel) / int(sparse_k) - 1.0
    ratio = omega / float(n_eff)
    denominator = 2.0 - gamma * (1.0 + ratio)
    if denominator <= 0.0:
        raise ValueError(
            "gamma is outside the stability region of the unbiased-compressor model "
            f"(gamma={gamma}, omega/N_eff={ratio:.4f}); set cab.kappa_override instead"
        )
    return gamma * ratio / denominator


def canonical_rows(rows: torch.Tensor) -> torch.Tensor:
    """Flip each row so its largest-magnitude entry (lowest index on ties) is positive."""
    rows = rows.clone()
    pivots = rows.abs().argmax(dim=1)
    signs = torch.sign(rows[torch.arange(rows.shape[0]), pivots])
    signs[signs == 0] = 1.0
    return rows * signs[:, None]


def top_eigen_rows(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows = top-``rank`` eigenvectors of a symmetric matrix (descending), and all eigenvalues."""
    sym = 0.5 * (matrix.double() + matrix.double().T)
    values, vectors = torch.linalg.eigh(sym.cpu())
    order = torch.argsort(values, descending=True, stable=True)
    values = values[order]
    rows = vectors[:, order[:rank]].T.contiguous()
    return canonical_rows(rows), values


def orthonormalize_rows(rows: torch.Tensor) -> torch.Tensor:
    """Deterministic row orthonormalisation in CPU float64 (QR with positive R diagonal)."""
    q, r = torch.linalg.qr(rows.double().cpu().T, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1.0
    return (q * signs[None, :]).T.contiguous()


def subspace_overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    """||A1 A2^T||_F^2 / r for row-orthonormal A1, A2: 1 = same subspace, r/n = random."""
    product = first.double().cpu() @ second.double().cpu().T
    return float((product * product).sum().item() / first.shape[0])


def truncated_factors(gradient: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced rank-``rank`` factors (U sqrt(S), sqrt(S) V^T) of one gradient."""
    left, singular, right = torch.linalg.svd(gradient.double(), full_matrices=False)
    k = min(int(rank), singular.numel())
    root = singular[:k].sqrt()
    return left[:, :k] * root[None, :], root[:, None] * right[:k]


class LayerStatistics:
    """Streaming second-moment statistics of one target layer.

    Two copies are kept: ``reported`` from what crossed the wire (the protocol
    input) and ``exact`` from the true gradients (simulation-only diagnostics).
    With normalised weights p_i, Gbar = sum p_i G_i, W2 = sum p_i^2 G_i,
    Q = sum p_i^2 G_i^T G_i and P2 = sum p_i^2,
        sum p_i^2 (G_i - Gbar)^T (G_i - Gbar) = Q - W2^T Gbar - Gbar^T W2 + P2 Gbar^T Gbar.
    """

    def __init__(self, out_features: int, in_features: int, device: torch.device | str = "cpu"):
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.device = torch.device(device)
        options = {"dtype": torch.float64, "device": self.device}
        self._sums = {
            kind: {
                "mean": torch.zeros(out_features, in_features, **options),
                "weighted": torch.zeros(out_features, in_features, **options),
                "gram": torch.zeros(in_features, in_features, **options),
            }
            for kind in ("exact", "reported")
        }
        self._weights = {"exact": [0.0, 0.0], "reported": [0.0, 0.0]}

    def _add(self, kind: str, gradient: torch.Tensor, weight: float) -> None:
        if tuple(gradient.shape) != (self.out_features, self.in_features):
            raise ValueError(f"gradient has shape {tuple(gradient.shape)}")
        g = gradient.to(device=self.device, dtype=torch.float64)
        sums = self._sums[kind]
        sums["mean"] += weight * g
        sums["weighted"] += weight * weight * g
        sums["gram"] += weight * weight * (g.T @ g)
        self._weights[kind][0] += weight
        self._weights[kind][1] += weight * weight

    def add_exact(self, gradient: torch.Tensor, weight: float) -> None:
        self._add("exact", gradient, weight)

    def add_reported(self, left: torch.Tensor, right: torch.Tensor, weight: float) -> None:
        self._add("reported", left.double() @ right.double(), weight)

    def _terms(self, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
        total, p2 = self._weights[kind]
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("CAB statistics require weights that sum to one")
        sums = self._sums[kind]
        mean, weighted = sums["mean"], sums["weighted"]
        consensus = mean.T @ mean
        het = sums["gram"] - weighted.T @ mean - mean.T @ weighted + p2 * consensus
        return consensus, 0.5 * (het + het.T)

    def matrices(self, kappa: float, n_eff: float) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for kind in ("exact", "reported"):
            consensus, het = self._terms(kind)
            out[f"{kind}_consensus"] = consensus
            out[f"{kind}_het"] = n_eff * het
            out[f"{kind}_m"] = consensus - kappa * n_eff * het
        return out


def encode_client_payload(tensors: list[torch.Tensor]) -> tuple[list[torch.Tensor], int]:
    """Serialise one client's factors as one dense FP16 packet.

    A power-of-two scale (sent as its exponent, exact in FP16) keeps the largest
    value near one so small gradients do not underflow.  Returns the decoded,
    rescaled tensors and the packet size in bytes.
    """
    flat = torch.cat([value.float().reshape(-1) for value in tensors])
    peak = float(flat.abs().max().item())
    exponent = 0 if peak == 0.0 else -int(math.floor(math.log2(peak)))
    exponent = max(-14, min(14, exponent))
    payload = torch.cat([torch.tensor([float(exponent)]), flat.cpu() * (2.0 ** exponent)])
    packet = encode_dense(payload)
    decoded = decode_packet(packet, "cpu")
    if int(decoded[0].item()) != exponent:
        raise AssertionError("CAB payload exponent did not survive the FP16 wire")
    values = decoded[1:].double() * (2.0 ** -exponent)
    out, offset = [], 0
    for value in tensors:
        count = value.numel()
        out.append(values[offset : offset + count].reshape(value.shape))
        offset += count
    return out, packet.nbytes


INT8_HEADER_BYTES = 13  # same fixed header as a dense packet


def encode_int8_rows(tensors: list[torch.Tensor]) -> tuple[list[torch.Tensor], int]:
    """Symmetric int8 quantisation with one FP16 scale per row of each 2-D tensor
    (1-D tensors count as one row).  Row scales share one power-of-two exponent
    (sent as FP16, exact) so tiny gradients do not underflow FP16.  Returns the
    decoded tensors and the packet size: header + exponent + one byte per value
    + two bytes per row scale."""
    matrices = [value.double().cpu().reshape(1, -1) if value.dim() == 1 else value.double().cpu() for value in tensors]
    scales = [matrix.abs().amax(dim=1, keepdim=True) / 127.0 for matrix in matrices]
    largest = max((float(scale.max().item()) for scale in scales), default=0.0)
    exponent = 0 if largest == 0.0 else max(-14, min(14, -int(math.floor(math.log2(largest)))))
    out: list[torch.Tensor] = []
    nbytes = INT8_HEADER_BYTES + 2
    for value, matrix, scale in zip(tensors, matrices, scales):
        sent = (scale * (2.0 ** exponent)).to(torch.float16).double() * (2.0 ** -exponent)
        safe = torch.where(sent > 0, sent, torch.ones_like(sent))
        quantised = torch.clamp(torch.round(matrix / safe), -127, 127)
        out.append((quantised * sent).reshape(value.shape))
        nbytes += matrix.numel() + 2 * matrix.shape[0]
    return out, nbytes


def encode_factor_payload(tensors: list[torch.Tensor], dtype: str) -> tuple[list[torch.Tensor], int]:
    """One client's truncated-gradient factors [U sqrt(S), sqrt(S) V^T, ...] in the
    chosen wire dtype.  int8 quantises the left factors per column (each has k
    columns) and the right factors per row, so every scale covers one rank-1 term."""
    if dtype == "float16":
        return encode_client_payload(tensors)
    if dtype != "int8":
        raise ValueError(f"unsupported payload dtype {dtype}")
    oriented = [value.T if index % 2 == 0 else value for index, value in enumerate(tensors)]
    decoded, nbytes = encode_int8_rows(oriented)
    return [value.T if index % 2 == 0 else value for index, value in enumerate(decoded)], nbytes


def broadcast_basis(
    rows_by_layer: list[torch.Tensor], orthonormalize: bool = True, dtype: str = "float16"
) -> tuple[list[torch.Tensor], int]:
    """Send all layer bases once (FP16, or int8 with a per-row scale); re-orthonormalise
    what arrives unless the basis is deliberately non-orthonormal (whitened rows)."""
    if dtype == "int8":
        received_rows, nbytes = encode_int8_rows(rows_by_layer)
        return [orthonormalize_rows(rows) if orthonormalize else rows for rows in received_rows], nbytes
    if dtype != "float16":
        raise ValueError(f"unsupported basis dtype {dtype}")
    flat = torch.cat([rows.float().reshape(-1) for rows in rows_by_layer])
    packet = encode_dense(flat)
    decoded = decode_packet(packet, "cpu")
    out, offset = [], 0
    for rows in rows_by_layer:
        count = rows.numel()
        received = decoded[offset : offset + count].reshape(rows.shape).double()
        out.append(orthonormalize_rows(received) if orthonormalize else received)
        offset += count
    return out, packet.nbytes


def _regularised(metric: torch.Tensor, ridge: float) -> torch.Tensor:
    sym = 0.5 * (metric.double().cpu() + metric.double().cpu().T)
    d = sym.shape[0]
    scale = float(torch.trace(sym).item()) / d
    if not scale > 0:
        raise ValueError("activation second moment must have a positive trace")
    return sym + float(ridge) * scale * torch.eye(d, dtype=torch.float64)


def _inverse_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(matrix)
    return (vectors * values.clamp_min(1e-300).rsqrt()[None, :]) @ vectors.T


def gram_refine_mix(gram: torch.Tensor, trace: float, dim: int) -> torch.Tensor:
    """r x r mixing M with (M A) Sigma (M A)^T = (tr(Sigma) / d) I, given the exact
    Gram A Sigma A^T of the broadcast rows: M = sqrt(tr(Sigma) / d) Gram^{-1/2}."""
    sym = 0.5 * (gram.double() + gram.double().T)
    return math.sqrt(float(trace) / int(dim)) * _inverse_sqrt(sym)


def generalized_top_rows(
    matrix: torch.Tensor, metric: torch.Tensor, rank: int, ridge: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``rank`` generalized eigenvectors of (matrix, metric + ridge) as rows.

    Returns rows with A (metric + ridge) A^T = I and all generalized eigenvalues
    (descending).  Solved as the ordinary eigenproblem of S^{-1/2} M S^{-1/2}.
    """
    inv_sqrt = _inverse_sqrt(_regularised(metric, ridge))
    whitened = inv_sqrt @ (0.5 * (matrix.double().cpu() + matrix.double().cpu().T)) @ inv_sqrt
    rows, values = top_eigen_rows(whitened, rank)
    return canonical_rows(rows @ inv_sqrt), values


def whiten_rows(rows: torch.Tensor, metric: torch.Tensor, ridge: float) -> torch.Tensor:
    """Same row space, re-mixed so that A (metric + ridge) A^T = I."""
    rows = rows.double().cpu()
    gram = rows @ _regularised(metric, ridge) @ rows.T
    return canonical_rows(_inverse_sqrt(0.5 * (gram + gram.T)) @ rows)


def scale_to_reference(rows: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    """Scale A (metric) A^T = I rows by c = sqrt(tr(metric) / d), so E||A x||^2 =
    r tr(Sigma) / d, the expectation for a row-orthonormal random A."""
    d = metric.shape[0]
    return rows.double().cpu() * math.sqrt(float(torch.trace(metric.double().cpu()).item()) / d)


def activation_sketch(second_moment: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-``rank`` spike plus diagonal remainder: Sigma ~ F F^T + diag(rest).

    The anisotropy that breaks the Euclidean objective lives in a few spike
    directions, so a short eigen-sketch keeps it while the bulk is diagonal.
    """
    sym = 0.5 * (second_moment.double() + second_moment.double().T)
    values, vectors = torch.linalg.eigh(sym)
    order = torch.argsort(values, descending=True, stable=True)[: int(rank)]
    factor = vectors[:, order] * values[order].clamp_min(0.0).sqrt()[None, :]
    rest = (torch.diagonal(sym) - (factor * factor).sum(dim=1)).clamp_min(0.0)
    return factor, rest


def sketch_matrix(factor: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
    return factor.double() @ factor.double().T + torch.diag(rest.double())


def kfac_capture(
    rows: torch.Tensor,
    matrix: torch.Tensor,
    metric: torch.Tensor,
    ridge: float,
    best: float | None = None,
) -> float:
    """tr(M A^T (A S A^T)^{-1} A) / (sum of the top-r generalized eigenvalues):
    the fraction of the optimal one-step K-FAC decrease a basis attains (PSD M).
    Invariant to any invertible re-mixing of the rows."""
    rows = rows.double().cpu()
    regular = _regularised(metric, ridge)
    sym = 0.5 * (matrix.double().cpu() + matrix.double().cpu().T)
    gram = rows @ regular @ rows.T
    captured = float(torch.trace(torch.linalg.solve(gram, rows @ sym @ rows.T)).item())
    if best is None:
        _, values = generalized_top_rows(sym, metric, rows.shape[0], ridge)
        best = float(values[: rows.shape[0]].clamp_min(0.0).sum().item())
    return captured / best if best > 0 else float("nan")


def anisotropy(rows: torch.Tensor, second_moment: torch.Tensor, mean: torch.Tensor) -> dict[str, float]:
    """common_mode = ||A xbar||^2 / E||A x||^2; energy_ratio = E||A x||^2 / (r tr(S) / d)."""
    rows = rows.double().cpu()
    second = second_moment.double().cpu()
    energy = float(torch.trace(rows @ second @ rows.T).item())
    projected_mean = float((rows @ mean.double().cpu()).pow(2).sum().item())
    reference = rows.shape[0] * float(torch.trace(second).item()) / second.shape[0]
    return {"common_mode": projected_mean / energy, "energy_ratio": energy / reference}


def client_input_moments(
    model: nn.Module,
    modules: list[nn.Module],
    groups: list[int],
    dataset: Any,
    client_indices: torch.Tensor,
    examples: int,
    batch_size: int,
    seed: int,
    mixed_precision_dtype: torch.dtype,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Token-averaged input second and first moments of each module group.

    Uses the same seeded example sample as ``client_weight_gradients``, skips
    padding tokens (they receive no gradient), and computes each group (modules
    sharing one input, e.g. query and value of a layer) once.
    """
    count = max(groups) + 1
    first_module = {}
    for module, group in zip(modules, groups):
        first_module.setdefault(group, module)
    seconds: list[torch.Tensor | None] = [None] * count
    firsts: list[torch.Tensor | None] = [None] * count
    tokens = [0] * count
    state: dict[str, torch.Tensor] = {}

    def make_hook(group: int):
        def hook(_module, inputs):
            x = inputs[0].detach().double()
            mask = state.get("mask")
            if mask is None:  # inputs without a token axis or padding
                mask = torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
            mask = mask.to(x.dtype).reshape(*x.shape[:-1], 1)
            flat = (x * mask).reshape(-1, x.shape[-1])
            gram = flat.T @ flat
            total = flat.sum(dim=0)
            seconds[group] = gram if seconds[group] is None else seconds[group] + gram
            firsts[group] = total if firsts[group] is None else firsts[group] + total
            tokens[group] += int(mask.sum().item())

        return hook

    was_training = model.training
    handles = [first_module[group].register_forward_pre_hook(make_hook(group)) for group in range(count)]
    try:
        model.eval()
        generator = torch.Generator(device=client_indices.device).manual_seed(int(seed))
        order = torch.randperm(client_indices.numel(), generator=generator, device=client_indices.device)
        take = order[: min(int(examples), client_indices.numel())]
        with torch.no_grad():
            for start in range(0, take.numel(), int(batch_size)):
                positions = take[start : start + int(batch_size)]
                batch = dataset.batch(client_indices.index_select(0, positions))
                state["mask"] = batch.get("attention_mask")
                with torch.autocast(
                    device_type=client_indices.device.type,
                    dtype=mixed_precision_dtype,
                    enabled=client_indices.device.type == "cuda",
                ):
                    model(**batch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if any(value is None or tokens[g] == 0 for g, value in enumerate(seconds)):
        raise RuntimeError("activation moment sample is empty")
    return (
        [seconds[g] / tokens[g] for g in range(count)],
        [firsts[g] / tokens[g] for g in range(count)],
    )


def client_weight_gradients(
    model: nn.Module,
    weights: list[torch.Tensor],
    dataset: Any,
    client_indices: torch.Tensor,
    examples: int,
    batch_size: int,
    seed: int,
    mixed_precision_dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Mean loss gradient with respect to frozen target weights on one client.

    Uses evaluation mode (no dropout), a seeded sample of at most ``examples``
    client examples, and ``torch.autograd.grad`` so no parameter ``.grad`` or
    ``requires_grad`` state is left modified.
    """
    flags = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    was_training = model.training
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for weight in weights:
            weight.requires_grad_(True)
        model.eval()
        generator = torch.Generator(device=client_indices.device).manual_seed(int(seed))
        order = torch.randperm(client_indices.numel(), generator=generator, device=client_indices.device)
        take = order[: min(int(examples), client_indices.numel())]
        sums = [torch.zeros_like(weight, dtype=torch.float32) for weight in weights]
        seen = 0
        for start in range(0, take.numel(), int(batch_size)):
            positions = take[start : start + int(batch_size)]
            batch = dataset.batch(client_indices.index_select(0, positions))
            with torch.autocast(
                device_type=client_indices.device.type,
                dtype=mixed_precision_dtype,
                enabled=client_indices.device.type == "cuda",
            ):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite CAB gradient loss: {loss.item()}")
            grads = torch.autograd.grad(loss, weights)
            count = int(positions.numel())
            for total, grad in zip(sums, grads):
                total += grad.float() * count
            seen += count
        if seen == 0:
            raise RuntimeError("CAB gradient sample is empty")
        return [total / seen for total in sums]
    finally:
        for parameter in model.parameters():
            parameter.requires_grad_(flags[id(parameter)])
        model.train(was_training)
