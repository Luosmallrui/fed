from __future__ import annotations

import math
import os
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from .cab import WHITENED_METHODS


STANDARD_LORA_METHODS = {"fedavg_lora", "flasc", "full_lora_ef"}
FIXED_RANDOM_METHODS = {
    "ffa_lora",
    "random_a_ef",
    "random_a_topk",
    "random_a_dense",
    "random_block_ef",
}
HYBRID4_METHODS = {
    "hybrid4_a_dense",
    "hybrid4_a_decay_dense",
    "hybrid4_a_ef",
    "hybrid4_a_decay_ef",
}
STABILITY_SPECTRAL_METHODS = {
    "spectral_rot_ef",
    "spectral_block_ef",
    *HYBRID4_METHODS,
}
FIXED_SPECTRAL_METHODS = {
    "spectral_a_ef",
    "spectral_a_topk",
    "spectral_a_dense",
    *STABILITY_SPECTRAL_METHODS,
}
# Fixed A that is deliberately not row-orthonormal (PEFT Kaiming init: FFA-LoRA, FedSVD).
NON_ORTHONORMAL_FIXED_A_METHODS = {"ffa_lora", "fedsvd_dense", "fedsvd_ef"}


class RobertaLinearClassifier(nn.Module):
    """RoBERTa with a small, explicitly accounted-for classification head."""

    def __init__(self, model_path: str, num_labels: int, dtype: torch.dtype):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            add_pooling_layer=False,
        )
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(float(getattr(self.encoder.config, "hidden_dropout_prob", 0.1)))
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, labels: torch.Tensor | None = None, **inputs):
        output = self.encoder(**inputs)
        logits = self.classifier(self.dropout(output.last_hidden_state[:, 0]))
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits)


DECODER_MODEL_TYPES = {"qwen2", "llama", "mistral"}


def is_decoder_model(model_path: str) -> bool:
    """Decided from the local config.json; anything without one is treated as the encoder path."""
    config_file = Path(model_path) / "config.json"
    if not config_file.is_file():
        return False
    return str(json.loads(config_file.read_text(encoding="utf-8")).get("model_type")) in DECODER_MODEL_TYPES


class DecoderLastTokenClassifier(nn.Module):
    """Decoder-only LM backbone (no LM head) with the same small linear head as the
    encoder path, applied to the last non-padding token (inputs are right-padded)."""

    def __init__(self, model_path: str, num_labels: int, dtype: torch.dtype):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True, dtype=dtype)
        hidden = int(self.encoder.config.hidden_size)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, labels: torch.Tensor | None = None, **inputs):
        mask = inputs["attention_mask"]
        output = self.encoder(input_ids=inputs["input_ids"], attention_mask=mask, use_cache=False)
        last = mask.long().sum(dim=1) - 1
        hidden = output.last_hidden_state[torch.arange(last.numel(), device=last.device), last]
        logits = self.classifier(hidden.to(self.classifier.weight.dtype))
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float, seed: int):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        generator = torch.Generator(device=base.weight.device).manual_seed(seed)
        with torch.no_grad():
            # Match PEFT's default LoRA initialization exactly.
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + update.to(base.dtype) * self.scaling


class FixedALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        fixed_a: torch.Tensor,
        rank: int,
        alpha: float,
        dropout: float,
        basis_audit: dict[str, float | int | str] | None = None,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("fixed_A", fixed_a.detach().float().contiguous(), persistent=True)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        # Construction-only diagnostics are ordinary Python metadata: they do
        # not add trainable parameters, state-dict tensors, or wire bytes.
        self.basis_audit = dict(basis_audit or {})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.fixed_A), self.lora_B)
        return base + update.to(base.dtype) * self.scaling


class LoRAXSLinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        left: torch.Tensor,
        singular: torch.Tensor,
        right: torch.Tensor,
        rank: int,
        alpha: float,
        dropout: float,
        init_std: float,
        seed: int,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("fixed_left", left.detach().float().contiguous(), persistent=True)
        self.register_buffer("fixed_right", right.detach().float().contiguous(), persistent=True)
        self.register_buffer("fixed_singular", singular.detach().float().contiguous(), persistent=True)
        self.lora_R = nn.Parameter(torch.empty(rank, rank, device=base.weight.device, dtype=torch.float32))
        generator = torch.Generator(device=base.weight.device).manual_seed(seed)
        with torch.no_grad():
            self.lora_R.normal_(mean=0.0, std=init_std, generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        hidden = F.linear(self.dropout(x), self.fixed_right)
        hidden = hidden * self.fixed_singular
        # The reference LoRA-XS construction uses
        #   delta_W = U_r @ R @ diag(S_r) @ Vh_r.
        # F.linear applies matrices transposed, so the singular values must be
        # applied before the trainable R in the forward path.
        hidden = F.linear(hidden, self.lora_R)
        update = F.linear(hidden, self.fixed_left)
        return base + update.to(base.dtype) * self.scaling


@dataclass
class TensorLayout:
    names: list[str]
    shapes: list[tuple[int, ...]]
    numels: list[int]

    @classmethod
    def from_model(cls, model: nn.Module, names: Iterable[str]) -> "TensorLayout":
        parameters = dict(model.named_parameters())
        ordered = sorted(names)
        return cls(
            names=ordered,
            shapes=[tuple(parameters[name].shape) for name in ordered],
            numels=[parameters[name].numel() for name in ordered],
        )

    @property
    def total_numel(self) -> int:
        return sum(self.numels)

    def flatten(self, model: nn.Module) -> torch.Tensor:
        parameters = dict(model.named_parameters())
        if not self.names:
            device = next(model.parameters()).device
            return torch.empty(0, device=device, dtype=torch.float32)
        return torch.cat([parameters[name].detach().float().reshape(-1) for name in self.names])

    def assign(self, model: nn.Module, vector: torch.Tensor) -> None:
        if vector.numel() != self.total_numel:
            raise ValueError(f"layout needs {self.total_numel} values, got {vector.numel()}")
        parameters = dict(model.named_parameters())
        offset = 0
        with torch.no_grad():
            for name, shape, count in zip(self.names, self.shapes, self.numels):
                target = parameters[name]
                value = vector[offset : offset + count].reshape(shape).to(target.dtype)
                target.copy_(value)
                offset += count


@dataclass
class ModelBundle:
    model: nn.Module
    adapter_layout: TensorLayout
    head_layout: TensorLayout
    target_names: list[str]
    spectral_seconds: float


def _module_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    path = qualified_name.split(".")
    parent = model
    for part in path[:-1]:
        parent = getattr(parent, part)
    return parent, path[-1]


def _random_a(base: nn.Linear, rank: int, seed: int, orthogonal: bool) -> torch.Tensor:
    if orthogonal:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        raw = torch.randn(base.in_features, rank, generator=generator, device="cpu", dtype=torch.float64)
        q, _ = torch.linalg.qr(raw, mode="reduced")
        return q.T.float().contiguous().to(base.weight.device)
    generator = torch.Generator(device=base.weight.device).manual_seed(seed)
    value = torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32)
    return nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)


def _load_spectral_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return torch.load(path, map_location="cpu", weights_only=True)


def _save_spectral_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(cache, temporary)
    os.replace(temporary, path)


def _canonicalize_svd_signs(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the per-vector SVD sign ambiguity from the right singular rows."""
    left = left.clone()
    right = right.clone()
    for index in range(right.shape[0]):
        pivot = int(right[index].abs().argmax().item())
        if float(right[index, pivot].item()) < 0:
            right[index].neg_()
            left[:, index].neg_()
    return left, right


def _stabilize_svd_basis(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repair small cuSOLVER orthogonality error without changing subspaces."""
    left64 = left.detach().double().cpu()
    right64 = right.detach().double().cpu()

    left_values, left_vectors = torch.linalg.eigh(left64.T @ left64)
    left_inv_sqrt = left_vectors @ torch.diag(left_values.clamp_min(1e-12).rsqrt()) @ left_vectors.T
    left64 = left64 @ left_inv_sqrt

    right_values, right_vectors = torch.linalg.eigh(right64 @ right64.T)
    right_inv_sqrt = right_vectors @ torch.diag(right_values.clamp_min(1e-12).rsqrt()) @ right_vectors.T
    right64 = right_inv_sqrt @ right64

    left32, right32 = _canonicalize_svd_signs(left64.float(), right64.float())
    return left32.to(left.device), right32.to(right.device)


def _sylvester_hadamard_8() -> torch.Tensor:
    """Return the frozen normalized order-8 Sylvester matrix on CPU."""
    value = torch.ones((1, 1), dtype=torch.float64, device="cpu")
    for _ in range(3):
        value = torch.cat(
            (
                torch.cat((value, value), dim=1),
                torch.cat((value, -value), dim=1),
            ),
            dim=0,
        )
    return value / math.sqrt(8.0)


def _rotate_spectral_basis(spectral_a: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | str]]:
    """Apply the preregistered CPU-float64 Hadamard coordinate rotation."""
    if spectral_a.ndim != 2 or spectral_a.shape[0] != 8:
        raise ValueError("spectral_rot_ef requires rank exactly eight")
    source64 = spectral_a.detach().to(device="cpu", dtype=torch.float64).contiguous()
    hadamard = _sylvester_hadamard_8()
    rotated64 = hadamard @ source64
    projector_error = float(
        ((rotated64.T @ rotated64) - (source64.T @ source64)).abs().max().item()
    )
    audit: dict[str, float | str] = {
        "fixed_basis_recipe": "spectral_rot_hadamard8_cpu_float64_v1",
        "spectral_rotation_projector_error": projector_error,
        "spectral_rotation_coordinate_map": "B_rot=B_spectral@H8.T",
    }
    return rotated64.float().contiguous().to(spectral_a.device), audit


def _canonicalize_q_column_signs(value: torch.Tensor) -> torch.Tensor:
    """Canonicalize QR columns by their lowest-index largest-magnitude entry."""
    value = value.clone()
    for column in range(value.shape[1]):
        pivot = int(value[:, column].abs().argmax().item())
        if float(value[pivot, column].item()) < 0:
            value[:, column].neg_()
    return value


def _hybrid4_spectral_basis(
    spectral_a: torch.Tensor,
    basis_seed: int,
    canonical_target_index: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Build the frozen four-spectral/four-complement Gaussian basis."""
    if spectral_a.ndim != 2 or spectral_a.shape[0] != 8:
        raise ValueError("Hybrid4 fixed-A methods require rank exactly eight")

    spectral32 = spectral_a.detach().to(device="cpu", dtype=torch.float32).contiguous()
    leading64 = spectral32[:4].double()
    input_dim = int(spectral32.shape[1])
    derived_seed = int(basis_seed) + 104729 * (int(canonical_target_index) + 1) + 3001
    generator = torch.Generator(device="cpu").manual_seed(derived_seed)
    complement = torch.randn(
        (input_dim, 4), generator=generator, device="cpu", dtype=torch.float64
    )
    # The two projections are intentional and frozen.  They suppress the
    # small residual caused by the stabilized basis having been stored in
    # float32, without changing the leading four rows themselves.
    for _ in range(2):
        complement = complement - leading64.T @ (leading64 @ complement)
    complement, _ = torch.linalg.qr(complement, mode="reduced")
    complement = _canonicalize_q_column_signs(complement)

    hybrid32 = torch.cat((spectral32[:4], complement.T.float()), dim=0).contiguous()
    hybrid64 = hybrid32.double()
    identity = torch.eye(8, dtype=torch.float64, device="cpu")
    audit: dict[str, float | int | str] = {
        "fixed_basis_recipe": "hybrid4_spectral_gaussian_qr_v1",
        "hybrid_basis_seed": derived_seed,
        "hybrid_leading_row_equality_max_error": float(
            (hybrid32[:4] - spectral32[:4]).abs().max().item()
        ),
        "hybrid_mutual_orthogonality_max_error": float(
            (hybrid64[:4] @ hybrid64[4:].T).abs().max().item()
        ),
        "hybrid_row_orthogonality_max_error": float(
            (hybrid64 @ hybrid64.T - identity).abs().max().item()
        ),
    }
    return hybrid32.to(spectral_a.device), audit


def build_model(
    model_path: str,
    num_labels: int,
    method: str,
    rank: int,
    alpha: float,
    dropout: float,
    target_suffixes: list[str],
    basis_seed: int,
    lora_xs_init_std: float,
    spectral_cache_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> ModelBundle:
    from time import perf_counter

    classifier_type = DecoderLastTokenClassifier if is_decoder_model(model_path) else RobertaLinearClassifier
    model = classifier_type(model_path, num_labels, dtype=dtype).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.classifier.float()
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(name.endswith(suffix) for suffix in target_suffixes)
    ]
    if not targets:
        raise RuntimeError(f"no target modules matched suffixes {target_suffixes}")
    canonical_target_indices = {
        name: index for index, name in enumerate(sorted(name for name, _ in targets))
    }

    spectral_cache = _load_spectral_cache(Path(spectral_cache_path)) if (
        method in FIXED_SPECTRAL_METHODS or method == "lora_xs"
    ) else {}
    cache_changed = False
    spectral_seconds = 0.0

    for index, (name, base) in enumerate(targets):
        if min(base.in_features, base.out_features) < rank:
            raise ValueError(f"rank {rank} is too large for {name}")
        replacement: nn.Module
        if method in STANDARD_LORA_METHODS:
            replacement = LoRALinear(base, rank, alpha, dropout, basis_seed + index)
        elif method in FIXED_RANDOM_METHODS:
            fixed_a = _random_a(base, rank, basis_seed + index, orthogonal=method != "ffa_lora")
            replacement = FixedALinear(base, fixed_a, rank, alpha, dropout)
        elif method in FIXED_SPECTRAL_METHODS or method == "lora_xs":
            cached = spectral_cache.get(name)
            weight_for_hash = base.weight.detach().float().cpu().contiguous()
            weight_hash = hashlib.sha256(weight_for_hash.numpy().tobytes()).hexdigest()
            cache_valid = (
                cached is not None
                and cached.get("weight_hash") == weight_hash
                and cached["left"].shape[0] == base.out_features
                and cached["right"].shape[1] == base.in_features
                and cached["right"].shape[0] >= rank
                and cached["singular"].shape[0] >= rank
            )
            if not cache_valid:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = perf_counter()
                weight = base.weight.detach().float()
                left, singular, right = torch.linalg.svd(weight, full_matrices=False)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                spectral_seconds += perf_counter() - started
                cached = {
                    "left": left[:, :rank].cpu(),
                    "singular": singular[:rank].cpu(),
                    "right": right[:rank, :].cpu(),
                    "weight_hash": weight_hash,
                }
                spectral_cache[name] = cached
                cache_changed = True
            assert cached is not None
            left = cached["left"][:, :rank].to(device)
            singular = cached["singular"][:rank].to(device)
            right = cached["right"][:rank, :].to(device)
            # Canonical signs make independently reconstructed bases agree in
            # the usual (non-degenerate) case.  The base-weight hash is stored
            # with the cache and the full basis hash is recorded per run.
            left, right = _stabilize_svd_basis(left.float(), right.float())
            if method == "lora_xs":
                replacement = LoRAXSLinear(
                    base,
                    left,
                    singular,
                    right,
                    rank,
                    alpha,
                    dropout,
                    lora_xs_init_std,
                    basis_seed + index,
                )
            else:
                basis_audit: dict[str, float | int | str]
                if method == "spectral_rot_ef":
                    fixed_a, basis_audit = _rotate_spectral_basis(right)
                elif method in HYBRID4_METHODS:
                    fixed_a, basis_audit = _hybrid4_spectral_basis(
                        right,
                        basis_seed,
                        canonical_target_indices[name],
                    )
                else:
                    # P2 deliberately uses exactly the original stabilized
                    # spectral bytes.  Its block balancing occurs only in the
                    # codec and changes neither the basis nor packet schema.
                    fixed_a = right
                    basis_audit = {
                        "fixed_basis_recipe": "stabilized_spectral_svd_v1",
                    }
                replacement = FixedALinear(
                    base,
                    fixed_a,
                    rank,
                    alpha,
                    dropout,
                    basis_audit=basis_audit,
                )
        elif method == "fedsb":
            # Placeholder; the data-dependent bases and the budget-matched rank are
            # installed by fedpeft.baselines.install_fedsb before training.
            replacement = LoRAXSLinear(
                base,
                torch.zeros(base.out_features, rank, device=base.weight.device),
                torch.ones(rank, device=base.weight.device),
                torch.zeros(rank, base.in_features, device=base.weight.device),
                rank,
                float(rank),
                0.0,
                0.0,
                basis_seed + index,
            )
        else:
            raise ValueError(f"unsupported method: {method}")
        parent, child = _module_parent(model, name)
        setattr(parent, child, replacement)

    if cache_changed:
        _save_spectral_cache(Path(spectral_cache_path), spectral_cache)

    adapter_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and any(marker in name for marker in ("lora_A", "lora_B", "lora_R"))
    ]
    head_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad and name.startswith("classifier.")
    ]
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in set(adapter_names + head_names)
    ]
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters: {unexpected}")
    return ModelBundle(
        model=model,
        adapter_layout=TensorLayout.from_model(model, adapter_names),
        head_layout=TensorLayout.from_model(model, head_names),
        target_names=[name for name, _ in targets],
        spectral_seconds=spectral_seconds,
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def fixed_basis_numel(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, FixedALinear):
            total += module.fixed_A.numel()
        elif isinstance(module, LoRAXSLinear):
            total += module.fixed_left.numel()
            total += module.fixed_singular.numel()
            total += module.fixed_right.numel()
    return total


def full_lora_reference_numel(model: nn.Module, rank: int) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, (LoRALinear, FixedALinear, LoRAXSLinear)):
            total += rank * (module.base.in_features + module.base.out_features)
    return total


def model_invariants(model: nn.Module, method: str) -> dict[str, float | int | str | bool]:
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    illegal = [
        name
        for name, _ in trainable
        if not name.startswith("classifier.")
        and not any(marker in name for marker in ("lora_A", "lora_B", "lora_R"))
    ]
    fixed_a_errors = []
    basis_recipes: set[str] = set()
    rotation_projector_errors: list[float] = []
    hybrid_leading_errors: list[float] = []
    hybrid_mutual_errors: list[float] = []
    hybrid_row_errors: list[float] = []
    hybrid_basis_seeds: list[int] = []
    digest = hashlib.sha256()
    for name, module in model.named_modules():
        if isinstance(module, FixedALinear):
            value = module.fixed_A.detach().float()
            digest.update(name.encode())
            digest.update(value.cpu().contiguous().numpy().tobytes())
            if method not in NON_ORTHONORMAL_FIXED_A_METHODS and method not in WHITENED_METHODS:
                # Activation-whitened bases are deliberately not row-orthonormal.
                # Audit in CPU float64 so CUDA TF32 acceleration does not make
                # a valid orthonormal basis fail the invariant check.
                audit_value = value.double().cpu()
                identity = torch.eye(audit_value.shape[0], dtype=torch.float64)
                fixed_a_errors.append(float((audit_value @ audit_value.T - identity).abs().max().item()))
            audit = module.basis_audit
            if "fixed_basis_recipe" in audit:
                basis_recipes.add(str(audit["fixed_basis_recipe"]))
            if "spectral_rotation_projector_error" in audit:
                rotation_projector_errors.append(float(audit["spectral_rotation_projector_error"]))
            if "hybrid_leading_row_equality_max_error" in audit:
                hybrid_leading_errors.append(float(audit["hybrid_leading_row_equality_max_error"]))
            if "hybrid_mutual_orthogonality_max_error" in audit:
                hybrid_mutual_errors.append(float(audit["hybrid_mutual_orthogonality_max_error"]))
            if "hybrid_row_orthogonality_max_error" in audit:
                hybrid_row_errors.append(float(audit["hybrid_row_orthogonality_max_error"]))
            if "hybrid_basis_seed" in audit:
                hybrid_basis_seeds.append(int(audit["hybrid_basis_seed"]))
        elif isinstance(module, LoRAXSLinear):
            for label, value in (
                ("left", module.fixed_left),
                ("singular", module.fixed_singular),
                ("right", module.fixed_right),
            ):
                digest.update(f"{name}:{label}".encode())
                digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return {
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "unexpected_trainable_count": len(illegal),
        "all_base_parameters_frozen": not illegal,
        "fixed_a_orthogonality_max_error": max(fixed_a_errors, default=0.0),
        "fixed_basis_hash": digest.hexdigest(),
        "fixed_basis_recipe": ",".join(sorted(basis_recipes)) if basis_recipes else "none",
        "spectral_rotation_projector_error_max": max(rotation_projector_errors, default=0.0),
        "hybrid_leading_row_equality_max_error": max(hybrid_leading_errors, default=0.0),
        "hybrid_mutual_orthogonality_max_error": max(hybrid_mutual_errors, default=0.0),
        "hybrid_row_orthogonality_max_error": max(hybrid_row_errors, default=0.0),
        "hybrid_basis_seed_min": min(hybrid_basis_seeds, default=0),
        "hybrid_basis_seed_max": max(hybrid_basis_seeds, default=0),
    }
