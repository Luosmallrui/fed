"""Why does the kappa=0 CAB basis (top eigenvectors of Gbar^T Gbar) stall at
majority-class accuracy while a random orthonormal A trains normally?

Hypothesis: RoBERTa hidden states are anisotropic (a large common mean
direction shared by every token).  The initial gradient of a target weight,
G = sum_t delta_t x_t^T, then has its row space dominated by that mean
direction, so a basis built from initial gradients projects mostly onto a
component that is nearly identical for every token and example.  The adapter
output B A x is then close to a constant shift: it can move the class prior
but carries little discriminative signal.

For every target layer and basis A (orthonormal rows) this script measures,
from the frozen model's input activations x to that layer:
  common_mode   ||A xbar||^2 / E||A x||^2      (share of projected energy that
                                               is the same for every token)
  mean_capture  ||A xbar||^2 / ||xbar||^2      (random ~ r/d)
  energy_gain   (E||A x||^2 / E||x||^2) / (r/d)
  var_gain      (tr(A C A^T) / tr C) / (r/d)  C = token covariance
  fisher        tr(A Sb A^T) / tr(A Sw A^T)   class separation of mean-pooled
                                               example features
  overlap_*     subspace overlap with the top-r eigenvectors of E[x x^T]
                (uncentred) and of C (centred)

Usage (server):
  PYTHONPATH=src python scripts/proposals/cab_basis_anisotropy_check.py \
      --config configs/roberta_cab_ef_dev.yaml --task mnli --seed 361 \
      --basis cab0=docs/results/cab_ef_dev/mnli__cab0_a_ef__seed-361__wire-05/cab_basis.pt \
      --basis cab_05=docs/results/cab_ef_dev/mnli__cab_a_ef__seed-361__wire-05/cab_basis.pt
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from fedpeft.adapters import FixedALinear, build_model
from fedpeft.cab import subspace_overlap, top_eigen_rows
from fedpeft.config import load_config
from fedpeft.data import load_task_to_device, prepare_tokenized_dataset

TASK_NUM_LABELS = {"sst2": 2, "rte": 2, "mnli": 3}


def build(cfg: dict, task: str, method: str, device: torch.device):
    return build_model(
        model_path=cfg["model"]["path"],
        num_labels=TASK_NUM_LABELS[task],
        method=method,
        rank=int(cfg["model"]["rank"]),
        alpha=float(cfg["model"]["alpha"]),
        dropout=float(cfg["model"]["dropout"]),
        target_suffixes=list(cfg["model"]["target_suffixes"]),
        basis_seed=int(cfg["model"]["basis_seed"]),
        lora_xs_init_std=float(cfg["model"]["lora_xs_init_std"]),
        spectral_cache_path=cfg["project"]["spectral_cache"],
        device=device,
        dtype=torch.float32,
    ).model


def fixed_targets(model: torch.nn.Module) -> list[tuple[str, FixedALinear]]:
    return sorted(
        ((name, module) for name, module in model.named_modules() if isinstance(module, FixedALinear)),
        key=lambda item: item[0],
    )


class InputMoments:
    """Token-level first/second moments and per-class pooled-example moments."""

    def __init__(self, dim: int, num_labels: int, device: torch.device):
        options = {"dtype": torch.float64, "device": device}
        self.tokens = 0
        self.sum = torch.zeros(dim, **options)
        self.gram = torch.zeros(dim, dim, **options)
        self.class_count = torch.zeros(num_labels, **options)
        self.class_sum = torch.zeros(num_labels, dim, **options)
        self.pooled_gram = torch.zeros(dim, dim, **options)
        self.mask: torch.Tensor | None = None
        self.labels: torch.Tensor | None = None

    def hook(self, _module, inputs):
        x = inputs[0].detach().double()
        mask = self.mask.to(x.dtype).unsqueeze(-1)
        flat = (x * mask).reshape(-1, x.shape[-1])
        self.tokens += int(self.mask.sum().item())
        self.sum += flat.sum(0)
        self.gram += flat.T @ flat
        pooled = (x * mask).sum(1) / mask.sum(1)
        self.pooled_gram += pooled.T @ pooled
        self.class_count.index_add_(0, self.labels, torch.ones_like(self.labels, dtype=torch.float64))
        self.class_sum.index_add_(0, self.labels, pooled)

    def summary(self) -> dict[str, torch.Tensor]:
        mean = self.sum / self.tokens
        second = self.gram / self.tokens
        cov = second - torch.outer(mean, mean)
        n = self.class_count.sum()
        pooled_mean = self.class_sum.sum(0) / n
        class_means = self.class_sum / self.class_count[:, None].clamp_min(1)
        diff = class_means - pooled_mean
        between = (diff.T * self.class_count) @ diff / n
        total = self.pooled_gram / n - torch.outer(pooled_mean, pooled_mean)
        within = total - between
        return {"mean": mean, "second": second, "cov": cov, "between": between, "within": within}


def measure(rows: torch.Tensor, m: dict[str, torch.Tensor], rank: int) -> dict[str, float]:
    a = rows.double().to(m["mean"].device)
    d = a.shape[1]
    base = rank / d
    proj_energy = float(torch.trace(a @ m["second"] @ a.T))
    proj_mean = float((a @ m["mean"]).pow(2).sum())
    return {
        "common_mode": proj_mean / proj_energy,
        "mean_capture": proj_mean / float(m["mean"].pow(2).sum()),
        "energy_gain": proj_energy / float(torch.trace(m["second"])) / base,
        "var_gain": float(torch.trace(a @ m["cov"] @ a.T)) / float(torch.trace(m["cov"])) / base,
        "fisher": float(torch.trace(a @ m["between"] @ a.T)) / float(torch.trace(a @ m["within"] @ a.T)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", default="mnli")
    parser.add_argument("--seed", type=int, default=361)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--basis", action="append", default=[], help="name=path/to/cab_basis.pt")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"], local_files_only=True, use_fast=True)
    tokenized = prepare_tokenized_dataset(
        cfg["data"]["root"],
        cfg["project"]["tokenized_dir"],
        args.task,
        tokenizer,
        int(cfg["data"]["max_length"]),
        int(cfg["data"]["preprocessing_workers"]),
    )
    train, _ = load_task_to_device(tokenized, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    sample = torch.randperm(train.size, generator=generator)[: args.examples].to(device)

    random_model = build(cfg, args.task, "random_a_ef", device)
    spectral_model = build(cfg, args.task, "spectral_a_ef", device)
    targets = fixed_targets(random_model)
    names = [name for name, _ in targets]
    rank = int(targets[0][1].rank)
    bases: dict[str, list[torch.Tensor]] = {
        "random": [module.fixed_A.detach().cpu() for _, module in targets],
        "spectral": [module.fixed_A.detach().cpu() for _, module in fixed_targets(spectral_model)],
    }
    del spectral_model
    record_energy = {}
    for item in args.basis:
        label, path = item.split("=", 1)
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached["target_names"] != names:
            raise RuntimeError(f"{path}: target names differ from the random model")
        bases[label] = [rows.float() for rows in cached["bases"]]
        record_energy[label] = {
            "kappa": cached["kappa"],
            "reported_energy_fraction_by_client": cached["record"].get("reported_energy_fraction_by_client"),
        }

    moments = [InputMoments(module.base.in_features, TASK_NUM_LABELS[args.task], device) for _, module in targets]
    handles = [module.register_forward_pre_hook(stat.hook) for (_, module), stat in zip(targets, moments)]
    random_model.eval()
    with torch.no_grad():
        for start in range(0, sample.numel(), args.batch_size):
            batch = train.batch(sample[start : start + args.batch_size])
            labels = batch.pop("labels")
            for stat in moments:
                stat.mask, stat.labels = batch["attention_mask"], labels
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                random_model(**batch)
    for handle in handles:
        handle.remove()

    per_layer = []
    for index, (name, stat) in enumerate(zip(names, moments)):
        m = stat.summary()
        top_second, _ = top_eigen_rows(m["second"].cpu(), rank)
        top_cov, _ = top_eigen_rows(m["cov"].cpu(), rank)
        layer = {
            "name": name,
            "input_common_mode": float(m["mean"].pow(2).sum() / torch.trace(m["second"])),
        }
        for label, rows in bases.items():
            values = measure(rows[index], m, rank)
            values["overlap_top_second"] = subspace_overlap(rows[index].double(), top_second)
            values["overlap_top_cov"] = subspace_overlap(rows[index].double(), top_cov)
            layer[label] = values
        per_layer.append(layer)

    keys = ["common_mode", "mean_capture", "energy_gain", "var_gain", "fisher", "overlap_top_second", "overlap_top_cov"]
    print(f"task={args.task} examples={args.examples} layers={len(names)} rank={rank} d={targets[0][1].base.in_features}")
    print(
        "input common-mode share E: mean %.3f  (min %.3f, max %.3f)"
        % (
            statistics.fmean(layer["input_common_mode"] for layer in per_layer),
            min(layer["input_common_mode"] for layer in per_layer),
            max(layer["input_common_mode"] for layer in per_layer),
        )
    )
    print("\nmedian over layers (random expectation: common_mode ~ input share, gains 1, mean_capture ~ r/d)")
    print(f"{'basis':10s} " + " ".join(f"{k:>18s}" for k in keys))
    medians = {}
    for label in bases:
        medians[label] = {k: float(np.median([layer[label][k] for layer in per_layer])) for k in keys}
        print(f"{label:10s} " + " ".join(f"{medians[label][k]:18.4f}" for k in keys))
    for label, info in record_energy.items():
        energy = info["reported_energy_fraction_by_client"]
        if energy:
            print(f"{label}: kappa={info['kappa']:.4f}; rank-r reported gradient energy by client min={min(energy):.6f}")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"medians": medians, "per_layer": per_layer, "records": record_energy}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
