"""Subspace versus conditioning: why does normalization fix some frozen bases and not others?

Extends basis_conditioning_check.py with the pooled initial gradient Gbar of every target layer.  For each
basis A (and Sigma = E[x x^T] = mu mu^T + C of the layer inputs, ridge eps tr(Sigma)/d):
  kappa, rho, top_common   as in basis_conditioning_check.py (conditioning; changed by normalization)
  capture     tr(Gbar^T Gbar A^T (A S A^T)^{-1} A) / (top-r generalized eigenvalues of (Gbar^T Gbar, S)):
              the fraction of the optimal one-step K-FAC decrease the row space attains (subspace quality;
              invariant to normalization, 1 for ours with exact statistics)
  bias_share  share of the whitened gradient Gbar A^T K^{-1/2} that lies along the whitened mean
              K^{-1/2} A mu (how much of the captured signal is the bias channel; also invariant)
Bases: random and spectral (built) and cached CAB-protocol bases (name=path/to/cab_basis.pt).  Fed-SB is not
supported: it starts from a non-zero update, so its layer inputs are not those of the base model measured here.

Usage (server):
  PYTHONPATH=src python scripts/proposals/basis_quality_check.py --config C --task sst2 \
      --basis eva=.../sst2__eva_a_ef__seed-1001__wire-05/cab_basis.pt --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cab_basis_anisotropy_check as helpers  # noqa: E402
from basis_conditioning_check import conditioning  # noqa: E402
from cab_basis_anisotropy_check import InputMoments, build, fixed_targets  # noqa: E402

from fedpeft.cab import _inverse_sqrt, _regularised, generalized_top_rows, kfac_capture  # noqa: E402
from fedpeft.config import load_config  # noqa: E402
from fedpeft.data import load_task_to_device, prepare_tokenized_dataset  # noqa: E402

NUM_LABELS = {"sst2": 2, "mnli": 3, "qnli": 2}
helpers.TASK_NUM_LABELS.update(NUM_LABELS)  # build() looks the head size up there
RIDGE = 1e-3


def bias_share(rows: torch.Tensor, gbar: torch.Tensor, m: dict[str, torch.Tensor]) -> float:
    a = rows.double().cpu()
    root = _inverse_sqrt(a @ _regularised(m["second"], RIDGE) @ a.T)
    whitened_grad = gbar.double().cpu() @ a.T @ root
    mean = root @ a @ m["mean"].double().cpu()
    direction = mean / mean.norm()
    return float((whitened_grad @ direction).pow(2).sum() / whitened_grad.pow(2).sum())


def load_bases(spec: str, names: list[str]) -> list[torch.Tensor]:
    cached = torch.load(spec, map_location="cpu", weights_only=False)
    if cached["target_names"] != names:
        raise RuntimeError(f"{spec}: target names differ")
    return [rows.double() for rows in cached["bases"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--seed", type=int, default=361)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--basis", action="append", default=[], help="name=path/to/cab_basis.pt")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    torch.set_num_threads(8)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"], local_files_only=True, use_fast=True)
    tokenized = prepare_tokenized_dataset(cfg["data"]["root"], cfg["project"]["tokenized_dir"], args.task, tokenizer,
                                          int(cfg["data"]["max_length"]), int(cfg["data"]["preprocessing_workers"]))
    train, _ = load_task_to_device(tokenized, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    sample = torch.randperm(train.size, generator=generator)[: args.examples].to(device)

    model = build(cfg, args.task, "random_a_ef", device)
    targets = fixed_targets(model)
    names = [name for name, _ in targets]
    bases = {"random": [module.fixed_A.detach().double().cpu() for _, module in targets]}
    spectral = build(cfg, args.task, "spectral_a_ef", device)
    bases["spectral"] = [module.fixed_A.detach().double().cpu() for _, module in fixed_targets(spectral)]
    del spectral
    for item in args.basis:
        label, spec = item.split("=", 1)
        bases[label] = load_bases(spec, names)

    moments = [InputMoments(module.base.in_features, NUM_LABELS[args.task], device) for _, module in targets]
    handles = [module.register_forward_pre_hook(stat.hook) for (_, module), stat in zip(targets, moments)]
    weights = [module.base.weight for _, module in targets]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for weight in weights:
        weight.requires_grad_(True)
    model.eval()
    sums = [torch.zeros_like(weight, dtype=torch.float64) for weight in weights]
    for start in range(0, sample.numel(), args.batch_size):
        positions = sample[start : start + args.batch_size]
        batch = train.batch(positions)
        for stat in moments:
            stat.mask, stat.labels = batch["attention_mask"], batch["labels"]
        loss = model(**batch).loss
        for total, grad in zip(sums, torch.autograd.grad(loss, weights)):
            total += grad.double() * positions.numel()
    for handle in handles:
        handle.remove()
    gbars = [total.cpu() / sample.numel() for total in sums]

    per_layer = []
    for index, (name, stat) in enumerate(zip(names, moments)):
        m = {k: v.cpu() for k, v in stat.summary().items()}
        gram = gbars[index].T @ gbars[index]
        _, values = generalized_top_rows(gram, m["second"], max(int(rows[index].shape[0]) for rows in bases.values()), RIDGE)
        layer = {"name": name}
        for label, rows in bases.items():
            entry = conditioning(rows[index].float(), m)
            best = float(values[: rows[index].shape[0]].clamp_min(0.0).sum())
            entry["capture"] = kfac_capture(rows[index], gram, m["second"], RIDGE, best=best)
            entry["bias_share"] = bias_share(rows[index], gbars[index], m)
            entry["rank"] = int(rows[index].shape[0])
            layer[label] = entry
        per_layer.append(layer)
    keys = ("kappa", "rho", "top_common", "capture", "bias_share")
    medians = {label: {k: float(np.median([layer[label][k] for layer in per_layer])) for k in keys} for label in bases}
    print(f"task={args.task} examples={args.examples} layers={len(names)}")
    for label, med in medians.items():
        print(f"{label:18s} rank {per_layer[0][label]['rank']:3d}  kappa {med['kappa']:9.1f}  rho {med['rho']:.3f}  "
              f"top_common {med['top_common']:.3f}  capture {med['capture']:.3f}  bias_share {med['bias_share']:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"task": args.task, "medians": medians, "per_layer": per_layer}, indent=2),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
