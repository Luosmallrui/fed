"""Conditioning of frozen bases: the quantities behind the bias-channel analysis (Section 5.2).

For every target layer and basis A, with Sigma = E[x x^T] = mu mu^T + C of the layer inputs:
  kappa        lambda_max / lambda_min of K = A Sigma A^T (the input factor of the B-Hessian)
  bound        rho ||mu||^2 / lambda_max(C), the lower bound on kappa of Lemma 2, with rho the
               fraction of the mean direction inside row(A)
  top_common   share of the top eigen-direction's input energy that is common to all tokens,
               ((A^T w)^T mu)^2 / (A^T w)^T Sigma (A^T w) for the top eigenvector w of K
Bases: random and spectral (built), plus any cached CAB-protocol bases (name=path/to/cab_basis.pt).

Usage (server):
  PYTHONPATH=src python scripts/proposals/basis_conditioning_check.py --config C --task mnli \
      --basis grad-energy=.../mnli__cab0_a_ef__seed-1001__wire-10/cab_basis.pt ... --out out.json
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
from cab_basis_anisotropy_check import TASK_NUM_LABELS, InputMoments, build, fixed_targets  # noqa: E402

from fedpeft.config import load_config  # noqa: E402
from fedpeft.data import load_task_to_device, prepare_tokenized_dataset  # noqa: E402


def conditioning(rows: torch.Tensor, m: dict[str, torch.Tensor]) -> dict[str, float]:
    a = rows.double().to(m["mean"].device)
    k = a @ m["second"] @ a.T
    values, vectors = torch.linalg.eigh(0.5 * (k + k.T))
    q, _ = torch.linalg.qr(a.T)  # orthonormal basis of row(A)
    mu = m["mean"]
    rho = float((q.T @ mu).pow(2).sum() / mu.pow(2).sum())
    lam_c = float(torch.linalg.eigvalsh(m["cov"]).max())
    top = a.T @ vectors[:, -1]
    return {
        "kappa": float(values[-1] / values[0]),
        "bound": rho * float(mu.pow(2).sum()) / lam_c,
        "rho": rho,
        "top_common": float((top @ mu) ** 2 / (top @ m["second"] @ top)),
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
    tokenized = prepare_tokenized_dataset(cfg["data"]["root"], cfg["project"]["tokenized_dir"], args.task, tokenizer,
                                          int(cfg["data"]["max_length"]), int(cfg["data"]["preprocessing_workers"]))
    train, _ = load_task_to_device(tokenized, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    sample = torch.randperm(train.size, generator=generator)[: args.examples].to(device)

    model = build(cfg, args.task, "random_a_ef", device)
    targets = fixed_targets(model)
    names = [name for name, _ in targets]
    bases = {"random": [module.fixed_A.detach().cpu() for _, module in targets]}
    spectral = build(cfg, args.task, "spectral_a_ef", device)
    bases["spectral"] = [module.fixed_A.detach().cpu() for _, module in fixed_targets(spectral)]
    del spectral
    for item in args.basis:
        label, path = item.split("=", 1)
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached["target_names"] != names:
            raise RuntimeError(f"{path}: target names differ")
        bases[label] = [rows.float() for rows in cached["bases"]]

    moments = [InputMoments(module.base.in_features, TASK_NUM_LABELS[args.task], device) for _, module in targets]
    handles = [module.register_forward_pre_hook(stat.hook) for (_, module), stat in zip(targets, moments)]
    model.eval()
    with torch.no_grad():
        for start in range(0, sample.numel(), args.batch_size):
            batch = train.batch(sample[start : start + args.batch_size])
            labels = batch.pop("labels")
            for stat in moments:
                stat.mask, stat.labels = batch["attention_mask"], labels
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(**batch)
    for handle in handles:
        handle.remove()

    per_layer = []
    for index, (name, stat) in enumerate(zip(names, moments)):
        m = stat.summary()
        per_layer.append({"name": name, **{label: conditioning(rows[index], m) for label, rows in bases.items()}})
    medians = {label: {k: float(np.median([layer[label][k] for layer in per_layer])) for k in ("kappa", "bound", "rho", "top_common")}
               for label in bases}
    print(f"task={args.task} examples={args.examples} layers={len(names)}")
    for label, med in medians.items():
        print(f"{label:22s} kappa {med['kappa']:10.1f}  bound {med['bound']:10.1f}  rho {med['rho']:.3f}  top_common {med['top_common']:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"medians": medians, "per_layer": per_layer}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
