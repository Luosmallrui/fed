"""Zero-training check: does the RoBERTa failure mode exist in decoder LLMs?

On RoBERTa the q/v inputs share a large mean direction (about 66% of the second
moment), so bases built from gradient energy (cab0, LoRA-GA-like) or activation
variance (EVA) capture it, the projected input is a near-constant "bias channel",
and federated training stalls at the majority class.  This script measures the same
quantities on a causal LM without any training:

  input_common     ||xbar||^2 / E||x||^2 of the q/v input (token level)
  top1_share       largest eigenvalue of E[x x^T] / its trace
  grad_rank_r      share of the pooled gradient energy in its top-r singular values
  per basis (random, cab0 = top-r eigvecs of Gbar^T Gbar, eva = top-r eigvecs of
  E[x x^T], gev0 = generalized eigvecs of (Gbar^T Gbar, E[x x^T])):
    mean_capture   ||P xbar||^2 / ||xbar||^2 (P = projector on the row space)
    energy_ratio   E||A x||^2 / (r tr(S) / d) for row-orthonormal A (gev0: its scaled rows)
    common_mode    ||A xbar||^2 / E||A x||^2

Statistics are reported with and without the first position, because LLM "massive
activations" sit on the first token (attention sink) rather than on every token.
The task loss is the answer token of a GLUE prompt (MNLI: yes / maybe / no;
SST-2: negative / positive), computed only on those tokens.

Usage (server):
  python scripts/proposals/llm_anisotropy_check.py --model <snapshot dir> --task mnli \
      --examples 256 --batch-size 8 --rank 16 --out logs/llm_aniso_qwen7b_mnli.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

GLUE_ROOT = "/data/home/202412012250/bishe_dcef/datasets"
MNLI_ANSWERS = {0: "yes", 1: "maybe", 2: "no"}
SST2_ANSWERS = {0: "negative", 1: "positive"}


def prompts(task: str, examples: int, seed: int) -> list[tuple[str, str]]:
    data = load_from_disk(f"{GLUE_ROOT}/{task}")["train"].shuffle(seed=seed).select(range(examples))
    out = []
    for row in data:
        if task == "mnli":
            text = (
                f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}\n"
                "Does the premise entail the hypothesis? Answer yes, maybe, or no.\nAnswer:"
            )
            out.append((text, " " + MNLI_ANSWERS[int(row["label"])]))
        else:
            text = f"Review: {row['sentence']}\nIs the sentiment negative or positive?\nAnswer:"
            out.append((text, " " + SST2_ANSWERS[int(row["label"])]))
    return out


def encode(tokenizer, batch: list[tuple[str, str]], max_length: int, device) -> dict[str, torch.Tensor]:
    ids, labels = [], []
    for prompt, answer in batch:
        p = tokenizer(prompt, add_special_tokens=False)["input_ids"][-(max_length - 4):]
        a = tokenizer(answer, add_special_tokens=False)["input_ids"]
        ids.append(p + a)
        labels.append([-100] * len(p) + a)
    width = max(len(x) for x in ids)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(ids), width), pad, dtype=torch.long)
    label_ids = torch.full((len(ids), width), -100, dtype=torch.long)
    mask = torch.zeros((len(ids), width), dtype=torch.long)
    for i, (x, y) in enumerate(zip(ids, labels)):
        input_ids[i, : len(x)] = torch.tensor(x)
        label_ids[i, : len(y)] = torch.tensor(y)
        mask[i, : len(x)] = 1
    return {"input_ids": input_ids.to(device), "attention_mask": mask.to(device), "labels": label_ids.to(device)}


class Moments:
    def __init__(self, dim: int, device):
        self.n = {"all": 0, "rest": 0}
        self.s1 = {k: torch.zeros(dim, dtype=torch.float64, device=device) for k in self.n}
        self.s2 = {k: torch.zeros(dim, dim, dtype=torch.float64, device=device) for k in self.n}

    def add(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        x = x.detach().double()
        for key, keep in (("all", mask), ("rest", mask.clone().index_fill_(1, torch.tensor([0], device=mask.device), 0))):
            w = keep.to(x.dtype).unsqueeze(-1)
            flat = (x * w).reshape(-1, x.shape[-1])
            self.n[key] += int(keep.sum().item())
            self.s1[key] += flat.sum(0)
            self.s2[key] += flat.T @ flat

    def get(self, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.s1[key] / self.n[key], self.s2[key] / self.n[key]


def top_rows(sym: torch.Tensor, rank: int) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(0.5 * (sym + sym.T))
    return vectors[:, torch.argsort(values, descending=True)[:rank]].T.contiguous()


def gev_rows(matrix: torch.Tensor, metric: torch.Tensor, rank: int, ridge: float = 1e-3) -> torch.Tensor:
    d = metric.shape[0]
    reg = 0.5 * (metric + metric.T) + ridge * torch.trace(metric) / d * torch.eye(d, dtype=metric.dtype, device=metric.device)
    values, vectors = torch.linalg.eigh(reg)
    inv_sqrt = (vectors * values.clamp_min(1e-300).rsqrt()[None, :]) @ vectors.T
    rows = top_rows(inv_sqrt @ matrix @ inv_sqrt, rank) @ inv_sqrt
    return rows * math.sqrt(float(torch.trace(metric)) / d)


def orthonormal(rows: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(rows.T, mode="reduced")
    return q.T


def basis_metrics(rows: torch.Tensor, mean: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    """Row-space metrics use an orthonormal representative; energy uses the rows as installed."""
    ortho = orthonormal(rows)
    r, d = rows.shape
    energy = float(torch.trace(rows @ second @ rows.T))
    return {
        "mean_capture": float((ortho @ mean).pow(2).sum() / mean.pow(2).sum()),
        "energy_ratio": energy / (r * float(torch.trace(second)) / d),
        "common_mode": float((rows @ mean).pow(2).sum()) / energy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=["mnli", "sst2"], default="mnli")
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = model.model.layers
    targets = []
    for index, layer in enumerate(layers):
        for name in ("q_proj", "v_proj"):
            module = getattr(layer.self_attn, name)
            module.weight.requires_grad_(True)
            targets.append((index, name, module))
    dim = targets[0][2].in_features
    moments = [Moments(dim, device) for _ in layers]
    state: dict[str, torch.Tensor] = {}

    def make_hook(index: int):
        def hook(_module, inputs):
            moments[index].add(inputs[0], state["mask"])
        return hook

    handles = [layer.self_attn.q_proj.register_forward_pre_hook(make_hook(i)) for i, layer in enumerate(layers)]
    grads = [torch.zeros_like(module.weight, dtype=torch.float32) for _, _, module in targets]
    data = prompts(args.task, args.examples, args.seed)
    seen = 0
    for start in range(0, len(data), args.batch_size):
        batch = encode(tokenizer, data[start : start + args.batch_size], args.max_length, device)
        state["mask"] = batch["attention_mask"]
        loss = model(**batch).loss
        weights = [module.weight for _, _, module in targets]
        for total, grad in zip(grads, torch.autograd.grad(loss, weights)):
            total += grad.float() * len(batch["input_ids"])
        seen += len(batch["input_ids"])
    for handle in handles:
        handle.remove()

    per_module = []
    generator = torch.Generator(device="cpu").manual_seed(20260910)
    layer_cache: dict[int, dict[str, object]] = {}
    for (index, name, module), grad in zip(targets, grads):
        g = (grad / seen).double()
        mean_all, second_all = moments[index].get("all")
        mean_rest, second_rest = moments[index].get("rest")
        if index not in layer_cache:  # q and v share one input: compute input-only quantities once
            layer_cache[index] = {
                "top1": float(torch.linalg.eigvalsh(second_all)[-1] / torch.trace(second_all)),
                "top1_rest": float(torch.linalg.eigvalsh(second_rest)[-1] / torch.trace(second_rest)),
                "eva": top_rows(second_all, args.rank),
            }
        cache = layer_cache[index]
        consensus = g.T @ g
        singular = torch.linalg.svdvals(g)
        record = {
            "layer": index,
            "module": name,
            "input_common": float(mean_all.pow(2).sum() / torch.trace(second_all)),
            "input_common_no_first": float(mean_rest.pow(2).sum() / torch.trace(second_rest)),
            "top1_share": cache["top1"],
            "top1_share_no_first": cache["top1_rest"],
            "grad_rank_r": float(singular[: args.rank].pow(2).sum() / singular.pow(2).sum()),
        }
        random_rows = orthonormal(torch.randn(args.rank, dim, generator=generator, dtype=torch.float64).to(device))
        bases = {
            "random": random_rows,
            "cab0": top_rows(consensus, args.rank),
            "eva": cache["eva"],
            "gev0": gev_rows(consensus, second_all, args.rank),
        }
        for label, rows in bases.items():
            record[label] = basis_metrics(rows, mean_all, second_all)
            record[label + "_no_first"] = basis_metrics(rows, mean_rest, second_rest)
        per_module.append(record)

    def median(key: str, sub: str | None = None) -> float:
        values = [m[key] if sub is None else m[key][sub] for m in per_module]
        return statistics.median(values)

    print(f"model={Path(args.model).parts[-3]} task={args.task} examples={seen} rank={args.rank} d={dim} modules={len(per_module)}")
    for key in ("input_common", "input_common_no_first", "top1_share", "top1_share_no_first", "grad_rank_r"):
        values = [m[key] for m in per_module]
        print(f"  {key:22s} median {statistics.median(values):.3f}  min {min(values):.3f}  max {max(values):.3f}")
    print(f"  {'basis':16s} {'mean_capture':>12s} {'energy_ratio':>12s} {'common_mode':>12s}   (median over modules)")
    for label in ("random", "cab0", "eva", "gev0", "random_no_first", "cab0_no_first", "eva_no_first", "gev0_no_first"):
        print(f"  {label:16s} " + " ".join(f"{median(label, k):12.3f}" for k in ("mean_capture", "energy_ratio", "common_mode")))
    Path(args.out).write_text(json.dumps({"args": vars(args), "modules": per_module}, indent=1))


if __name__ == "__main__":
    main()
