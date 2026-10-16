"""Configs for the Web-service tasks (banking77, moderation) on the RoBERTa confirmation protocol.

dev    One config per learning rate in {3e-4, 1e-3, 3e-3}: FedIT, FFA-LoRA, FoCuS (gev0), RoLoRA and FedSVD
       uncompressed, Fed-SB at the 10% budget (train-holdout, seed 42), the rule used for the GLUE tasks.
dev2   Reads the first sweep and writes one extension config per task and rate for every pick on the grid
       boundary (1e-4 below, 1e-2 above).  Prints NONE when no pick is on a boundary.
main   Reads both sweeps, picks each method's rate per task (best mean of the last three evaluations;
       compressed variants reuse the rate of their uncompressed counterpart, the gradient-energy basis
       the random basis's), and writes the confirmation config: official held-out sets, seeds
       1001-1003, budgets 5% and 10%.

Usage:  python scripts/make_web_configs.py dev
        python scripts/make_web_configs.py dev2 | main <dev results root>   (on the server)
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVER = "/data/home/202412012250/bishe_web"
TASKS = ("banking77", "moderation")
SCHEDULE = {  # rounds, local steps, batch size, evaluation interval
    "banking77": (30, 8, 64, 2),
    "moderation": (30, 16, 128, 3),
}
GRID = ("3e-4", "1e-3", "3e-3")
SECOND_EXTENSION = {("3e-2", "banking77"): ("fedsvd_dense",)}
BELOW, ABOVE = "1e-4", "1e-2"
DENSE = ("fedavg_lora", "random_a_dense", "gev0_a_dense", "rolora_dense", "fedsvd_dense")
SWEPT = (*DENSE, "fedsb")
RATE_SOURCE = {  # confirmation method -> swept method whose rate it uses
    "fedavg_lora": "fedavg_lora",
    "full_lora_ef": "fedavg_lora",
    "random_a_ef": "random_a_dense",
    "cab0_a_ef": "random_a_dense",
    "gev0_a_ef": "gev0_a_dense",
    "rolora_ef": "rolora_dense",
    "fedsvd_ef": "fedsvd_dense",
    "fedsb": "fedsb",
}
SPARSE = ("gev0_a_ef", "random_a_ef", "cab0_a_ef", "full_lora_ef", "rolora_ef", "fedsvd_ef", "fedsb")


def base_config(tasks=TASKS) -> dict:
    c = yaml.safe_load((ROOT / "configs" / "roberta_normfix_main.yaml").read_text())
    c["project"]["root"] = SERVER
    c["project"]["spectral_cache"] = "/data/home/202412012250/bishe_dcef/artifacts/spectral/roberta-base-qv-r8-fp32.pt"
    c["data"]["tasks"] = list(tasks)
    for key, position in (("rounds", 0), ("local_steps", 1), ("batch_size", 2), ("eval_every", 3)):
        c["training"][key] = {t: SCHEDULE[t][position] for t in tasks}
    c["cab"]["gamma"] = {t: 0.1 for t in tasks}
    c["baselines"] = {"fedsb_gradient_examples": 1024, "fedsb_gradient_batch_size": 64}
    # Banking77's 77-class head alone exceeds a 10% message budget: the ratio applies to the adapter.
    c["communication"]["budget_scope"] = "adapter"
    return c


def rates(values: dict[str, float]) -> dict:
    return {"learning_rate": dict(values), "head_learning_rate": dict(values)}


def write(name: str, c: dict, note: str) -> None:
    (ROOT / "configs" / f"{name}.yaml").write_text(f"# {note}\n" + yaml.safe_dump(c, sort_keys=False))
    print("wrote", name)


def sweep_config(tag: str, methods, tasks, output: str) -> dict:
    c = base_config(tasks)
    rate = float(tag)
    c["project"]["phase"] = "tune"
    c["project"]["output_dir"] = f"{SERVER}/docs/results/{output}"
    c["evaluation"] = {"policy": "train_holdout"}
    c["data"]["partition_seeds"] = [42]
    c["training"]["learning_rate"] = rate
    c["training"]["head_learning_rate"] = rate
    c["training"]["method_overrides"] = {m: rates({t: rate for t in tasks}) for m in methods}
    c["communication"]["wire_ratios"] = [0.1]
    c["matrix"]["dense_methods"] = [m for m in methods if m != "fedsb"]
    c["matrix"]["sparse_methods"] = [m for m in methods if m == "fedsb"]
    c["project"]["expected_runs"] = len(tasks) * len(methods)
    return c


def dev() -> None:
    for tag in GRID:
        write(f"roberta_web_dev_lr_{tag}", sweep_config(tag, SWEPT, TASKS, f"web_dev_lr_{tag}"),
              f"Web-task learning-rate sweep, lr {tag} (train-holdout, seed 42; Fed-SB at the 10% budget).")


def last3(path: Path) -> float | None:
    summary = path / "summary.json"
    if not summary.exists():
        return None
    data = json.loads(summary.read_text())
    if data.get("status") != "COMPLETE":
        return None
    curve = [r["primary_metric"] for r in data["rounds"] if r.get("primary_metric") is not None]
    return sum(curve[-3:]) / len(curve[-3:])


def run_name(task: str, method: str) -> str:
    return f"{task}__{method}__seed-42__{'wire-10' if method == 'fedsb' else 'dense'}"


def scores(dev_root: Path, extended: bool) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for tag in GRID:
        for method in SWEPT:
            for task in TASKS:
                value = last3(dev_root / f"web_dev_lr_{tag}" / run_name(task, method))
                if value is None:
                    raise SystemExit(f"missing dev run: {tag} {task} {method}")
                result.setdefault(method, {}).setdefault(task, {})[tag] = value
    if extended:
        # second extension, run by hand: FedSVD on banking77 picked 1e-2 (3e-2 collapses, 1e-2 stands)
        for (tag, task), methods in SECOND_EXTENSION.items():
            for method in methods:
                value = last3(dev_root / f"web_dev3_lr_{tag}_{task}" / run_name(task, method))
                if value is not None:
                    result[method][task][tag] = value
        for tag in (BELOW, ABOVE):
            for task in TASKS:
                for method in SWEPT:
                    value = last3(dev_root / f"web_dev2_lr_{tag}_{task}" / run_name(task, method))
                    if value is not None:
                        result[method][task][tag] = value
    return result


def boundary(picks: dict[str, dict[str, str]]) -> dict[tuple[str, str], list[str]]:
    need: dict[tuple[str, str], list[str]] = {}
    for method, by_task in picks.items():
        for task, tag in by_task.items():
            if tag in (GRID[0], GRID[-1]):
                need.setdefault((BELOW if tag == GRID[0] else ABOVE, task), []).append(method)
    return need


def dev2(dev_root: Path) -> None:
    s = scores(dev_root, extended=False)
    picks = {m: {t: max(v, key=v.get) for t, v in bt.items()} for m, bt in s.items()}
    need = boundary(picks)
    print(json.dumps({"picks": picks, "extend": {f"{k[0]} {k[1]}": v for k, v in need.items()}}, indent=1))
    if not need:
        print("NONE")
        return
    for (tag, task), methods in need.items():
        write(f"roberta_web_dev2_lr_{tag}_{task}", sweep_config(tag, methods, (task,), f"web_dev2_lr_{tag}_{task}"),
              f"Web-task sweep extension, lr {tag} on {task}: {', '.join(methods)} picked a grid boundary.")


def main_config(dev_root: Path) -> None:
    s = scores(dev_root, extended=True)
    picks = {m: {t: max(v, key=v.get) for t, v in bt.items()} for m, bt in s.items()}
    record = {"scores": s, "chosen": picks}
    print(json.dumps(record, indent=1))
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "web_lr_choice.json").write_text(json.dumps(record, indent=1))
    c = base_config()
    c["project"]["output_dir"] = f"{SERVER}/docs/results/web_main"
    c["training"]["method_overrides"] = {
        m: rates({t: float(picks[source][t]) for t in TASKS}) for m, source in RATE_SOURCE.items()
    }
    c["communication"]["wire_ratios"] = [0.05, 0.1]
    c["matrix"]["dense_methods"] = ["fedavg_lora"]
    c["matrix"]["sparse_methods"] = list(SPARSE)
    c["project"]["expected_runs"] = len(TASKS) * 3 * (1 + 2 * len(SPARSE))
    write("roberta_web_main", c, "Web-service tasks, confirmation protocol (official held-out sets, seeds 1001-1003, "
          f"5%/10%), each method at its swept rate: {json.dumps(picks, sort_keys=True)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["dev"]:
        dev()
    elif sys.argv[1:2] == ["dev2"] and len(sys.argv) == 3:
        dev2(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["main"] and len(sys.argv) == 3:
        main_config(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
