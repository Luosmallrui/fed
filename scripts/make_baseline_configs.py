"""Configs for the published federated LoRA baselines on the RoBERTa confirmation protocol.

dev   One config per learning rate (train-holdout, seed 42, same rounds / steps / clients as
      configs/roberta_confirm_main.yaml).  Each dense baseline is swept over {3e-4, 1e-3, 3e-3},
      Fed-SB (10% budget) over {1e-3, 3e-3, 1e-2} (it trains only an r x r matrix).
dev2  Extends the grid wherever the first sweep picked a boundary value: Fed-SB down to
      {1e-4, 3e-4} (SST-2 / MNLI picked 1e-3), FLoRA down to 1e-4 (QNLI picked 3e-4), RoLoRA and
      FedSVD up to 1e-2 on SST-2 (picked 3e-3); then FLoRA down to 3e-5 on QNLI (picked 1e-4).
      Separate configs keep the completed runs valid.
main  Reads the dev results and picks, per task and method, the learning rate with the best
      mean of the last three evaluations; the Top-k/EF variants reuse the rate of their dense
      counterpart, FedIT + Top-k/EF reuses FedAvg-LoRA's confirmed rate.  Official validation,
      seeds 1001-1003, budgets 5/10/20%.  gev0 / FedIT / FFA-LoRA runs are reused from
      confirm_main, so they are not repeated here.

gev0dev   Symmetric check: gev0 had inherited FFA-LoRA's rates, while every baseline was swept.
          Same rule as the baselines: sweep gev0_a_dense over {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
          (train-holdout, seed 42), pick the best last-3 mean, gev0_a_ef shares it.
gev0main  For every task whose pick differs from the confirmation rate, rerun gev0_a_dense and
          gev0_a_ef on the confirmation protocol (configs/roberta_gev0_retuned.yaml), whichever
          way the result goes.  Writes nothing when no rate changes.
gev0repro The confirmation config relocated to this code tree, to check that a gev0 run
          reproduces confirm_main exactly here.

Usage:  python scripts/make_baseline_configs.py dev | dev2 | gev0dev | gev0repro
        python scripts/make_baseline_configs.py main | gev0main <dev results root>   (on the server)
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVER = "/data/home/202412012250/bishe_base"
TASKS = ("sst2", "mnli", "qnli")
DENSE = ("flora_dense", "fedex_dense", "rolora_dense", "fedsvd_dense")
GRID = {"3e-4": DENSE, "1e-3": (*DENSE, "fedsb"), "3e-3": (*DENSE, "fedsb"), "1e-2": ("fedsb",)}
# tag -> (methods, tasks)
GRID2 = {
    "1e-4": (("flora_dense", "fedsb"), TASKS),
    "3e-4": (("fedsb",), TASKS),
    "1e-2": (("rolora_dense", "fedsvd_dense"), ("sst2",)),
    # second extension: FLoRA on QNLI picked 1e-4 after the first one
    "3e-5": (("flora_dense",), ("qnli",)),
}
EF_SHARES = {"rolora_ef": "rolora_dense", "fedsvd_ef": "fedsvd_dense"}


def base_config() -> dict:
    c = yaml.safe_load((ROOT / "configs" / "roberta_confirm_main.yaml").read_text())
    c.pop("cab")
    c["project"]["root"] = SERVER
    c["project"]["spectral_cache"] = f"{SERVER}/artifacts/spectral/roberta-base-qv-r8-fp32.pt"
    c["baselines"] = {"fedsb_gradient_examples": 1024, "fedsb_gradient_batch_size": 64}
    return c


def lr_overrides(rate: float) -> dict:
    values = {task: rate for task in TASKS}
    return {"learning_rate": dict(values), "head_learning_rate": dict(values)}


def write(name: str, c: dict, note: str) -> None:
    (ROOT / "configs" / f"{name}.yaml").write_text(f"# {note}\n" + yaml.safe_dump(c, sort_keys=False))
    print("wrote", name)


def dev() -> None:
    for tag, methods in GRID.items():
        c = base_config()
        rate = float(tag)
        c["project"]["phase"] = "tune"
        c["project"]["output_dir"] = f"{SERVER}/docs/results/baselines_dev_lr_{tag}"
        c["evaluation"] = {"policy": "train_holdout"}
        c["data"]["partition_seeds"] = [42]
        c["training"]["learning_rate"] = rate
        c["training"]["head_learning_rate"] = rate
        c["training"]["method_overrides"] = {m: lr_overrides(rate) for m in methods}
        c["communication"]["wire_ratios"] = [0.1]
        c["matrix"]["dense_methods"] = [m for m in methods if m != "fedsb"]
        c["matrix"]["sparse_methods"] = [m for m in methods if m == "fedsb"]
        c["project"]["expected_runs"] = len(TASKS) * len(methods)
        write(
            f"roberta_baselines_dev_lr_{tag}",
            c,
            f"Baseline learning-rate sweep, lr {tag} (train-holdout, seed 42; Fed-SB at the 10% budget).",
        )


def dev2() -> None:
    for tag, (methods, tasks) in GRID2.items():
        c = base_config()
        rate = float(tag)
        c["project"]["phase"] = "tune"
        c["project"]["output_dir"] = f"{SERVER}/docs/results/baselines_dev2_lr_{tag}"
        c["evaluation"] = {"policy": "train_holdout"}
        c["data"]["tasks"] = list(tasks)
        c["data"]["partition_seeds"] = [42]
        c["training"]["learning_rate"] = rate
        c["training"]["head_learning_rate"] = rate
        c["training"]["method_overrides"] = {m: lr_overrides(rate) for m in methods}
        for key in ("rounds", "local_steps", "batch_size", "eval_every"):
            c["training"][key] = {t: v for t, v in c["training"][key].items() if t in tasks}
        c["communication"]["wire_ratios"] = [0.1]
        c["matrix"]["dense_methods"] = [m for m in methods if m != "fedsb"]
        c["matrix"]["sparse_methods"] = [m for m in methods if m == "fedsb"]
        c["project"]["expected_runs"] = len(tasks) * len(methods)
        write(
            f"roberta_baselines_dev2_lr_{tag}",
            c,
            f"Baseline learning-rate sweep extension, lr {tag}: {', '.join(methods)} on {', '.join(tasks)} "
            "(train-holdout, seed 42; Fed-SB at the 10% budget).",
        )


GEV0_GRID = ("1e-4", "3e-4", "1e-3", "3e-3", "1e-2")


def confirm_config() -> dict:
    c = yaml.safe_load((ROOT / "configs" / "roberta_confirm_main.yaml").read_text())
    c["project"]["root"] = SERVER
    c["project"]["spectral_cache"] = f"{SERVER}/artifacts/spectral/roberta-base-qv-r8-fp32.pt"
    return c


def gev0dev() -> None:
    for tag in GEV0_GRID:
        c = confirm_config()
        rate = float(tag)
        c["project"]["phase"] = "tune"
        c["project"]["output_dir"] = f"{SERVER}/docs/results/gev0_dev_lr_{tag}"
        c["evaluation"] = {"policy": "train_holdout"}
        c["data"]["partition_seeds"] = [42]
        c["training"]["learning_rate"] = rate
        c["training"]["head_learning_rate"] = rate
        c["training"]["method_overrides"] = {"gev0_a_dense": lr_overrides(rate)}
        c["communication"]["wire_ratios"] = [0.1]
        c["matrix"]["dense_methods"] = ["gev0_a_dense"]
        c["matrix"]["sparse_methods"] = []
        c["project"]["expected_runs"] = len(TASKS)
        write(f"roberta_gev0_dev_lr_{tag}", c, f"gev0 learning-rate sweep, lr {tag} (train-holdout, seed 42), same rule as the baselines.")


def gev0repro() -> None:
    c = confirm_config()
    c["project"]["output_dir"] = f"{SERVER}/docs/results/gev0_repro"
    write("roberta_gev0_repro", c, "confirm_main relocated to bishe_base: one gev0 run must reproduce confirm_main exactly.")


def gev0main(dev_root: Path) -> None:
    confirm = yaml.safe_load((ROOT / "configs" / "roberta_confirm_main.yaml").read_text())
    current = confirm["training"]["method_overrides"]["gev0_a_dense"]["learning_rate"]
    scores: dict[str, dict[str, float]] = {}
    for tag in GEV0_GRID:
        for task in TASKS:
            value = last3(dev_root / f"gev0_dev_lr_{tag}" / f"{task}__gev0_a_dense__seed-42__dense")
            if value is None:
                raise SystemExit(f"missing gev0 dev run: lr {tag} {task}")
            scores.setdefault(task, {})[tag] = value
    chosen = {task: max(values, key=values.get) for task, values in scores.items()}
    changed = [task for task in TASKS if float(chosen[task]) != float(current[task])]
    record = {"scores": scores, "chosen": chosen, "confirmation_rates": current, "changed_tasks": changed}
    print(json.dumps(record, indent=1))
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "gev0_lr_choice.json").write_text(json.dumps(record, indent=1))
    if not changed:
        print("UNCHANGED")
        return
    c = confirm_config()
    c["project"]["output_dir"] = f"{SERVER}/docs/results/gev0_retuned"
    c["data"]["tasks"] = changed
    for key in ("rounds", "local_steps", "batch_size", "eval_every"):
        c["training"][key] = {t: v for t, v in c["training"][key].items() if t in changed}
    rates = {task: float(chosen[task]) for task in changed}
    c["training"]["method_overrides"] = {
        m: {"learning_rate": dict(rates), "head_learning_rate": dict(rates)} for m in ("gev0_a_dense", "gev0_a_ef")
    }
    for key in ("gamma",):
        c["cab"][key] = {t: v for t, v in c["cab"][key].items() if t in changed}
    c["matrix"]["dense_methods"] = ["gev0_a_dense"]
    c["matrix"]["sparse_methods"] = ["gev0_a_ef"]
    c["project"]["expected_runs"] = len(changed) * 3 * (1 + 3)
    write(
        "roberta_gev0_retuned",
        c,
        f"gev0 rerun on the confirmation protocol for tasks whose swept rate changed: {json.dumps(rates)} "
        f"(confirmation rates were {json.dumps(current)}).",
    )


def last3(path: Path) -> float | None:
    summary = path / "summary.json"
    if not summary.exists():
        return None
    data = json.loads(summary.read_text())
    if data.get("status") != "COMPLETE":
        return None
    curve = [r["primary_metric"] for r in data["rounds"] if r.get("primary_metric") is not None]
    return sum(curve[-3:]) / len(curve[-3:])


def main_config(dev_root: Path) -> None:
    scores: dict[str, dict[str, dict[str, float]]] = {}
    sweeps = [("baselines_dev_lr_", tag, methods, TASKS) for tag, methods in GRID.items()]
    sweeps += [("baselines_dev2_lr_", tag, methods, tasks) for tag, (methods, tasks) in GRID2.items()]
    for prefix, tag, methods, tasks in sweeps:
        for method in methods:
            for task in tasks:
                suffix = "wire-10" if method == "fedsb" else "dense"
                value = last3(dev_root / f"{prefix}{tag}" / f"{task}__{method}__seed-42__{suffix}")
                if value is None:
                    raise SystemExit(f"missing dev run: {prefix}{tag} {task} {method}")
                scores.setdefault(method, {}).setdefault(task, {})[tag] = value
    chosen = {m: {t: max(v, key=v.get) for t, v in by_task.items()} for m, by_task in scores.items()}
    print(json.dumps({"scores": scores, "chosen": chosen}, indent=1))
    confirm = yaml.safe_load((ROOT / "configs" / "roberta_confirm_main.yaml").read_text())
    c = base_config()
    c["project"]["output_dir"] = f"{SERVER}/docs/results/baselines_main"
    overrides = {}
    for method, by_task in chosen.items():
        rates = {task: float(tag) for task, tag in by_task.items()}
        overrides[method] = {"learning_rate": dict(rates), "head_learning_rate": dict(rates)}
    for method, source in EF_SHARES.items():
        overrides[method] = copy.deepcopy(overrides[source])
    overrides["full_lora_ef"] = copy.deepcopy(confirm["training"]["method_overrides"]["fedavg_lora"])
    c["training"]["method_overrides"] = overrides
    c["matrix"]["dense_methods"] = list(DENSE)
    c["matrix"]["sparse_methods"] = ["full_lora_ef", "rolora_ef", "fedsvd_ef", "fedsb"]
    c["project"]["expected_runs"] = len(TASKS) * 3 * (len(DENSE) + 4 * 3)
    write(
        "roberta_baselines_main",
        c,
        "Published federated LoRA baselines, confirmation protocol (official validation, seeds "
        "1001-1003, 5/10/20%).  Learning rates picked by best last-3 mean in the seed-42 dev "
        f"sweep: {json.dumps(chosen, sort_keys=True)}",
    )
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "baseline_lr_choice.json").write_text(json.dumps({"scores": scores, "chosen": chosen}, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["dev"]:
        dev()
    elif sys.argv[1:2] == ["dev2"]:
        dev2()
    elif sys.argv[1:2] == ["gev0dev"]:
        gev0dev()
    elif sys.argv[1:2] == ["gev0repro"]:
        gev0repro()
    elif sys.argv[1:2] == ["gev0main"] and len(sys.argv) == 3:
        gev0main(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["main"] and len(sys.argv) == 3:
        main_config(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
