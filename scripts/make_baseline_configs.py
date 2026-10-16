"""Configs for the published federated LoRA baselines on the RoBERTa confirmation protocol.

dev   One config per learning rate (train-holdout, seed 42, same rounds / steps / clients as
      configs/roberta_confirm_main.yaml).  Each dense baseline is swept over {3e-4, 1e-3, 3e-3},
      Fed-SB (10% budget) over {1e-3, 3e-3, 1e-2} (it trains only an r x r matrix).
dev2  Extends the grid wherever the first sweep picked a boundary value: Fed-SB down to
      {1e-4, 3e-4} (SST-2 / MNLI picked 1e-3), FLoRA down to 1e-4 (QNLI picked 3e-4), RoLoRA and
      FedSVD up to 1e-2 on SST-2 (picked 3e-3); then FLoRA down to 3e-5 on QNLI (picked 1e-4).
      Separate configs keep the completed runs valid.
dev3  Fills in the remaining cells so that every dense baseline covers the full grid
      {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} that gev0 was swept over (19 runs); sweepcheck then reports
      whether any selected rate changes.
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
normfix   Does the curvature normalization fix other data-driven frozen bases?  EVA (eva_a_ef) and
          EVA with its subspace normalized (evawh_a_ef) at the fixed-basis learning rates, and Fed-SB
          with its input side normalized (fedsb_norm) at Fed-SB's swept rates; confirmation protocol,
          code tree bishe_norm.  Also writes a smoke config.
gev0repro The confirmation config relocated to this code tree, to check that a gev0 run
          reproduces confirm_main exactly here.
normfixdev     Normalization shrinks Fed-SB's input scale, so fedsb_norm gets its own sweep
               {3e-4 .. 3e-2} (train-holdout, seed 42, 10% budget), same rule as Fed-SB.
normfixdev2    Extends that sweep to 1e-4 on QNLI, which picked the lowest rate.
normfixretune  Reads both sweeps (optionally for a comma-separated subset of tasks); for every task whose
               pick differs from Fed-SB's rate, reruns fedsb_norm on the confirmation protocol
               (configs/roberta_normfix_retuned[_tasks].yaml).  Stops if a pick is on the grid boundary.

Usage:  python scripts/make_baseline_configs.py dev | dev2 | dev3 | gev0dev | gev0repro | normfix | normfixdev | normfixdev2
        python scripts/make_baseline_configs.py main | gev0main | sweepcheck <dev results root>   (on the server)
        python scripts/make_baseline_configs.py normfixretune <dev results root> [task,task]   (on the server)
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
# Fill-in: every dense baseline on the same 5-point grid {1e-4 .. 1e-2} that gev0 was swept over
# (the first two sweeps only extended a grid where its best value lay on the boundary).
GRID3 = [  # (suffix, rate, methods, tasks)
    ("1e-4", "1e-4", ("fedex_dense", "rolora_dense", "fedsvd_dense"), TASKS),
    ("1e-2", "1e-2", ("flora_dense", "fedex_dense"), TASKS),
    ("1e-2_mq", "1e-2", ("rolora_dense", "fedsvd_dense"), ("mnli", "qnli")),
]
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


def dev3() -> None:
    for suffix, tag, methods, tasks in GRID3:
        c = base_config()
        rate = float(tag)
        c["project"]["phase"] = "tune"
        c["project"]["output_dir"] = f"{SERVER}/docs/results/baselines_dev3_lr_{suffix}"
        c["evaluation"] = {"policy": "train_holdout"}
        c["data"]["tasks"] = list(tasks)
        c["data"]["partition_seeds"] = [42]
        c["training"]["learning_rate"] = rate
        c["training"]["head_learning_rate"] = rate
        c["training"]["method_overrides"] = {m: lr_overrides(rate) for m in methods}
        for key in ("rounds", "local_steps", "batch_size", "eval_every"):
            c["training"][key] = {t: v for t, v in c["training"][key].items() if t in tasks}
        c["communication"]["wire_ratios"] = [0.1]
        c["matrix"]["dense_methods"] = list(methods)
        c["matrix"]["sparse_methods"] = []
        c["project"]["expected_runs"] = len(tasks) * len(methods)
        write(
            f"roberta_baselines_dev3_lr_{suffix}",
            c,
            f"Baseline sweep fill-in, lr {tag}: {', '.join(methods)} on {', '.join(tasks)} "
            "(train-holdout, seed 42), so that every dense baseline covers the full 5-point grid.",
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


NORM_SERVER = "/data/home/202412012250/bishe_norm"


def normfix() -> None:
    confirm = yaml.safe_load((ROOT / "configs" / "roberta_confirm_main.yaml").read_text())
    fedsb_rates = json.loads((ROOT / "docs" / "results" / "baselines" / "LR_CHOICE.json").read_text())["chosen"]["fedsb"]
    c = copy.deepcopy(confirm)
    c["project"]["root"] = NORM_SERVER
    c["project"]["output_dir"] = f"{NORM_SERVER}/docs/results/normfix_main"
    c["project"]["spectral_cache"] = f"{NORM_SERVER}/artifacts/spectral/roberta-base-qv-r8-fp32.pt"
    c["baselines"] = {"fedsb_gradient_examples": 1024, "fedsb_gradient_batch_size": 64}
    fixed = copy.deepcopy(confirm["training"]["method_overrides"]["gev0_a_ef"])
    rates = {task: float(tag) for task, tag in fedsb_rates.items()}
    c["training"]["method_overrides"] = {
        "eva_a_ef": copy.deepcopy(fixed),
        "evawh_a_ef": copy.deepcopy(fixed),
        "fedsb_norm": {"learning_rate": dict(rates), "head_learning_rate": dict(rates)},
    }
    c["matrix"]["dense_methods"] = []
    c["matrix"]["sparse_methods"] = ["eva_a_ef", "evawh_a_ef", "fedsb_norm"]
    c["project"]["expected_runs"] = len(TASKS) * 3 * 3 * 3
    write("roberta_normfix_main", c, "Normalization as a general fix: EVA vs EVA-normalized (fixed-basis rates) and "
          f"Fed-SB-normalized (Fed-SB's swept rates {json.dumps(fedsb_rates, sort_keys=True)}); confirmation protocol.")
    smoke = copy.deepcopy(c)
    smoke["project"]["phase"] = "smoke"
    smoke["project"]["output_dir"] = f"{NORM_SERVER}/docs/results/normfix_smoke"
    smoke["project"]["partition_dir"] = f"{NORM_SERVER}/artifacts/partitions-smoke"
    smoke["project"].pop("expected_runs")
    smoke["data"].update({"tasks": ["sst2"], "num_clients": 3, "clients_per_round": 3, "partition_seeds": [1001]})
    smoke["training"]["rounds"] = {"sst2": 4}
    smoke["training"]["local_steps"] = {"sst2": 4}
    smoke["training"]["batch_size"] = {"sst2": 32}
    smoke["training"]["eval_every"] = {"sst2": 1}
    smoke["communication"]["wire_ratios"] = [0.1]
    smoke["cab"]["gradient_examples"] = {"sst2": 256}
    smoke["cab"]["gamma"] = {"sst2": 0.1}
    smoke["baselines"] = {"fedsb_gradient_examples": 256, "fedsb_gradient_batch_size": 64}
    smoke["matrix"]["sparse_methods"] = ["evawh_a_ef", "fedsb_norm"]
    write("roberta_normfix_smoke", smoke, "Smoke test for evawh_a_ef and fedsb_norm.")


# Normalization shrinks the input scale of Fed-SB's high-energy right basis to c, so the rate swept for
# unnormalized Fed-SB is not its rate; sweep fedsb_norm by the same rule (it shares Fed-SB's grid top).
FEDSB_NORM_GRID = ("3e-4", "1e-3", "3e-3", "1e-2", "3e-2")
# boundary extension: QNLI picked the lowest rate of the first grid
FEDSB_NORM_GRID2 = {"1e-4": ("qnli",)}


def normfixdev() -> None:
    for tag in FEDSB_NORM_GRID:
        c = yaml.safe_load((ROOT / "configs" / "roberta_normfix_main.yaml").read_text())
        rate = float(tag)
        c["project"]["phase"] = "tune"
        c["project"]["output_dir"] = f"{NORM_SERVER}/docs/results/normfix_dev_lr_{tag}"
        c["project"]["expected_runs"] = len(TASKS)
        c["evaluation"] = {"policy": "train_holdout"}
        c["data"]["partition_seeds"] = [42]
        c["training"]["learning_rate"] = rate
        c["training"]["head_learning_rate"] = rate
        c["training"]["method_overrides"] = {"fedsb_norm": lr_overrides(rate)}
        c["communication"]["wire_ratios"] = [0.1]
        c["matrix"]["sparse_methods"] = ["fedsb_norm"]
        write(f"roberta_normfix_dev_lr_{tag}", c,
              f"Fed-SB-normalized learning-rate sweep, lr {tag} (train-holdout, seed 42, 10% budget), same rule as Fed-SB.")


def normfixdev2() -> None:
    for tag, tasks in FEDSB_NORM_GRID2.items():
        c = yaml.safe_load((ROOT / "configs" / "roberta_normfix_dev_lr_1e-3.yaml").read_text())
        rate = float(tag)
        c["project"]["output_dir"] = f"{NORM_SERVER}/docs/results/normfix_dev2_lr_{tag}"
        c["project"]["expected_runs"] = len(tasks)
        c["data"]["tasks"] = list(tasks)
        for key in ("rounds", "local_steps", "batch_size", "eval_every"):
            c["training"][key] = {t: v for t, v in c["training"][key].items() if t in tasks}
        c["training"]["learning_rate"] = rate
        c["training"]["head_learning_rate"] = rate
        c["training"]["method_overrides"] = {"fedsb_norm": lr_overrides(rate)}
        write(f"roberta_normfix_dev2_lr_{tag}", c,
              f"Fed-SB-normalized sweep extension, lr {tag} on {', '.join(tasks)} (picked the lowest rate of the first grid).")


def normfixretune(dev_root: Path, tasks: tuple[str, ...] = TASKS) -> None:
    """Pick fedsb_norm's rate for ``tasks`` from both sweeps; rerun the tasks whose rate differs from Fed-SB's."""
    fedsb_rates = json.loads((ROOT / "docs" / "results" / "baselines" / "LR_CHOICE.json").read_text())["chosen"]["fedsb"]
    sweeps = [(f"normfix_dev_lr_{tag}", tag, TASKS) for tag in FEDSB_NORM_GRID]
    sweeps += [(f"normfix_dev2_lr_{tag}", tag, extra) for tag, extra in FEDSB_NORM_GRID2.items()]
    scores: dict[str, dict[str, float]] = {}
    for directory, tag, swept in sweeps:
        for task in tasks:
            if task not in swept:
                continue
            value = last3(dev_root / directory / f"{task}__fedsb_norm__seed-42__wire-10")
            if value is None:
                raise SystemExit(f"missing fedsb_norm dev run: {directory} {task}")
            scores.setdefault(task, {})[tag] = value
    chosen = {task: max(values, key=values.get) for task, values in scores.items()}
    changed = [task for task in tasks if float(chosen[task]) != float(fedsb_rates[task])]
    boundary = [task for task in tasks if float(chosen[task]) in (min(map(float, scores[task])), max(map(float, scores[task])))]
    suffix = "" if tuple(tasks) == TASKS else "_" + "_".join(tasks)
    record = {"scores": scores, "chosen": chosen, "fedsb_rates": fedsb_rates, "changed_tasks": changed,
              "boundary_tasks": boundary}
    print(json.dumps(record, indent=1))
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / f"fedsb_norm_lr_choice{suffix}.json").write_text(json.dumps(record, indent=1))
    if boundary:
        raise SystemExit(f"rate on the grid boundary for {boundary}: extend the sweep first")
    if not changed:
        print("UNCHANGED")
        return
    c = yaml.safe_load((ROOT / "configs" / "roberta_normfix_main.yaml").read_text())
    c["project"]["output_dir"] = f"{NORM_SERVER}/docs/results/normfix_retuned{suffix}"
    c["data"]["tasks"] = changed
    for key in ("rounds", "local_steps", "batch_size", "eval_every"):
        c["training"][key] = {t: v for t, v in c["training"][key].items() if t in changed}
    for key in ("gamma", "gradient_examples"):
        if key in c.get("cab", {}) and isinstance(c["cab"][key], dict):
            c["cab"][key] = {t: v for t, v in c["cab"][key].items() if t in changed}
    rates = {task: float(chosen[task]) for task in changed}
    c["training"]["method_overrides"] = {"fedsb_norm": {"learning_rate": dict(rates), "head_learning_rate": dict(rates)}}
    c["matrix"]["sparse_methods"] = ["fedsb_norm"]
    c["project"]["expected_runs"] = len(changed) * 3 * 3
    write(f"roberta_normfix_retuned{suffix}", c,
          f"Fed-SB-normalized on the confirmation protocol at its own swept rates {json.dumps(rates)} "
          f"(Fed-SB's rates were {json.dumps(fedsb_rates, sort_keys=True)}).")


def last3(path: Path) -> float | None:
    summary = path / "summary.json"
    if not summary.exists():
        return None
    data = json.loads(summary.read_text())
    if data.get("status") != "COMPLETE":
        return None
    curve = [r["primary_metric"] for r in data["rounds"] if r.get("primary_metric") is not None]
    return sum(curve[-3:]) / len(curve[-3:])


def sweep_scores(dev_root: Path, fill: bool = False) -> dict[str, dict[str, dict[str, float]]]:
    scores: dict[str, dict[str, dict[str, float]]] = {}
    sweeps = [(f"baselines_dev_lr_{tag}", tag, methods, TASKS) for tag, methods in GRID.items()]
    sweeps += [(f"baselines_dev2_lr_{tag}", tag, methods, tasks) for tag, (methods, tasks) in GRID2.items()]
    if fill:
        sweeps += [(f"baselines_dev3_lr_{suffix}", tag, methods, tasks) for suffix, tag, methods, tasks in GRID3]
    for directory, tag, methods, tasks in sweeps:
        for method in methods:
            for task in tasks:
                suffix = "wire-10" if method == "fedsb" else "dense"
                value = last3(dev_root / directory / f"{task}__{method}__seed-42__{suffix}")
                if value is None:
                    raise SystemExit(f"missing dev run: {directory} {task} {method}")
                scores.setdefault(method, {}).setdefault(task, {})[tag] = value
    return scores


def sweepcheck(dev_root: Path) -> None:
    """With the filled-in grid, does any baseline's selected rate change?  Prints CHANGED / UNCHANGED."""
    scores = sweep_scores(dev_root, fill=True)
    chosen = {m: {t: max(v, key=v.get) for t, v in by_task.items()} for m, by_task in scores.items()}
    previous = json.loads((ROOT / "docs" / "results" / "baselines" / "LR_CHOICE.json").read_text())["chosen"]
    changed = [(m, t, previous[m][t], chosen[m][t]) for m in chosen for t in chosen[m]
               if float(chosen[m][t]) != float(previous[m][t])]
    record = {"scores": scores, "chosen": chosen, "previous": previous, "changed": changed}
    print(json.dumps(record, indent=1))
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "baseline_lr_choice_full.json").write_text(json.dumps(record, indent=1))
    print("CHANGED" if changed else "UNCHANGED")


def main_config(dev_root: Path) -> None:
    scores = sweep_scores(dev_root)
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
    elif sys.argv[1:2] == ["dev3"]:
        dev3()
    elif sys.argv[1:2] == ["sweepcheck"] and len(sys.argv) == 3:
        sweepcheck(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["gev0dev"]:
        gev0dev()
    elif sys.argv[1:2] == ["normfix"]:
        normfix()
    elif sys.argv[1:2] == ["normfixdev"]:
        normfixdev()
    elif sys.argv[1:2] == ["normfixdev2"]:
        normfixdev2()
    elif sys.argv[1:2] == ["normfixretune"] and len(sys.argv) in (3, 4):
        normfixretune(Path(sys.argv[2]), tuple(sys.argv[3].split(",")) if len(sys.argv) == 4 else TASKS)
    elif sys.argv[1:2] == ["gev0repro"]:
        gev0repro()
    elif sys.argv[1:2] == ["gev0main"] and len(sys.argv) == 3:
        gev0main(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["main"] and len(sys.argv) == 3:
        main_config(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
