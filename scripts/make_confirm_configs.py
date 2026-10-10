"""Write the confirmation-stage configs (official validation, three fresh seeds).

Usage: python scripts/make_confirm_configs.py --client-rank 4 --activation-rank 8 \
           --factor-dtype int8 --basis-dtype int8 [--seeds 1001,1002,1003]

Every number in the paper comes from these runs.  Schedules and learning rates are
the ones fixed in the development stage (SST-2 / MNLI from cab_ef_dev, QNLI from the
seed-42 baseline sweep); nothing is re-tuned on official validation scores.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVER = "/data/home/202412012250/bishe_confirm"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-rank", type=int, required=True)
    parser.add_argument("--activation-rank", type=int, required=True)
    parser.add_argument("--factor-dtype", choices=["float16", "int8"], required=True)
    parser.add_argument("--basis-dtype", choices=["float16", "int8"], required=True)
    parser.add_argument("--seeds", default="1001,1002,1003")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    dev = yaml.safe_load((ROOT / "configs" / "roberta_gev_dev.yaml").read_text())
    qnli = yaml.safe_load((ROOT / "configs" / "roberta_qnli_dev.yaml").read_text())
    base = copy.deepcopy(dev)
    base["project"].update({"phase": "final", "root": SERVER})
    base["evaluation"] = {"policy": "official_validation"}
    base["data"]["tasks"] = ["sst2", "mnli", "qnli"]
    base["data"]["partition_seeds"] = seeds
    training = base["training"]
    for key in ("rounds", "local_steps", "batch_size", "eval_every"):
        values = {task: dev["training"][key][task] for task in ("sst2", "mnli")}
        values["qnli"] = qnli["training"][key]["qnli"]
        training[key] = values
    fixed_lr = {"sst2": 3e-4, "mnli": 1e-3, "qnli": 1e-3}
    dense_lr = {"sst2": 1e-3, "mnli": 1e-3, "qnli": 1e-3}
    fixed = {"learning_rate": dict(fixed_lr), "head_learning_rate": dict(fixed_lr)}
    dense = {"learning_rate": dict(dense_lr), "head_learning_rate": dict(dense_lr)}
    fixed_methods = [
        "random_a_dense", "gev0_a_dense",
        "random_a_ef", "spectral_a_ef", "cab0_a_ef", "cab0wh_a_ef", "gev0_a_ef", "eva_a_ef",
    ]
    training["method_overrides"] = {"fedavg_lora": dense, **{m: copy.deepcopy(fixed) for m in fixed_methods}}
    base["cab"].update(
        {
            "gamma": {"sst2": 0.1, "mnli": 0.1, "qnli": 0.1},
            "client_rank": args.client_rank,
            "activation_rank": args.activation_rank,
            "factor_dtype": args.factor_dtype,
            "basis_dtype": args.basis_dtype,
        }
    )
    protocol = (
        f"basis protocol: client gradient rank {args.client_rank}, input-sketch rank "
        f"{args.activation_rank}, factors {args.factor_dtype}, basis broadcast {args.basis_dtype}"
    )

    main_cfg = copy.deepcopy(base)
    main_cfg["project"]["output_dir"] = f"{SERVER}/docs/results/confirm_main"
    main_cfg["communication"]["wire_ratios"] = [0.05, 0.10, 0.20]
    main_cfg["matrix"].update(
        {
            "dense_methods": ["fedavg_lora", "random_a_dense", "gev0_a_dense"],
            "sparse_methods": ["random_a_ef", "spectral_a_ef", "cab0_a_ef", "cab0wh_a_ef", "gev0_a_ef"],
        }
    )
    main_cfg["project"]["expected_runs"] = 3 * len(seeds) * (3 + 5 * 3)
    eva_cfg = copy.deepcopy(base)
    eva_cfg["project"]["output_dir"] = f"{SERVER}/docs/results/confirm_eva"
    eva_cfg["communication"]["wire_ratios"] = [0.10]
    eva_cfg["matrix"].update({"dense_methods": [], "sparse_methods": ["eva_a_ef"]})
    eva_cfg["project"]["expected_runs"] = 3 * len(seeds)

    notes = {
        "confirm_main": f"""Confirmation stage (phase final): SST-2 / MNLI / QNLI, official validation,
fresh seeds {seeds}.  Paper numbers come from this stage only.
Arms: dense FedAvg-LoRA (C95 reference), random_a_dense (FFA-LoRA style) and
gev0_a_dense (method without compression), and at 5/10/20% exact-wire budgets with
bidirectional Top-k + EF: random, spectral, cab0 (Euclidean gradient basis), cab0wh
(whitening ablation) and gev0 (curvature-aware generalized eigenvectors).
Schedules and learning rates fixed in the development stage; {protocol}.""",
        "confirm_eva": f"""Confirmation stage supplement: EVA-style fixed A (top input second-moment
directions) at the 10% budget, same seeds {seeds}, official validation; {protocol}.""",
    }
    for name, cfg in (("confirm_main", main_cfg), ("confirm_eva", eva_cfg)):
        header = "".join(f"# {line}\n" for line in notes[name].splitlines())
        (ROOT / "configs" / f"roberta_{name}.yaml").write_text(
            header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        )
        print("wrote", name, cfg["project"]["expected_runs"], "runs")


if __name__ == "__main__":
    main()
