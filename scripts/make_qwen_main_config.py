"""Pick the Qwen2.5-0.5B learning rates from the baseline sweep and write the main configs.

Selection rule (fixed before any main-stage run, same as QNLI): the learning rate whose
run has the best mean of its last three evaluations, separately for random_a_ef at 10%
(shared by every fixed-A arm) and for dense fedavg_lora.

Writes configs/qwen05b_main.yaml (cab0 / cab0wh / gev0 at 10%, whitened arms with the
Gram refinement round) and configs/qwen05b_norefine.yaml (gev0 without it).  The
random_a_ef and fedavg_lora runs at the chosen rates are reused from the sweep.

Usage (server): python scripts/make_qwen_main_config.py <sweep results dir>
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVER = "/data/home/202412012250/bishe_llm2"


def last3(path: Path) -> float | None:
    summary = path / "summary.json"
    if not summary.exists():
        return None
    rounds = json.loads(summary.read_text())["rounds"]
    curve = [r["primary_metric"] for r in rounds if r.get("primary_metric") is not None]
    return sum(curve[-3:]) / len(curve[-3:])


def main() -> None:
    sweep = Path(sys.argv[1])
    scores = {"random_a_ef": {}, "fedavg_lora": {}}
    for directory in sorted(sweep.glob("qwen05b_lr_*")):
        lr = directory.name.removeprefix("qwen05b_lr_")
        for method, suffix in (("random_a_ef", "wire-10"), ("fedavg_lora", "dense")):
            value = last3(directory / f"mnli__{method}__seed-361__{suffix}")
            if value is not None:
                scores[method][lr] = value
    chosen = {method: max(values, key=values.get) for method, values in scores.items()}
    print(json.dumps({"scores": scores, "chosen": chosen}, indent=1))

    base = yaml.safe_load((ROOT / "configs" / f"qwen05b_lr_{chosen['random_a_ef']}.yaml").read_text())
    fixed_lr = float(chosen["random_a_ef"])
    dense_lr = float(chosen["fedavg_lora"])
    fixed = {"learning_rate": {"mnli": fixed_lr}, "head_learning_rate": {"mnli": fixed_lr}}
    for name, sparse, refine in (
        ("qwen05b_main", ["random_a_ef", "cab0_a_ef", "cab0wh_a_ef", "gev0_a_ef"], "gram"),
        ("qwen05b_norefine", ["gev0_a_ef"], "none"),
    ):
        c = copy.deepcopy(base)
        c["project"]["root"] = SERVER
        c["project"]["output_dir"] = f"{SERVER}/docs/results/{name}"
        c["project"]["spectral_cache"] = f"{SERVER}/artifacts/spectral/qwen2.5-0.5b-qv-r8-fp32.pt"
        c["training"]["learning_rate"] = fixed_lr
        c["training"]["head_learning_rate"] = fixed_lr
        c["training"]["method_overrides"] = {
            "fedavg_lora": {"learning_rate": {"mnli": dense_lr}, "head_learning_rate": {"mnli": dense_lr}},
            **{m: copy.deepcopy(fixed) for m in sparse},
        }
        c["matrix"]["sparse_methods"] = sparse
        c["matrix"]["dense_methods"] = ["fedavg_lora"]
        c["cab"]["basis_refine"] = refine
        c["project"]["expected_runs"] = 1 + len(sparse)
        note = (
            f"Qwen2.5-0.5B main comparison (MNLI, seed 361, 10%, train-holdout).  Learning rates "
            f"chosen by best last-3 mean in the baseline sweep: fixed-A {fixed_lr:g}, dense "
            f"{dense_lr:g}.  Whitened bases use basis_refine={refine}."
        )
        (ROOT / "configs" / f"{name}.yaml").write_text(f"# {note}\n" + yaml.safe_dump(c, sort_keys=False))
        print("wrote", name)
    (ROOT / "logs" / "qwen_lr_choice.json").write_text(json.dumps({"scores": scores, "chosen": chosen}, indent=1))


if __name__ == "__main__":
    main()
