"""Audit and summarise a DCEF or CAB campaign.

Reads every configured run's summary.json under the config's output_dir and
writes, into the same directory:
  DCEF_RUNS.csv   one row per run: technical audit + stability endpoints
  DCEF_PAIRS.csv  matched (task, seed, wire) differences against standard EF
  DCEF_REPORT.md  counts, paired summaries and the mechanism index

Stability endpoints follow the frozen definitions used since Stage A:
  target   = 0.95 x paired dense final primary score
  C95      = cumulative delivered bytes at the first evaluation >= target
  TailDrop = max(last three evaluated scores) - final score
  strict   = technically valid, C95 reached, final >= target, no evaluation
             below target after the first hit, TailDrop <= task limit
Practical-difference thresholds for win/tie/loss are the preregistered 0.003
(SST-2, MNLI, MRPC, QQP) and 0.01 (RTE) on the primary score.

Usage:
  python3 scripts/build_dcef_report.py --config configs/roberta_dcef_dev.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedpeft.config import load_config, run_id  # noqa: E402

TAIL_DROP_LIMIT = {"sst2": 0.01, "mnli": 0.01, "rte": 0.02, "mrpc": 0.01, "qqp": 0.01, "qnli": 0.01}
PRACTICAL_DELTA = {"sst2": 0.003, "mnli": 0.003, "rte": 0.01, "mrpc": 0.003, "qqp": 0.003, "qnli": 0.003}
CONSERVATION_LIMIT = 1e-6
COMPARISONS = (
    ("full_lora_dcef", "full_lora_ef"),
    ("full_lora_dcef_client", "full_lora_ef"),
    ("full_lora_dcef", "full_lora_dcef_client"),
    ("random_a_dcef", "random_a_ef"),
    ("spectral_a_dcef", "spectral_a_ef"),
    ("cab_a_dcef", "cab0_a_dcef"),
    ("cab_a_dcef", "random_a_dcef"),
    ("cab_a_dcef", "spectral_a_dcef"),
    ("cab0_a_dcef", "random_a_dcef"),
    ("spectral_a_dcef", "random_a_dcef"),
    ("cab_a_dcef", "full_lora_dcef"),
    ("cab_a_ef", "cab0_a_ef"),
    ("cab_a_ef", "random_a_ef"),
    ("cab_a_ef", "spectral_a_ef"),
    ("cab0_a_ef", "random_a_ef"),
    ("spectral_a_ef", "random_a_ef"),
    ("gev0_a_ef", "random_a_ef"),
    ("gev0_a_ef", "spectral_a_ef"),
    ("gev0_a_ef", "cab0_a_ef"),
    ("gev_a_ef", "random_a_ef"),
    ("gev_a_ef", "gev0_a_ef"),
    ("gev_a_ef", "cab_a_ef"),
    ("cab0wh_a_ef", "cab0_a_ef"),
    ("cab0wh_a_ef", "gev0_a_ef"),
    ("randomwh_a_ef", "random_a_ef"),
    ("gev0x_a_ef", "gev0_a_ef"),
    ("gev0x_a_ef", "random_a_ef"),
    ("gev0_a_ef", "cab0wh_a_ef"),
    ("cab0wh_a_ef", "random_a_ef"),
    ("eva_a_ef", "random_a_ef"),
    ("gev0_a_ef", "eva_a_ef"),
    ("gev0_a_dense", "random_a_dense"),
    ("random_a_dense", "fedavg_lora"),
    ("gev0_a_dense", "fedavg_lora"),
)
CAB_PREFIX = "cab"
# Methods that install a CAB-protocol basis without the "cab" name prefix.
CAB_FAMILY_EXTRA = {"gev_a_ef", "gev0_a_ef", "gev0x_a_ef", "randomwh_a_ef", "eva_a_ef", "evawh_a_ef", "gev0_a_dense"}


def is_cab_family(method: str) -> bool:
    return method.startswith(CAB_PREFIX) or method in CAB_FAMILY_EXTRA
DENSE_REFERENCE = "fedavg_lora"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if _finite(value):
        return f"{float(value):.{digits}f}"
    return str(value)


def expected_specs(cfg: dict[str, Any]) -> list[tuple[str, str, int, float | None]]:
    matrix = cfg["matrix"]
    dense = list(matrix.get("dense_methods", [])) + list(matrix.get("diagnostic_dense_methods", []))
    sparse = list(matrix.get("sparse_methods", [])) + list(matrix.get("diagnostic_sparse_methods", []))
    ratios = [float(value) for value in cfg["communication"]["wire_ratios"]]
    specs: list[tuple[str, str, int, float | None]] = []
    for task in cfg["data"]["tasks"]:
        for seed in cfg["data"]["partition_seeds"]:
            specs.extend((task, method, int(seed), None) for method in dense)
            specs.extend((task, method, int(seed), ratio) for method in sparse for ratio in ratios)
    return specs


def load_summary(output_dir: Path, spec: tuple[str, str, int, float | None]) -> dict[str, Any] | None:
    run_dir = output_dir / run_id(*spec)
    path = run_dir / "summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    failure = run_dir / "failure.json"
    if failure.exists():
        # A run that raised (e.g. a non-finite local loss after divergence) is an
        # outcome, not a missing run; keep its error type for the report.
        return json.loads(failure.read_text(encoding="utf-8"))
    return None


def evaluated_trajectory(summary: dict[str, Any]) -> list[tuple[int, float, int]]:
    out = []
    for record in summary.get("rounds", []):
        score = record.get("primary_metric")
        if score is None:
            continue
        out.append(
            (int(record["round"]), float(score), int(record["wire"]["total_delivered_bytes"]))
        )
    return out


def technical_audit(summary: dict[str, Any] | None, method: str) -> tuple[bool, list[str]]:
    if summary is None:
        return False, ["MISSING"]
    reasons: list[str] = []
    if summary.get("status") == "FAILED":
        reasons.append(f"FAILED:{summary.get('error_type', 'unknown')}")
        return False, reasons
    if summary.get("status") != "COMPLETE":
        reasons.append("NOT_COMPLETE")
        return False, reasons
    invariants = summary.get("invariants", {})
    if invariants.get("wire_cap_respected") is not True:
        reasons.append("WIRE_CAP")
    if invariants.get("finite_metrics") is not True:
        reasons.append("NON_FINITE_METRIC")
    # FedSVD re-factorises its A every round by design.
    if invariants.get("basis_unchanged") is not True and not method.startswith("fedsvd"):
        reasons.append("BASIS_CHANGED")
    if method.startswith("fedsb"):
        baseline = summary.get("baseline") or {}
        setup = int(summary.get("final", {}).get("wire", {}).get("setup_bytes", -1))
        if setup != int(baseline.get("setup_bytes", -2)) or setup <= 0:
            reasons.append("FEDSB_SETUP_BYTES_NOT_COUNTED")
    if float(invariants.get("ef_conservation_max", 0.0)) > CONSERVATION_LIMIT:
        reasons.append("EF_LEDGER")
    if "dcef" in method:
        dcef = summary.get("dcef")
        if not isinstance(dcef, dict):
            reasons.append("DCEF_SECTION_MISSING")
        else:
            if int(dcef["uplink_sparse_k"]) > int(dcef["downlink_sparse_k"]):
                reasons.append("DCEF_UPLINK_K")
            if method.endswith("_client") and int(dcef["uplink_control_bytes"]) != 0:
                reasons.append("DCEF_CLIENT_CONTROL_BYTES")
            if not method.endswith("_client") and int(dcef["uplink_control_bytes"]) <= 0:
                reasons.append("DCEF_CONTROL_BYTES_MISSING")
    if is_cab_family(method):
        cab = summary.get("cab")
        if not isinstance(cab, dict):
            reasons.append("CAB_SECTION_MISSING")
        else:
            setup = int(summary.get("final", {}).get("wire", {}).get("setup_bytes", -1))
            if setup != int(cab.get("setup_bytes", -2)) or setup <= 0:
                reasons.append("CAB_SETUP_BYTES_NOT_COUNTED")
            recipe = summary.get("invariants", {}).get("before", {}).get("fixed_basis_recipe", "")
            if str(cab.get("recipe")) not in str(recipe):
                reasons.append("CAB_BASIS_NOT_INSTALLED")
    return not reasons, reasons


def stability_row(
    spec: tuple[str, str, int, float | None],
    summary: dict[str, Any] | None,
    dense: dict[str, Any] | None,
) -> dict[str, Any]:
    task, method, seed, ratio = spec
    valid, reasons = technical_audit(summary, method)
    row: dict[str, Any] = {
        "task": task,
        "method": method,
        "seed": seed,
        "wire": "dense" if ratio is None else ratio,
        "technical_valid": valid,
        "technical_reasons": ";".join(reasons),
    }
    if summary is None or summary.get("status") != "COMPLETE":
        return row
    trajectory = evaluated_trajectory(summary)
    scores = [score for _, score, _ in trajectory]
    row["final"] = scores[-1]
    row["best"] = max(scores)
    row["total_delivered_bytes"] = int(summary["final"]["wire"]["total_delivered_bytes"])
    row["setup_bytes"] = int(summary["final"]["wire"].get("setup_bytes", 0))
    cab = summary.get("cab")
    if isinstance(cab, dict):
        overview = cab.get("summary", {})
        row["cab_kappa"] = cab.get("kappa")
        row["cab_gamma"] = cab.get("gamma")
        row["cab_n_eff"] = cab.get("n_eff")
        row["cab_overlap_reported_vs_exact"] = overview.get("overlap_reported_vs_exact_mean")
        row["cab_objective_gap_vs_exact"] = overview.get("objective_gap_vs_exact_mean")
        row["cab_reported_energy_min"] = overview.get("reported_energy_fraction_min")
        row["cab_overlap_vs_spectral"] = overview.get("overlap_vs_spectral_mean")
        row["cab_overlap_vs_consensus"] = overview.get("overlap_vs_consensus_basis_mean")
        row["cab_positive_eigenvalues_min"] = overview.get("positive_eigenvalues_min")
        row["cab_het_to_consensus"] = overview.get("het_to_consensus_trace_mean")
        for key in ("selected", "exact_optimum", "spectral", "random"):
            row[f"cab_objective_{key}"] = overview.get(f"objective_{key}_mean")
        # Activation-metric runs: share of the optimal one-step K-FAC decrease and
        # common-mode share of projected input energy, per candidate basis.
        for key in ("selected", "random", "spectral", "consensus", "eva", "gev0_exact"):
            if f"kfac_capture_{key}_mean" in overview:
                row[f"kfac_capture_{key}"] = overview[f"kfac_capture_{key}_mean"]
                row[f"common_mode_{key}"] = overview[f"common_mode_{key}_mean"]
                row[f"energy_ratio_{key}"] = overview[f"energy_ratio_{key}_mean"]
    row["sparse_k"] = summary.get("wire_budget", {}).get("sparse_k")
    dcef = summary.get("dcef")
    if isinstance(dcef, dict):
        row["uplink_sparse_k"] = dcef["uplink_sparse_k"]
        gammas = [
            record["diagnostics"]["dcef"]["gamma_bar"]
            for record in summary["rounds"]
            if record.get("diagnostics", {}).get("dcef", {}).get("gamma_bar") is not None
        ]
        if gammas:
            adapter = int(summary["model"]["adapter_parameters"])
            k = int(dcef["downlink_sparse_k"])
            row["gamma_bar_mean"] = statistics.fmean(gammas)
            row["gamma_bar_final"] = gammas[-1]
            # Standard EF is predicted unstable when gamma * d / K > 2.
            row["delay_index_gamma_d_over_k"] = statistics.fmean(gammas) * adapter / k
    tail = scores[-3:]
    row["tail_drop"] = max(tail) - scores[-1]
    row["tail_drop_limit"] = TAIL_DROP_LIMIT[task]
    if dense is None or dense.get("status") != "COMPLETE":
        row["c95_status"] = "NO_DENSE_REFERENCE"
        return row
    dense_trajectory = evaluated_trajectory(dense)
    dense_final = dense_trajectory[-1][1]
    target = 0.95 * dense_final
    row["dense_final"] = dense_final
    row["target"] = target
    row["retention"] = scores[-1] / dense_final if dense_final > 0 else float("nan")
    dense_hit = next((b for _, s, b in dense_trajectory if s >= target), None)
    hit_index = next((i for i, s in enumerate(scores) if s >= target), None)
    if hit_index is None:
        row["c95_status"] = "NOT_REACHED"
        row["post_hit_violation_count"] = ""
    else:
        row["c95_status"] = "REACHED"
        row["c95_round"] = trajectory[hit_index][0]
        row["c95_bytes"] = trajectory[hit_index][2]
        row["c95_training_bytes"] = row["c95_bytes"] - row["setup_bytes"]
        after = scores[hit_index:]
        row["post_hit_violation_count"] = sum(score < target for score in after)
        row["post_hit_max_drawdown"] = max(
            max(after[: j + 1]) - after[j] for j in range(len(after))
        )
        if dense_hit is not None and row["c95_bytes"] > 0:
            row["speedup95"] = dense_hit / row["c95_bytes"]
    row["strict_stable"] = (
        valid
        and row["c95_status"] == "REACHED"
        and scores[-1] >= target
        and row.get("post_hit_violation_count") == 0
        and row["tail_drop"] <= row["tail_drop_limit"]
    )
    return row


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(r["task"], r["method"], r["seed"], r["wire"]): r for r in rows}
    out = []
    for left_method, right_method in COMPARISONS:
        for (task, method, seed, wire), left in index.items():
            if method != left_method:
                continue
            right = index.get((task, right_method, seed, wire))
            if right is None or not left.get("technical_valid") or not right.get("technical_valid"):
                continue
            delta = float(left["final"]) - float(right["final"])
            threshold = PRACTICAL_DELTA[task]
            out.append(
                {
                    "comparison": f"{left_method}-{right_method}",
                    "task": task,
                    "seed": seed,
                    "wire": wire,
                    "left_final": left["final"],
                    "right_final": right["final"],
                    "delta_final": delta,
                    "direction": "WIN" if delta > threshold else "LOSS" if delta < -threshold else "TIE",
                    "left_strict": left.get("strict_stable"),
                    "right_strict": right.get("strict_stable"),
                    "left_post_hit": left.get("post_hit_violation_count", ""),
                    "right_post_hit": right.get("post_hit_violation_count", ""),
                    "delta_tail_drop": float(left["tail_drop"]) - float(right["tail_drop"]),
                    "delta_total_bytes": int(left["total_delivered_bytes"])
                    - int(right["total_delivered_bytes"]),
                    "left_delay_index": left.get("delay_index_gamma_d_over_k", ""),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _mean_sd(values: list[float]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:+.4f}"
    return f"{statistics.fmean(values):+.4f} ± {statistics.stdev(values):.4f}"


def render_markdown(cfg: dict[str, Any], rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> str:
    total = len(rows)
    valid = sum(1 for r in rows if r.get("technical_valid"))
    missing = [r for r in rows if r.get("technical_reasons") == "MISSING"]
    invalid = [r for r in rows if not r.get("technical_valid") and r.get("technical_reasons") != "MISSING"]
    lines = [
        "# DCEF / CAB campaign report",
        "",
        f"Config: `{cfg['_config_path']}`; evaluation policy: "
        f"`{cfg.get('evaluation', {}).get('policy', 'official_validation')}`; "
        f"seeds: {cfg['data']['partition_seeds']}.",
        "",
        f"Runs: {valid}/{total} technically valid; {len(missing)} missing; {len(invalid)} invalid.",
        "",
    ]
    failed = [r for r in rows if str(r.get("technical_reasons", "")).startswith("FAILED:")]
    if failed:
        lines += [
            "Runs that raised (a FloatingPointError is a non-finite local loss, i.e. divergence):",
            "",
            "| Method | Wire | Error | Count |",
            "|---|---|---|---:|",
        ]
        counts: dict[tuple[str, Any, str], int] = {}
        for r in failed:
            key = (r["method"], r["wire"], r["technical_reasons"].split(":", 1)[1])
            counts[key] = counts.get(key, 0) + 1
        for (method, wire, error), count in sorted(counts.items(), key=lambda x: (x[0][0], str(x[0][1]))):
            lines.append(f"| {method} | {wire} | {error} | {count} |")
        lines.append("")
        invalid = [r for r in invalid if r not in failed]
    if invalid:
        lines += ["Invalid runs:", ""]
        lines += [f"- {r['task']} {r['method']} seed {r['seed']} wire {r['wire']}: {r['technical_reasons']}" for r in invalid]
        lines.append("")
    lines += [
        "## Stability counts",
        "",
        "| Task | Method | Wire | n | C95 reached | Final ≥ target | Strict stable | Mean final | Mean Speedup95 | Mean γ̄·d/K |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups: dict[tuple[str, str, Any], list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("technical_valid"):
            groups.setdefault((r["task"], r["method"], r["wire"]), []).append(r)
    for (task, method, wire), members in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], str(x[0][2]))):
        reached = sum(1 for m in members if m.get("c95_status") == "REACHED")
        kept = sum(1 for m in members if _finite(m.get("target")) and m["final"] >= m["target"])
        strict = sum(1 for m in members if m.get("strict_stable") is True)
        speed = [float(m["speedup95"]) for m in members if _finite(m.get("speedup95"))]
        index = [float(m["delay_index_gamma_d_over_k"]) for m in members if _finite(m.get("delay_index_gamma_d_over_k"))]
        lines.append(
            f"| {task} | {method} | {wire} | {len(members)} | {reached} | {kept} | {strict} | "
            f"{statistics.fmean(m['final'] for m in members):.4f} | "
            f"{_fmt(statistics.fmean(speed)) if speed else ''} | {_fmt(statistics.fmean(index), 2) if index else ''} |"
        )
    lines += [
        "",
        "## Paired differences (left − right, final primary score)",
        "",
        "| Comparison | Task | Wire | n | Δ final mean ± SD | Win / Tie / Loss | Strict (left / right) | Post-hit violations (left / right) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    pair_groups: dict[tuple[str, str, Any], list[dict[str, Any]]] = {}
    for p in pairs:
        pair_groups.setdefault((p["comparison"], p["task"], p["wire"]), []).append(p)
    for (comparison, task, wire), members in sorted(pair_groups.items(), key=lambda x: (x[0][0], x[0][1], str(x[0][2]))):
        deltas = [float(m["delta_final"]) for m in members]
        wins = sum(m["direction"] == "WIN" for m in members)
        ties = sum(m["direction"] == "TIE" for m in members)
        losses = sum(m["direction"] == "LOSS" for m in members)
        left_strict = sum(m["left_strict"] is True for m in members)
        right_strict = sum(m["right_strict"] is True for m in members)
        left_hit = sum(int(m["left_post_hit"]) for m in members if m["left_post_hit"] != "")
        right_hit = sum(int(m["right_post_hit"]) for m in members if m["right_post_hit"] != "")
        lines.append(
            f"| {comparison} | {task} | {wire} | {len(members)} | {_mean_sd(deltas)} | "
            f"{wins} / {ties} / {losses} | {left_strict} / {right_strict} | {left_hit} / {right_hit} |"
        )
    cab_rows = [r for r in rows if r.get("technical_valid") and is_cab_family(str(r["method"]))]
    if cab_rows:
        lines += [
            "",
            "## CAB basis construction (zero-training diagnostics)",
            "",
            "Objective = predicted stationary excess error J(P)/tr(S) under exact statistics and the "
            "run's kappa (lower is better); overlaps are ||A1 A2^T||_F^2 / r.",
            "",
            "| Task | Method | Wire | n | κ | J selected / J exact | overlap ↔ spectral | J selected | J spectral | J random | setup / total bytes |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        cab_groups: dict[tuple[str, str, Any], list[dict[str, Any]]] = {}
        for r in cab_rows:
            cab_groups.setdefault((r["task"], r["method"], r["wire"]), []).append(r)

        def mean_of(members, key):
            values = [float(m[key]) for m in members if _finite(m.get(key))]
            return _fmt(statistics.fmean(values)) if values else ""

        for (task, method, wire), members in sorted(cab_groups.items(), key=lambda x: (x[0][0], x[0][1], str(x[0][2]))):
            share = statistics.fmean(
                m["setup_bytes"] / m["total_delivered_bytes"] for m in members if m["total_delivered_bytes"]
            )
            lines.append(
                f"| {task} | {method} | {wire} | {len(members)} | {mean_of(members, 'cab_kappa')} | "
                f"{mean_of(members, 'cab_objective_gap_vs_exact')} | {mean_of(members, 'cab_overlap_vs_spectral')} | "
                f"{mean_of(members, 'cab_objective_selected')} | {mean_of(members, 'cab_objective_spectral')} | "
                f"{mean_of(members, 'cab_objective_random')} | {share:.3f} |"
            )
        lines += [
            "",
            "C95 for CAB includes its one-time setup bytes (`c95_bytes`); `c95_training_bytes` in "
            "DCEF_RUNS.csv excludes them.",
        ]
    lines += [
        "",
        "Delivered bytes differ between matched EF and DCEF runs only by the DCEF control packet "
        "and the at most three sparse entries it displaces; see `delta_total_bytes` in DCEF_PAIRS.csv.",
        _evidence_note(cfg),
        "",
    ]
    return "\n".join(lines)


def _evidence_note(cfg: dict[str, Any]) -> str:
    policy = cfg.get("evaluation", {}).get("policy", "official_validation")
    if cfg["project"].get("phase") == "final" and policy == "official_validation":
        return (
            "Confirmation-stage evidence on official validation scores with seeds fixed before "
            "any run of this stage; schedules, learning rates and the basis protocol were frozen "
            "in the development stage."
        )
    return f"Development-stage evidence on {policy} scores; not a confirmation result."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["project"]["output_dir"])
    specs = expected_specs(cfg)
    summaries = {spec: load_summary(output_dir, spec) for spec in specs}
    rows = []
    for spec in specs:
        task, method, seed, ratio = spec
        if ratio is None and method == DENSE_REFERENCE:
            dense = summaries[spec]
        else:
            dense = summaries.get((task, DENSE_REFERENCE, seed, None))
        rows.append(stability_row(spec, summaries[spec], dense))
    pairs = paired_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "DCEF_RUNS.csv", rows)
    write_csv(output_dir / "DCEF_PAIRS.csv", pairs)
    report = render_markdown(cfg, rows, pairs)
    (output_dir / "DCEF_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
