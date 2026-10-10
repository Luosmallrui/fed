"""Zero-GPU diagnostic: do existing EF runs show the delay-overshoot signature?

Reads completed run directories (run_config.json + round-NNN.json) produced by
src/fedpeft/engine.py and, for every sparse run, pairs it with the dense
reference of the same task/seed in the same campaign. Nothing is retrained or
modified; output is a CSV plus a short console summary.

Signatures predicted by the delay-consistency analysis (standard EF):
  * post-target violations: after first reaching 95% of the paired dense final
    score, later evaluations fall below it again;
  * oscillation: large round-to-round swings of the primary metric after the hit;
  * residual backlog: ||residual|| / ||per-round update|| grows with the average
    transmission delay d/K, i.e. larger at 10% than at 20% wire, larger for
    full LoRA (d = 294,912) than for fixed-A (d = 147,456);
  * all of the above stronger at 10% than at 20%, and absent in dense runs.

Usage (on the server, from the repository root):
  python3 scripts/proposals/diagnose_ef_overshoot.py docs/results/fedavg_lora_wire \
      docs/results/roberta_screen docs/results/fixed_a_ablation --out docs/proposals/ef_overshoot.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def load_run(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "run_config.json"
    rounds = sorted(run_dir.glob("round-*.json"))
    if not cfg_path.exists() or not rounds:
        return None
    spec = json.loads(cfg_path.read_text())["run"]
    records = [json.loads(p.read_text()) for p in rounds]
    records.sort(key=lambda r: int(r["round"]))
    return {"dir": run_dir, "spec": spec, "records": records}


def evaluated(records: list[dict]) -> list[tuple[int, float]]:
    return [(int(r["round"]), float(r["primary_metric"])) for r in records if r.get("primary_metric") is not None]


def ratio_series(records: list[dict]) -> list[float]:
    out = []
    for r in records:
        d = r.get("diagnostics") or {}
        upd = d.get("adapter_update_norm")
        res = d.get("mean_client_residual_norm")
        srv = d.get("server_residual_norm")
        if upd and res is not None and upd > 0:
            out.append((res + (srv or 0.0)) / upd)
    return out


def slope(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return float("nan")
    xs = list(range(n))
    mx, my = statistics.fmean(xs), statistics.fmean(values)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / den if den else float("nan")


def analyse(run: dict, dense_final: float | None, target_frac: float) -> dict:
    spec = run["spec"]
    ev = evaluated(run["records"])
    scores = [s for _, s in ev]
    row = {
        "task": spec["task"],
        "method": spec["method"],
        "seed": spec["seed"],
        "wire": spec.get("wire_ratio"),
        "n_eval": len(ev),
        "final": scores[-1] if scores else float("nan"),
        "best": max(scores) if scores else float("nan"),
        "dense_final": dense_final if dense_final is not None else float("nan"),
    }
    row["best_minus_final"] = row["best"] - row["final"]
    if dense_final is not None and scores:
        target = target_frac * dense_final
        hit = next((i for i, s in enumerate(scores) if s >= target), None)
        row["target"] = target
        row["first_hit_round"] = ev[hit][0] if hit is not None else ""
        if hit is not None:
            after = scores[hit:]
            row["post_hit_violations"] = sum(s < target for s in after)
            row["post_hit_violation_frac"] = row["post_hit_violations"] / len(after)
            diffs = [abs(b - a) for a, b in zip(after, after[1:])]
            row["post_hit_mean_abs_swing"] = statistics.fmean(diffs) if diffs else 0.0
            row["post_hit_max_drawdown"] = max(
                (max(after[: j + 1]) - after[j] for j in range(len(after))), default=0.0
            )
    ratios = ratio_series(run["records"])
    if ratios:
        tail = ratios[-5:]
        row["residual_to_update_ratio_tail"] = statistics.fmean(tail)
        row["residual_to_update_ratio_slope"] = slope(ratios)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaigns", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/proposals/ef_overshoot.csv"))
    ap.add_argument("--target-frac", type=float, default=0.95)
    args = ap.parse_args()

    runs = []
    for campaign in args.campaigns:
        for run_dir in sorted(p for p in campaign.rglob("run_config.json")):
            run = load_run(run_dir.parent)
            if run is not None:
                runs.append(run)
    if not runs:
        raise SystemExit("no run directories with round-*.json found")

    dense = {}
    for run in runs:
        s = run["spec"]
        if s.get("wire_ratio") is None and s["method"] in ("fedavg_lora", "dense"):
            ev = evaluated(run["records"])
            if ev:
                dense[(s["task"], s["seed"])] = ev[-1][1]

    rows = []
    for run in runs:
        s = run["spec"]
        rows.append(analyse(run, dense.get((s["task"], s["seed"])), args.target_frac))

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("task", "method", "seed", "wire"), k))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def agg(sel, key):
        vals = [r[key] for r in rows if sel(r) and isinstance(r.get(key), (int, float)) and not math.isnan(r[key])]
        return (statistics.fmean(vals), len(vals)) if vals else (float("nan"), 0)

    print(f"wrote {len(rows)} rows to {args.out}")
    groups = sorted({(r["method"], r["wire"]) for r in rows}, key=lambda x: (x[0], x[1] or 0))
    print(f"{'method':22s} {'wire':>5s} | {'viol.frac':>9s} {'swing':>7s} {'drawdown':>8s} {'best-final':>10s} {'res/upd':>8s} {'res/upd slope':>13s}")
    for method, wire in groups:
        sel = lambda r, m=method, w=wire: r["method"] == m and r["wire"] == w
        vf, n = agg(sel, "post_hit_violation_frac")
        sw, _ = agg(sel, "post_hit_mean_abs_swing")
        dd, _ = agg(sel, "post_hit_max_drawdown")
        bf, _ = agg(sel, "best_minus_final")
        ru, _ = agg(sel, "residual_to_update_ratio_tail")
        rs, _ = agg(sel, "residual_to_update_ratio_slope")
        print(f"{method:22s} {str(wire):>5s} | {vf:9.3f} {sw:7.4f} {dd:8.4f} {bf:10.4f} {ru:8.2f} {rs:13.4f}   (n={n})")
    print("prediction under the overshoot mechanism: EF rows at 10% > 20% on viol.frac/swing/drawdown/res-upd; "
          "full-LoRA EF > fixed-A EF at the same wire; dense rows ~0.")


if __name__ == "__main__":
    main()
