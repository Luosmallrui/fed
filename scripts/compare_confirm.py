"""Quick per-task comparison of the confirmation runs (server paths); see summarize_confirm.py for the full summary.

Usage: python scripts/compare_confirm.py mnli qnli sst2
"""
import json, os, statistics, sys
D = "/data/home/202412012250/bishe_confirm/docs/results/confirm_main/"
tasks = sys.argv[1:] or ["mnli"]
seeds = (1001, 1002, 1003)
def load(task, method, seed, wire):
    rid = f"{task}__{method}__seed-{seed}__" + ("dense" if wire is None else f"wire-{wire}")
    p = D + rid + "/summary.json"
    return json.load(open(p)) if os.path.exists(p) else None
def c95(s, target):
    hit = next((x for x in s["rounds"] if x.get("primary_metric") is not None and x["primary_metric"] >= target), None)
    return (hit["round"], hit["wire"]["total_delivered_bytes"]) if hit else (None, None)
for task in tasks:
    print(f"#### {task}")
    targets = {}
    print("== dense")
    rows = {}
    for m in ("fedavg_lora", "random_a_dense", "gev0_a_dense"):
        for seed in seeds:
            s = load(task, m, seed, None)
            if s is None:
                continue
            if m == "fedavg_lora":
                targets[seed] = 0.95 * s["final"]["primary_metric"]
            rows.setdefault(m, {})[seed] = s
    for m, per in rows.items():
        finals = [per[x]["final"]["primary_metric"] for x in sorted(per)]
        hits = [c95(per[x], targets[x]) if x in targets else (None, None) for x in sorted(per)]
        cb = " ".join(f"{b/1e6:6.1f}" if b else "   -  " for _, b in hits)
        print(f"  {m:15s} n={len(per)} final " + " ".join(f"{v:.4f}" for v in finals) + f" mean {statistics.fmean(finals):.4f} C95MB {cb}")
    for w in ("05", "10", "20"):
        print(f"== wire {w}")
        table = {}
        for m in ("random_a_ef", "spectral_a_ef", "cab0_a_ef", "cab0wh_a_ef", "gev0_a_ef"):
            for seed in seeds:
                s = load(task, m, seed, w)
                if s is not None:
                    table.setdefault(m, {})[seed] = s
        for m, per in table.items():
            finals = {x: per[x]["final"]["primary_metric"] for x in per}
            hits = {x: c95(per[x], targets[x]) for x in per if x in targets}
            cb = " ".join(f"{hits[x][1]/1e6:6.1f}" if x in hits and hits[x][1] else "   -  " for x in sorted(per))
            line = f"  {m:14s} n={len(per)} final " + " ".join(f"{finals[x]:.4f}" for x in sorted(per)) + f" mean {statistics.fmean(finals.values()):.4f} C95MB {cb}"
            if m != "random_a_ef" and "random_a_ef" in table:
                common = sorted(set(per) & set(table["random_a_ef"]))
                d = [finals[x] - table["random_a_ef"][x]["final"]["primary_metric"] for x in common]
                if d:
                    line += f"  dRand {statistics.fmean(d):+.4f} W/L {sum(v > 0.003 for v in d)}/{sum(v < -0.003 for v in d)} (n={len(d)})"
            print(line)
