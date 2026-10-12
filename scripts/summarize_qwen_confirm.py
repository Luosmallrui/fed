import json, math, os, statistics
D = "/data/home/202412012250/bishe_llm3/docs/results/qwen05b_confirm/"
seeds = (1001, 1002, 1003)
def load(rid):
    p = D + rid + "/summary.json"
    return json.load(open(p)) if os.path.exists(p) else None
def sign_p(w, l):
    n = w + l
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)
dense = {s: load(f"mnli__fedavg_lora__seed-{s}__dense") for s in seeds}
print("dense fedavg_lora:", " ".join(f"{dense[s]['final']['primary_metric']:.4f}" for s in seeds),
      "mean %.4f" % statistics.fmean(dense[s]["final"]["primary_metric"] for s in seeds))
table = {}
for w in ("05", "10"):
    for m in ("random_a_ef", "cab0_a_ef", "gev0_a_ef"):
        finals, last3s, hits = [], [], []
        for s in seeds:
            r = load(f"mnli__{m}__seed-{s}__wire-{w}")
            c = [x["primary_metric"] for x in r["rounds"] if x.get("primary_metric") is not None]
            finals.append(c[-1]); last3s.append(sum(c[-3:]) / 3)
            target = 0.95 * dense[s]["final"]["primary_metric"]
            hit = next((x for x in r["rounds"] if x.get("primary_metric") is not None and x["primary_metric"] >= target), None)
            hits.append(hit["wire"]["total_delivered_bytes"] / 1e6 if hit else None)
            table[(w, m, s)] = (c[-1], sum(c[-3:]) / 3)
        reach = [h for h in hits if h is not None]
        print(f"{w}% {m:12s} final {statistics.fmean(finals):.4f} ± {statistics.stdev(finals):.4f}  last3 {statistics.fmean(last3s):.4f}  "
              f"C95 {len(reach)}/3" + (f" median {statistics.median(reach):.1f}MB" if reach else "") + "  per-seed " + " ".join(f"{f:.3f}" for f in finals))
for left, right in (("gev0_a_ef", "random_a_ef"), ("cab0_a_ef", "random_a_ef"), ("gev0_a_ef", "cab0_a_ef")):
    for idx, name in ((0, "final"), (1, "last3")):
        d = [table[(w, left, s)][idx] - table[(w, right, s)][idx] for w in ("05", "10") for s in seeds]
        wins, losses = sum(x > 0.003 for x in d), sum(x < -0.003 for x in d)
        print(f"{left} - {right} [{name}]: mean {statistics.fmean(d):+.4f}  W/T/L {wins}/{len(d)-wins-losses}/{losses}  p={sign_p(wins, losses):.2g}  " + " ".join(f"{x:+.3f}" for x in d))
