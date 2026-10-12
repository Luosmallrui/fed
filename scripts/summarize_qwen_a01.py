"""Qwen2.5-0.5B, MNLI Dirichlet alpha 0.1: curves, C95 and paired gev0 - random differences (seeds 361-363)."""
import json, os, glob
B = "/data/home/202412012250/bishe_llm3/docs/results/"
H = B + "qwen05b_head_mnli_a01/"
A = B + "qwen05b_a01/"
print("head dir:", sorted(os.path.basename(p) for p in glob.glob(H + "mnli__*")))
print("a01 dir:", sorted(os.path.basename(p) for p in glob.glob(A + "mnli__*")))
runs = {}
for seed in (361, 362, 363):
    for m in ("random_a_ef", "gev0_a_ef"):
        for root in (A, H):
            p = root + f"mnli__{m}__seed-{seed}__wire-10/summary.json"
            if os.path.exists(p):
                runs[(m, seed)] = p
                break
dense = {}
for root in (H, A):
    for p in glob.glob(root + "mnli__fedavg_lora__seed-*__dense/summary.json"):
        s = json.load(open(p))
        dense[int(p.split("seed-")[1].split("__")[0])] = s["final"]["primary_metric"]
print("dense", dense)
for (m, seed), p in sorted(runs.items(), key=lambda x: (x[0][1], x[0][0])):
    s = json.load(open(p))
    ev = [(r["round"], r["primary_metric"]) for r in s["rounds"] if r.get("primary_metric") is not None]
    c = [v for _, v in ev]
    first_up = next((r for r, v in ev if v >= 0.45), None)
    setup = s["final"]["wire"].get("setup_bytes", 0) / 1e6
    h = "-"
    if seed in dense:
        t = 0.95 * dense[seed]
        hit = next((r for r in s["rounds"] if r.get("primary_metric") is not None and r["primary_metric"] >= t), None)
        h = "r%d %.1fMB" % (hit["round"], hit["wire"]["total_delivered_bytes"] / 1e6) if hit else "never"
    print(f"{m:12s} s{seed} final {c[-1]:.4f} last3 {sum(c[-3:])/3:.4f} best {max(c):.4f} first>=0.45 r{first_up} setup {setup:.1f}MB C95 {h}")
    print("   ", [(r, round(v, 3)) for r, v in ev])

# Paired differences
P = {("random", 361): H + "mnli__random_a_ef__seed-361__wire-10"}
for s in (361, 362, 363):
    P[("gev0", s)] = A + f"mnli__gev0_a_ef__seed-{s}__wire-10"
    if s != 361:
        P[("random", s)] = A + f"mnli__random_a_ef__seed-{s}__wire-10"
def curve(p):
    s = json.load(open(p + "/summary.json"))
    return [r["primary_metric"] for r in s["rounds"] if r.get("primary_metric") is not None]
for s in (361, 362, 363):
    g, r = curve(P[("gev0", s)]), curve(P[("random", s)])
    print(s, "final %+.4f last3 %+.4f mean-of-curve %+.4f early(r5-15) %+.4f" % (
        g[-1] - r[-1], sum(g[-3:]) / 3 - sum(r[-3:]) / 3, sum(g) / len(g) - sum(r) / len(r), sum(g[:3]) / 3 - sum(r[:3]) / 3))
