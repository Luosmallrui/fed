"""Quick comparison of the Qwen2.5-0.5B MNLI arms (sweep + main comparison, server paths)."""
import json, os
B = "/data/home/202412012250"
S = B + "/bishe_llm/docs/results/"
M = B + "/bishe_llm2/docs/results/"
runs = [
    ("fedavg_lora dense lr1e-4", S + "qwen05b_lr_1e-4/mnli__fedavg_lora__seed-361__dense"),
    ("fedavg_lora dense lr3e-4", S + "qwen05b_lr_3e-4/mnli__fedavg_lora__seed-361__dense"),
    ("random lr1e-4", S + "qwen05b_lr_1e-4/mnli__random_a_ef__seed-361__wire-10"),
    ("random lr3e-4", S + "qwen05b_lr_3e-4/mnli__random_a_ef__seed-361__wire-10"),
    ("random lr1e-3 (chosen)", S + "qwen05b_lr_1e-3/mnli__random_a_ef__seed-361__wire-10"),
    ("cab0 (lr1e-3)", M + "qwen05b_main/mnli__cab0_a_ef__seed-361__wire-10"),
    ("cab0wh +gram", M + "qwen05b_main/mnli__cab0wh_a_ef__seed-361__wire-10"),
    ("gev0 +gram", M + "qwen05b_main/mnli__gev0_a_ef__seed-361__wire-10"),
    ("gev0 no refine", M + "qwen05b_norefine/mnli__gev0_a_ef__seed-361__wire-10"),
]
dense = json.load(open(runs[0][1] + "/summary.json"))
target = 0.95 * dense["final"]["primary_metric"]
print(f"C95 target (0.95 x dense lr1e-4 final {dense['final']['primary_metric']:.4f}) = {target:.4f}")
for label, path in runs:
    p = path + "/summary.json"
    if not os.path.exists(p):
        print(f"  {label:26s} (not finished)")
        continue
    s = json.load(open(p))
    c = [x["primary_metric"] for x in s["rounds"] if x.get("primary_metric") is not None]
    hit = next((x for x in s["rounds"] if x.get("primary_metric") is not None and x["primary_metric"] >= target), None)
    h = f"r{hit['round']} {hit['wire']['total_delivered_bytes']/1e6:.1f}MB" if hit else "-"
    extra = ""
    cab = s.get("cab")
    if cab:
        sm = cab["summary"]
        extra = (f" | setup {s['final']['wire']['setup_bytes']/1e6:.1f}MB energy_sel {sm.get('energy_ratio_selected_mean', float('nan')):.1f}"
                 f" installed {sm.get('energy_ratio_installed_mean', float('nan')):.2f} common_inst {sm.get('common_mode_installed_mean', float('nan')):.3f}")
    print(f"  {label:26s} final {c[-1]:.4f} last3 {sum(c[-3:])/3:.4f} C95 {h:12s} {[round(v,3) for v in c]}{extra}")
