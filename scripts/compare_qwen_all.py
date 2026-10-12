import json, os
B = "/data/home/202412012250/"
S = B + "bishe_llm2/docs/results/qwen05b_supp/"
runs = {
    ("10%", "random", 361): B + "bishe_llm/docs/results/qwen05b_lr_1e-3/mnli__random_a_ef__seed-361__wire-10",
    ("10%", "gev0", 361): B + "bishe_llm2/docs/results/qwen05b_main/mnli__gev0_a_ef__seed-361__wire-10",
    ("10%", "random", 362): S + "mnli__random_a_ef__seed-362__wire-10",
    ("10%", "gev0", 362): S + "mnli__gev0_a_ef__seed-362__wire-10",
    ("10%", "random", 363): S + "mnli__random_a_ef__seed-363__wire-10",
    ("10%", "gev0", 363): S + "mnli__gev0_a_ef__seed-363__wire-10",
    ("5%", "random", 361): S + "mnli__random_a_ef__seed-361__wire-05",
    ("5%", "gev0", 361): S + "mnli__gev0_a_ef__seed-361__wire-05",
    ("10%", "gev0 L3 (4.8MB)", 361): B + "bishe_llm3/docs/results/qwen05b_cost_L3/mnli__gev0_a_ef__seed-361__wire-10",
    ("10%", "gev0 L1", 361): B + "bishe_llm3/docs/results/qwen05b_cost_L1/mnli__gev0_a_ef__seed-361__wire-10",
    ("10%", "gev0 L2", 361): B + "bishe_llm3/docs/results/qwen05b_cost_L2/mnli__gev0_a_ef__seed-361__wire-10",
}
dense_paths = {361: B + "bishe_llm/docs/results/qwen05b_lr_1e-4/mnli__fedavg_lora__seed-361__dense",
               362: S + "mnli__fedavg_lora__seed-362__dense", 363: S + "mnli__fedavg_lora__seed-363__dense"}
dense = {k: json.load(open(v + "/summary.json"))["final"]["primary_metric"] for k, v in dense_paths.items() if os.path.exists(v + "/summary.json")}
for (wire, method, seed), path in runs.items():
    p = path + "/summary.json"
    if not os.path.exists(p):
        print(f"{wire:4s} {method:16s} s{seed} (not done)")
        continue
    s = json.load(open(p))
    c = [r["primary_metric"] for r in s["rounds"] if r.get("primary_metric") is not None]
    setup = s["final"]["wire"].get("setup_bytes", 0) / 1e6
    h = "-"
    if seed in dense:
        target = 0.95 * dense[seed]
        hit = next((r for r in s["rounds"] if r.get("primary_metric") is not None and r["primary_metric"] >= target), None)
        h = "r%d %.1fMB" % (hit["round"], hit["wire"]["total_delivered_bytes"] / 1e6) if hit else "never"
    print(f"{wire:4s} {method:16s} s{seed} final {c[-1]:.4f} last3 {sum(c[-3:])/3:.4f} setup {setup:5.1f}MB C95 {h:12s} {[round(v, 3) for v in c]}")
print("dense finals", {k: round(v, 4) for k, v in dense.items()})
