import json, os
B = "/data/home/202412012250/"
L1d, L2d, L3d = B + "bishe_llm/docs/results/", B + "bishe_llm2/docs/results/", B + "bishe_llm3/docs/results/"
def load(path):
    p = path + "/summary.json"
    return json.load(open(p)) if os.path.exists(p) else None
dense361 = load(L1d + "qwen05b_lr_1e-4/mnli__fedavg_lora__seed-361__dense")["final"]["primary_metric"]
target = 0.95 * dense361
def row(label, path):
    s = load(path)
    if s is None:
        print(f"  {label:34s} (missing)"); return
    c = [r["primary_metric"] for r in s["rounds"] if r.get("primary_metric") is not None]
    hit = next((r for r in s["rounds"] if r.get("primary_metric") is not None and r["primary_metric"] >= target), None)
    h = "r%d %.1fMB" % (hit["round"], hit["wire"]["total_delivered_bytes"] / 1e6) if hit else "never"
    setup = s["final"]["wire"].get("setup_bytes", 0) / 1e6
    extra = ""
    cab = s.get("cab")
    if cab and "kfac_capture_selected_mean" in cab["summary"]:
        sm = cab["summary"]
        extra = " | kfac %.3f ovl_exact %.3f energy_inst %.2f" % (sm["kfac_capture_selected_mean"], sm["overlap_reported_vs_exact_mean"], sm.get("energy_ratio_installed_mean", float("nan")))
    print(f"  {label:34s} final {c[-1]:.4f} last3 {sum(c[-3:])/3:.4f} r5 {c[0]:.3f} setup {setup:5.1f}MB C95 {h:12s}{extra}")
print(f"seed 361 dense FedAvg final {dense361:.4f}; C95 target {target:.4f}")
print("== rank 8, 10%, seed 361: random vs gev0 vs oracle")
row("random lr1e-3", L1d + "qwen05b_lr_1e-3/mnli__random_a_ef__seed-361__wire-10")
row("gev0 lr1e-3 (gram)", L2d + "qwen05b_main/mnli__gev0_a_ef__seed-361__wire-10")
row("gev0 oracle lr1e-3", L3d + "qwen05b_oracle/mnli__gev0x_a_ef__seed-361__wire-10")
row("gev0 no refine", L2d + "qwen05b_norefine/mnli__gev0_a_ef__seed-361__wire-10")
row("cab0wh (gram)", L2d + "qwen05b_main/mnli__cab0wh_a_ef__seed-361__wire-10")
row("cab0", L2d + "qwen05b_main/mnli__cab0_a_ef__seed-361__wire-10")
print("== learning-rate grid (10%, seed 361)")
for lr in ("1e-4", "3e-4", "1e-3", "3e-3"):
    rp = L2d + "qwen05b_lr_3e-3/mnli__random_a_ef__seed-361__wire-10" if lr == "3e-3" else L1d + f"qwen05b_lr_{lr}/mnli__random_a_ef__seed-361__wire-10"
    gp = L2d + "qwen05b_main/mnli__gev0_a_ef__seed-361__wire-10" if lr == "1e-3" else L3d + f"qwen05b_gev0_lr_{lr}/mnli__gev0_a_ef__seed-361__wire-10"
    row(f"random lr{lr}", rp)
    row(f"gev0   lr{lr}", gp)
print("== rank 2 (alpha 4), 10%, seed 361")
row("random r2", L3d + "qwen05b_r2/mnli__random_a_ef__seed-361__wire-10")
row("gev0 r2", L3d + "qwen05b_r2/mnli__gev0_a_ef__seed-361__wire-10")
print("== setup-cost variants (gev0, 10%, seed 361)")
for v in ("L1", "L2", "L3"):
    row(f"gev0 cost {v}", L3d + f"qwen05b_cost_{v}/mnli__gev0_a_ef__seed-361__wire-10")
print("== 5%, seed 361")
row("random 5%", L2d + "qwen05b_supp/mnli__random_a_ef__seed-361__wire-05")
row("gev0 5%", L2d + "qwen05b_supp/mnli__gev0_a_ef__seed-361__wire-05")
row("cab0wh 5%", L2d + "qwen05b_supp/mnli__cab0wh_a_ef__seed-361__wire-05")
print("== dense references", {s: round(load(L2d + f"qwen05b_supp/mnli__fedavg_lora__seed-{s}__dense")["final"]["primary_metric"], 4) for s in (362, 363) if load(L2d + f"qwen05b_supp/mnli__fedavg_lora__seed-{s}__dense")})
