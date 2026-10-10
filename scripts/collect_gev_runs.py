"""Collect cab_ef_dev, gev_dev and qnli_dev runs into one CSV (final, last-3 mean, C95 bytes incl. setup)."""
import csv, json, glob, os
B = "/data/home/202412012250"
roots = {
    "cab_ef_dev": B + "/bishe_cab/docs/results/cab_ef_dev",
    "gev_dev": B + "/bishe_gevdev/docs/results/gev_dev",
    "qnli_dev": B + "/bishe_qnli/docs/results/qnli_dev",
}
rows = []
for source, root in roots.items():
    for path in sorted(glob.glob(root + "/*/summary.json")):
        s = json.load(open(path))
        task, method, seed = s["run"]["task"], s["run"]["method"], s["run"]["seed"]
        wire = s["run"].get("wire_ratio")
        curve = [x["primary_metric"] for x in s["rounds"] if x.get("primary_metric") is not None]
        dense_path = f"{root}/{task}__fedavg_lora__seed-{seed}__dense/summary.json"
        if not os.path.exists(dense_path):
            dense_path = f"{roots['cab_ef_dev']}/{task}__fedavg_lora__seed-{seed}__dense/summary.json"
        target = 0.95 * json.load(open(dense_path))["final"]["primary_metric"]
        hit = next((x for x in s["rounds"] if x.get("primary_metric") is not None and x["primary_metric"] >= target), None)
        rows.append({
            "source": source, "task": task, "method": method, "seed": seed, "wire": wire if wire is not None else "dense",
            "final": round(curve[-1], 5), "last3_mean": round(sum(curve[-3:]) / len(curve[-3:]), 5), "best": round(max(curve), 5),
            "c95_target": round(target, 5), "c95_round": hit["round"] if hit else "", "c95_bytes": hit["wire"]["total_delivered_bytes"] if hit else "",
            "setup_bytes": s["final"]["wire"].get("setup_bytes", 0), "total_bytes": s["final"]["wire"]["total_delivered_bytes"],
            "source_hash": s.get("provenance", {}).get("source_hash", s.get("source_hash", "")),
        })
with open("/tmp/GEV_DEV_RUNS.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(len(rows), "rows")
