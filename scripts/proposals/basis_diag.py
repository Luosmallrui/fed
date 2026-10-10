"""Print setup bytes and zero-training basis diagnostics from every cab_basis.pt under a results root.

Usage: python scripts/proposals/basis_diag.py <results root>
"""
import glob, sys, torch
root = sys.argv[1]
for f in sorted(glob.glob(root + "/*/*/cab_basis.pt")):
    r = torch.load(f, map_location="cpu", weights_only=False)["record"]
    s = r["summary"]
    parts = f.split("/")
    up = sum(r["uplink_factor_bytes_per_client"]) / len(r["uplink_factor_bytes_per_client"]) / 1e3
    act = r.get("activation") or {}
    sk = sum(act.get("uplink_sketch_bytes_per_client", [0])) / max(1, len(act.get("uplink_sketch_bytes_per_client", [1]))) / 1e3
    line = (f"{parts[-3]:12s} {parts[-2][6:]:30s} setup {r['setup_bytes']/1e6:5.2f} MB | per client: grad {up:6.1f} KB"
            f" sketch {sk:6.1f} KB basis {r['downlink_basis_bytes_per_client']/1e3:6.1f} KB")
    if "kfac_capture_selected_mean" in s:
        line += (f" | kfac {s['kfac_capture_selected_mean']:.3f} common {s['common_mode_selected_mean']:.3f}"
                 f" energy {s['energy_ratio_selected_mean']:.2f} ovl_exact {s['overlap_reported_vs_exact_mean']:.3f}")
    print(line)
