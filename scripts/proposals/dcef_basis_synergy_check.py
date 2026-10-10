"""Synergy check: delay-consistent EF (DCEF) x communication-aware basis (CAB).

Designed label-skew instance from cab_synthetic_check.py; reports final excess
||BA - Dbar||^2 for bases chosen with different kappa under standard Top-k+EF,
DCEF with oracle gamma, and DCEF with client-reported secant gamma estimates.
"""
import numpy as np, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import cab_synthetic_check as cab
from dcef_synthetic_check import top_k

def run(C, p, eta, tau, k, mode, rounds=600, tail=200):
    """Isotropic local GD in coefficient space. Returns tail mean ||B - cbar||^2 and gamma_hat."""
    m, r = C[0].shape
    cbar = sum(pi*ci for pi, ci in zip(p, C))
    B = np.zeros((m, r)); e_c = [np.zeros((m, r)) for _ in C]; e_s = np.zeros((m, r))
    g_true = 1 - (1-eta)**tau
    prev = [None]*len(C); num = np.zeros(len(C)); den = np.zeros(len(C)); gh = np.full(len(C), 0.5)
    out = []
    for t in range(rounds):
        q = []
        for i, ci in enumerate(C):
            if mode == "topk_ef":
                u = g_true*(ci - B) + e_c[i]
            else:  # dcef: start from virtual iterate
                s = B + e_c[i]
                y = ci + (1-g_true)*(s - ci)
                u = y - B
                d = y - s
                if prev[i] is not None:  # client-side secant LS estimate of gamma (dense local info)
                    ds = s - prev[i][0]; dd = d - prev[i][1]
                    num[i] += -np.sum(dd*ds); den[i] += np.sum(ds*ds)
                    if den[i] > 0: gh[i] = np.clip(num[i]/den[i], 0, 1)
                prev[i] = (s.copy(), d.copy())
            qi = top_k(u, k); e_c[i] = u - qi; q.append(qi)
        g = sum(pi*qi for pi, qi in zip(p, q))
        if mode == "topk_ef":
            v = g + e_s
        elif mode == "dcef_oracle":
            v = g + (1-g_true)*e_s
        else:
            v = g + (1-float(np.dot(p, gh)))*e_s
        down = top_k(v, k); e_s = v - down; B = B + down
        if t >= rounds - tail: out.append(np.linalg.norm(B - cbar)**2)
    return float(np.mean(out)) if np.all(np.isfinite(out)) else np.inf, float(np.dot(p, gh)), g_true

rng = np.random.default_rng(3)
Dbar, D, p, W0 = cab.designed_instance(rng)
m, n = Dbar.shape; r = 4
Sig = cab.sigma_het(D, p)
fmt = lambda v: "     DIVERGED" if (not np.isfinite(v) or v > 1e6) else f"{v:13.4f}"
print("final excess ||BA - Dbar||^2 = bias(basis) + transport floor; designed label-skew instance")
for eta, tau in ((0.1, 4), (0.25, 4)):
    for rho in (0.05, 0.1, 0.2):
        k = max(1, int(round(rho*m*r)))
        print(f"\n eta={eta} tau={tau} gamma={1-(1-eta)**tau:.3f} rho={rho} gamma*d/K={(1-(1-eta)**tau)*m*r/k:.2f}")
        for kap in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
            A = cab.top_eig_rows(Dbar.T@Dbar - kap*Sig, r)[0]
            bias = np.linalg.norm(Dbar@(np.eye(n) - A.T@A))**2
            C = [(Dbar + Di)@A.T for Di in D]
            row = []
            for mode in ("topk_ef", "dcef_oracle", "dcef_est"):
                v, gh, gt = run(C, p, eta, tau, k, mode)
                row.append(f"{mode}={fmt(bias+v)}" + (f" (gamma_hat={gh:.3f})" if mode=="dcef_est" else ""))
            print(f"   basis kappa={kap:4.2f} bias={bias:6.3f} | " + "  ".join(row))
        for name, A in (("random", cab.orth(n, r, np.random.default_rng(0)).T), ("spectral", np.linalg.svd(W0, full_matrices=False)[2][:r])):
            bias = np.linalg.norm(Dbar@(np.eye(n) - A.T@A))**2
            C = [(Dbar + Di)@A.T for Di in D]
            row = [f"{mode}={fmt(bias+run(C, p, eta, tau, k, mode)[0])}" for mode in ("topk_ef", "dcef_oracle", "dcef_est")]
            print(f"   basis {name:10s} bias={bias:6.3f} | " + "  ".join(row))
