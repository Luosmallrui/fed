"""Numerical checks for delay-consistent error feedback (DCEF).

Claim (quadratic local objectives, linear local training):
  one round of local training maps the start state s to
      s + Gamma (c_i - s),      Gamma = I - (I - eta H)^tau.
  Among residual rules  e_{t+1} = Lambda (u_t - q_t),  u_t = Delta_t + e_t,
  the unique Lambda for which a coordinate delivered after any gap of T rounds
  lands exactly on the dense-FedAvg trajectory is
      Lambda* = I - Gamma = (I - eta H)^tau.
  Standard EF is Lambda = I (overshoot T*Gamma, unstable when T*gamma > 2);
  no-EF is Lambda = 0.

  Client side, Lambda* is realised without knowing Gamma by starting local
  training from the client's virtual model  x + e_i  and uploading
  (local end state - x).  Server side uses v = g + (I - Gamma_hat) e_s with a
  scalar gamma_hat = sum_i p_i gamma_i, where each client reports a secant
  least-squares estimate gamma_i = -<d_{t+1}-d_t, s_{t+1}-s_t>/||s_{t+1}-s_t||^2
  of its own local contraction (or the oracle Gamma).

Modes compared under bidirectional Top-k (the RoBERTa protocol):
  dense, topk_noef, topk_ef (standard), dcef_oracle, dcef_est.
Pure numpy.
"""

from __future__ import annotations

import numpy as np


def orth(n, k, rng):
    q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return q[:, :k]


def top_k(x, k):
    flat = x.reshape(-1)
    if k >= flat.size:
        return x.copy()
    idx = np.argpartition(np.abs(flat), -k)[-k:]
    out = np.zeros_like(flat)
    out[idx] = flat[idx]
    return out.reshape(x.shape)


def make_instance(rng, m=64, r=8, N=10, het=1.0, aniso=1.0):
    """B-only problem in coefficient space (fixed orthonormal A absorbed).
    Client i: f_i(B) = 0.5 * tr((B - C_i) H (B - C_i)^T), H = r x r SPD.
    aniso = condition number of H (1 = isotropic)."""
    Q = orth(r, r, rng)
    eig = np.geomspace(1.0, 1.0 / aniso, r)
    H = Q @ np.diag(eig) @ Q.T
    Cbar = rng.standard_normal((m, r))
    D = [het * rng.standard_normal((m, r)) for _ in range(N)]
    mean = sum(D) / N
    D = [d - mean for d in D]
    C = [Cbar + d for d in D]
    p = np.full(N, 1.0 / N)
    return C, Cbar, H, p


def local_train(start, Ci, H, eta, tau):
    """Exact tau-step GD on f_i from `start` (rows independent)."""
    Prop = np.linalg.matrix_power(np.eye(H.shape[0]) - eta * H, tau)
    return Ci + (start - Ci) @ Prop.T


def run(C, Cbar, H, p, *, eta, tau, k, mode, rounds, seed=0):
    m, r = Cbar.shape
    Bopt = sum(pi * ci for pi, ci in zip(p, C))  # global optimum (equal-H clients)
    B = np.zeros((m, r))
    e_c = [np.zeros((m, r)) for _ in C]
    e_s = np.zeros((m, r))
    Gam = np.eye(r) - np.linalg.matrix_power(np.eye(r) - eta * H, tau)
    gam_scalar = float(np.trace(Gam) / r)
    prev = None  # per-client (start, displacement) history for the secant estimate
    gam_hat = 0.5
    traj = []
    for t in range(rounds):
        if mode == "dense":
            g = sum(pi * (local_train(B, ci, H, eta, tau) - B) for pi, ci in zip(p, C))
            B = B + g
        elif mode in ("topk_noef", "topk_ef"):
            q = []
            for i, ci in enumerate(C):
                delta = local_train(B, ci, H, eta, tau) - B
                u = delta + (e_c[i] if mode == "topk_ef" else 0.0)
                qi = top_k(u, k)
                if mode == "topk_ef":
                    e_c[i] = u - qi
                q.append(qi)
            g = sum(pi * qi for pi, qi in zip(p, q))
            v = g + (e_s if mode == "topk_ef" else 0.0)
            down = top_k(v, k)
            if mode == "topk_ef":
                e_s = v - down
            B = B + down
        elif mode in ("dcef_oracle", "dcef_est"):
            if prev is None:
                prev = [None] * len(C)
                num = np.zeros(len(C))
                den = np.zeros(len(C))
                gam_i = np.full(len(C), 0.5)
            q = []
            for i, ci in enumerate(C):
                start = B + e_c[i]  # virtual-iterate start
                end = local_train(start, ci, H, eta, tau)
                u = end - B
                # client-side secant least squares: d_{t+1} - d_t = -Gamma (s_{t+1} - s_t),
                # d = end - start; uses dense local vectors, uploads one scalar.
                disp = end - start
                if prev[i] is not None:
                    ds = start - prev[i][0]
                    dd = disp - prev[i][1]
                    num[i] += -np.sum(dd * ds)
                    den[i] += np.sum(ds * ds)
                    if den[i] > 0:
                        gam_i[i] = float(np.clip(num[i] / den[i], 0.0, 1.0))
                prev[i] = (start.copy(), disp.copy())
                qi = top_k(u, k)
                e_c[i] = u - qi
                q.append(qi)
            g = sum(pi * qi for pi, qi in zip(p, q))
            gam_hat = float(np.dot(p, gam_i))
            if mode == "dcef_oracle":
                v = g + e_s - e_s @ Gam.T
            else:
                v = g + (1.0 - gam_hat) * e_s
            down = top_k(v, k)
            e_s = v - down
            B = B + down
        else:
            raise ValueError(mode)
        traj.append(np.linalg.norm(B - Bopt) ** 2 / np.linalg.norm(Bopt) ** 2)
    return np.array(traj), gam_scalar, gam_hat


def summarize(traj, tail=0.25, target=0.05):
    n = len(traj)
    tail_mean = traj[int((1 - tail) * n):].mean()
    hit = np.where(traj <= target)[0]
    first = int(hit[0]) if hit.size else -1
    # retention: fraction of post-hit rounds that stay within target
    retain = float((traj[first:] <= target).mean()) if first >= 0 else 0.0
    return tail_mean, first, retain


def fmt(v):
    return "   DIVERGED" if not np.isfinite(v) or v > 1e6 else f"{v:11.4f}"


if __name__ == "__main__":
    import sys

    verbose = "-v" in sys.argv
    rounds = 400
    modes = ("dense", "topk_noef", "topk_ef", "dcef_oracle", "dcef_est")
    print("relative error ||B - B*||^2/||B*||^2 at the tail; hit = first round <= 0.05; "
          "keep = fraction of rounds after hit that stay <= 0.05")
    summary = {}
    stable_ef, unstable_ef = [], []
    for aniso in (1.0, 20.0):
        for het in (0.0, 1.0):
            if verbose:
                print(f"\n=== H condition number={aniso:g}, heterogeneity={het:g}")
            for rho in (0.03, 0.06, 0.1, 0.2):
                for eta, tau in ((0.3, 8), (0.1, 8), (0.03, 8)):
                    rng = np.random.default_rng(0)
                    C, Cbar, H, p = make_instance(rng, het=het, aniso=aniso)
                    m, r = Cbar.shape
                    k = max(1, int(round(rho * m * r)))
                    cells = []
                    for mode in modes:
                        traj, g, gh = run(C, Cbar, H, p, eta=eta, tau=tau, k=k, mode=mode, rounds=rounds)
                        tm, first, keep = summarize(traj)
                        diverged = (not np.isfinite(tm)) or tm > 1e6
                        summary.setdefault((het, mode), []).append((diverged, tm, keep))
                        if mode == "topk_ef" and het == 0.0:
                            (unstable_ef if (diverged or tm > 1.0) else stable_ef).append(g / rho)
                        cells.append(f"{mode}:{fmt(tm)} hit={first:4d} keep={keep:4.2f}")
                    if verbose:
                        print(f"rho={rho:4.2f} gamma={g:5.3f} gamma*d/K={g/rho:6.2f} gamma_hat={gh:5.3f}\n   "
                              + "\n   ".join(cells))
    print("\nsummary over 2 curvature settings x 4 budgets x 3 step sizes = 24 settings per heterogeneity level")
    for het in (0.0, 1.0):
        print(f" heterogeneity={het:g}")
        for mode in modes:
            rows = summary[(het, mode)]
            div = sum(d for d, _, _ in rows)
            finite = [tm for d, tm, _ in rows if not d]
            keep = [kp for d, _, kp in rows if not d]
            print(f"   {mode:12s} diverged {div:2d}/24 | median tail error (non-diverged) {np.median(finite):.4f} "
                  f"| median keep {np.median(keep):.2f}")
    print(f" standard EF (homogeneous): largest stable gamma*d/K = {max(stable_ef):.2f}, "
          f"smallest unstable gamma*d/K = {min(unstable_ef):.2f}  (theory: 2)")
