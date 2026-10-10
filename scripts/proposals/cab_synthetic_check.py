"""Numerical checks for the communication-aware fixed basis (CAB) derivation.

Model: fixed row-orthonormal A (r x n), B-only federated LoRA, client losses
f_i(W) = 0.5 * ||W - T_i||_F^2 with T_i = W0 + Dbar + D_i and sum_i p_i D_i = 0.
Each round, clients start from the communicated B, run tau local GD steps
(lr eta) on B, and upload a compressed Delta B_i.

Closed form under test (stationary regime, unbiased omega-compressor):
    J(P)   = ||Dbar (I-P)||^2 + kappa * tr(P Sigma_het)
    kappa  = gamma (omega/Neff) / (2 - gamma (1 + omega/Neff)),
    gamma  = 1 - (1-eta)^tau,  Neff = 1 / sum_i p_i^2,
    Sigma_het = Neff * sum_i p_i^2 D_i^T D_i,
    A*     = top-r eigenvectors of M = Dbar^T Dbar - kappa * Sigma_het.

Checks:
  1. one-round expectation identity (Monte Carlo, unbiased rand-k)
  2. Ky Fan optimality of A* (vs random and Stiefel local search)
  3. stationary noise floor formula (simulation, unbiased rand-k)
  4. basis comparison under rand-k and under Top-k + bidirectional EF
     (the protocol used in the RoBERTa experiments), on a designed
     scenario and on random instances.

Pure numpy; no repository code or data is touched.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def orth(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return q[:, :k]


def top_eig_rows(M: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    idx = np.argsort(w)[::-1][:r]
    return V[:, idx].T.copy(), w[idx]


def sigma_het(D: list[np.ndarray], p: np.ndarray) -> np.ndarray:
    neff = 1.0 / np.sum(p**2)
    return neff * sum(pi**2 * Di.T @ Di for pi, Di in zip(p, D))


def kappa_theory(gamma: float, omega: float, neff: float) -> float:
    a = omega / neff
    denom = 2.0 - gamma * (1.0 + a)
    if denom <= 0:
        return np.inf
    return gamma * a / denom


def J_theory(A: np.ndarray, Dbar: np.ndarray, Sig: np.ndarray, kappa: float) -> float:
    P = A.T @ A
    n = P.shape[0]
    bias = np.linalg.norm(Dbar @ (np.eye(n) - P)) ** 2
    return bias + kappa * np.trace(P @ Sig)


def rand_k(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    flat = x.reshape(-1)
    d = flat.size
    idx = rng.choice(d, size=k, replace=False)
    out = np.zeros_like(flat)
    out[idx] = flat[idx] * (d / k)
    return out.reshape(x.shape)


def top_k(x: np.ndarray, k: int) -> np.ndarray:
    flat = x.reshape(-1)
    if k >= flat.size:
        return x.copy()
    idx = np.argpartition(np.abs(flat), -k)[-k:]
    out = np.zeros_like(flat)
    out[idx] = flat[idx]
    return out.reshape(x.shape)


# ----------------------------------------------------------------------------
# federated simulation (B-only, fixed A)
# ----------------------------------------------------------------------------

def simulate(
    A: np.ndarray,
    Dbar: np.ndarray,
    D: list[np.ndarray],
    p: np.ndarray,
    *,
    eta: float,
    tau: int,
    k: int,
    mode: str,
    rounds: int,
    tail: int,
    seed: int,
) -> float:
    """Return mean excess error ||B A - Dbar||^2 over the last `tail` rounds."""
    rng = np.random.default_rng(seed)
    m = Dbar.shape[0]
    r = A.shape[0]
    d = m * r
    c = [(Dbar + Di) @ A.T for Di in D]  # client in-subspace optima
    B = np.zeros((m, r))
    e_client = [np.zeros((m, r)) for _ in D]
    e_server = np.zeros((m, r))
    gamma = 1.0 - (1.0 - eta) ** tau
    acc = []
    for t in range(rounds):
        deltas = [gamma * (ci - B) for ci in c]  # exact tau-step GD on B
        if mode == "dense" or k >= d:
            B = B + sum(pi * di for pi, di in zip(p, deltas))
        elif mode == "randk":
            B = B + sum(pi * rand_k(di, k, rng) for pi, di in zip(p, deltas))
        elif mode == "topk_ef":
            q = []
            for i, di in enumerate(deltas):
                u = di + e_client[i]
                qi = top_k(u, k)
                e_client[i] = u - qi
                q.append(qi)
            g = sum(pi * qi for pi, qi in zip(p, q))
            v = g + e_server
            down = top_k(v, k)
            e_server = v - down
            B = B + down
        elif mode == "dcef":
            # delay-consistent EF: clients start local training from B + e_i,
            # server discounts its residual by (1 - gamma)
            q = []
            for i, ci in enumerate(c):
                start = B + e_client[i]
                u = start + gamma * (ci - start) - B
                qi = top_k(u, k)
                e_client[i] = u - qi
                q.append(qi)
            g = sum(pi * qi for pi, qi in zip(p, q))
            v = g + (1.0 - gamma) * e_server
            down = top_k(v, k)
            e_server = v - down
            B = B + down
        elif mode == "topk_noef":
            g = sum(pi * top_k(di, k) for pi, di in zip(p, deltas))
            B = B + top_k(g, k)
        else:
            raise ValueError(mode)
        if t >= rounds - tail:
            acc.append(np.linalg.norm(B @ A - Dbar) ** 2)
    return float(np.mean(acc))


# ----------------------------------------------------------------------------
# problem instances
# ----------------------------------------------------------------------------

def designed_instance(rng: np.random.Generator, n=64, m=64, N=10, het=1.5):
    """Flat consensus spectrum; strongest consensus directions carry label-skew-like
    client disagreement (clients push +/-het along v1, v2), plus pure-noise
    heterogeneous directions."""
    V = orth(n, n, rng)
    U = orth(m, m, rng)
    sig = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65])
    Dbar = sum(sig[j] * np.outer(U[:, j], V[:, j]) for j in range(len(sig)))
    signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    D = []
    for i in range(N):
        Di = np.zeros((m, n))
        for j in (0, 1):
            Di += het * signs[i] * rng.uniform(0.7, 1.3) * np.outer(U[:, 20 + j], V[:, j])
        for j in range(10, 14):
            Di += 0.6 * rng.standard_normal() * np.outer(U[:, 30 + j], V[:, j])
        D.append(Di)
    mean = sum(D) / N
    D = [Di - mean for Di in D]
    # pretrained weight whose top right-singular directions are only partly task-aligned
    Vw = np.column_stack([V[:, 0], V[:, 10], V[:, 40], V[:, 41]])
    W0 = U[:, :4] @ np.diag([3.0, 2.5, 2.0, 1.5]) @ Vw.T + 0.1 * rng.standard_normal((m, n))
    p = np.full(N, 1.0 / N)
    return Dbar, D, p, W0


def random_instance(rng: np.random.Generator, n=48, m=48, N=10):
    """Random consensus spectrum and random heterogeneity geometry."""
    rc = int(rng.integers(6, 14))
    U = orth(m, rc, rng)
    V = orth(n, rc, rng)
    sig = np.sort(rng.uniform(0.3, 1.0, rc))[::-1]
    Dbar = U @ np.diag(sig) @ V.T
    rh = int(rng.integers(2, 8))
    # heterogeneity directions: mixture of consensus and fresh directions
    mix = rng.uniform(0, 1)
    Vh = orth(n, rh, rng)
    take = min(rh, rc)
    Vh[:, :take] = mix * V[:, :take] + (1 - mix) * Vh[:, :take]
    Vh, _ = np.linalg.qr(Vh)
    scale = rng.uniform(0.3, 2.0)
    D = []
    for _ in range(N):
        Uh = rng.standard_normal((m, rh)) / np.sqrt(m)
        D.append(scale * Uh @ np.diag(rng.uniform(0.5, 1.5, rh)) @ Vh.T * np.sqrt(m))
    mean = sum(D) / N
    D = [Di - mean for Di in D]
    W0 = rng.standard_normal((m, n))
    p = np.full(N, 1.0 / N)
    return Dbar, D, p, W0


def bases(Dbar, D, p, W0, r, kappa, rng):
    Sig = sigma_het(D, p)
    n = Dbar.shape[1]
    out = {
        "random": orth(n, r, rng).T,
        "spectral(W0)": np.linalg.svd(W0, full_matrices=False)[2][:r],
        "consensus-PCA (k=0)": top_eig_rows(Dbar.T @ Dbar, r)[0],
        "CAB (k=theory)": top_eig_rows(Dbar.T @ Dbar - kappa * Sig, r)[0],
    }
    return out, Sig


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------

def check_one_round_identity(rng):
    m = n = 32
    r, N, k = 4, 6, 20
    U = [rng.standard_normal((m, n)) for _ in range(N)]
    p = rng.dirichlet(np.ones(N))
    Ubar = sum(pi * Ui for pi, Ui in zip(p, U))
    A = orth(n, r, rng).T
    P = A.T @ A
    d = m * r
    omega = d / k - 1
    pred = np.linalg.norm(Ubar @ (np.eye(n) - P)) ** 2 + omega * sum(
        pi**2 * np.trace(P @ Ui.T @ Ui) for pi, Ui in zip(p, U)
    )
    trials = 20000
    vals = np.empty(trials)
    for t in range(trials):
        dW = sum(pi * rand_k(Ui @ A.T, k, rng) for pi, Ui in zip(p, U)) @ A
        vals[t] = np.linalg.norm(dW - Ubar) ** 2
    se = vals.std() / np.sqrt(trials)
    print(f"[1] one-round identity: predicted={pred:.4f}  monte-carlo={vals.mean():.4f} +/- {1.96*se:.4f}")


def check_ky_fan(rng):
    n, r = 40, 5
    X = rng.standard_normal((n, n))
    S = X @ X.T
    Y = rng.standard_normal((n, 3 * n))
    H = Y @ Y.T / n
    M = S - 0.7 * H
    A_star, w = top_eig_rows(M, r)
    best = np.trace(A_star @ M @ A_star.T)
    rand_best = max(np.trace(Q.T @ M @ Q) for Q in (orth(n, r, rng) for _ in range(5000)))
    # Riemannian gradient ascent on the Stiefel manifold from random starts
    local_best = -np.inf
    for _ in range(20):
        Q = orth(n, r, rng)
        for _ in range(3000):
            G = 2 * M @ Q
            Q, _ = np.linalg.qr(Q + 1e-3 * (G - Q @ (Q.T @ G)))
        local_best = max(local_best, np.trace(Q.T @ M @ Q))
    print(
        f"[2] Ky Fan: closed-form={best:.4f}  best-of-5000-random={rand_best:.4f}  "
        f"best-of-20-Stiefel-ascent={local_best:.4f}  sum(top-r eig)={w.sum():.4f}"
    )


def check_floor(rng):
    Dbar, D, p, _ = designed_instance(rng)
    m = Dbar.shape[0]
    r = 4
    A = orth(Dbar.shape[1], r, rng).T
    eta, tau = 0.1, 4
    gamma = 1 - (1 - eta) ** tau
    neff = 1 / np.sum(p**2)
    Sig = sigma_het(D, p)
    print("[3] stationary floor (unbiased rand-k):")
    for k in (13, 26, 51):
        omega = m * r / k - 1
        kap = kappa_theory(gamma, omega, neff)
        pred = J_theory(A, Dbar, Sig, kap)
        sims = [
            simulate(A, Dbar, D, p, eta=eta, tau=tau, k=k, mode="randk", rounds=3000, tail=2500, seed=s)
            for s in range(4)
        ]
        print(f"    K={k:3d} omega={omega:6.2f} kappa={kap:.3f}  J_theory={pred:.4f}  J_sim={np.mean(sims):.4f} +/- {np.std(sims):.4f}")


def compare_bases(rng, instance_fn, label, r=4, eta=0.1, tau=4, ks=(13, 26, 51, 10**9), seeds=4, rounds=600, tail=300):
    Dbar, D, p, W0 = instance_fn(rng)
    m = Dbar.shape[0]
    gamma = 1 - (1 - eta) ** tau
    neff = 1 / np.sum(p**2)
    print(f"[4] basis comparison - {label}")
    for k in ks:
        d = m * r
        kk = min(k, d)
        omega = d / kk - 1
        kap = kappa_theory(gamma, omega, neff)
        B, Sig = bases(Dbar, D, p, W0, r, kap, np.random.default_rng(123))
        modes = ("dense",) if kk >= d else ("randk", "topk_ef", "topk_noef")
        for mode in modes:
            row = []
            for name, A in B.items():
                v = np.mean([
                    simulate(A, Dbar, D, p, eta=eta, tau=tau, k=kk, mode=mode, rounds=rounds, tail=tail, seed=s)
                    for s in range(seeds)
                ])
                row.append(f"{name}={v:.3f}")
            print(f"    rho={kk/d:5.3f} kappa={kap:5.3f} {mode:9s} | " + "  ".join(row))


def kappa_sweep(rng, instance_fn, r=4, eta=0.1, tau=4, k=26, seeds=4, rounds=600, tail=300):
    Dbar, D, p, W0 = instance_fn(rng)
    m = Dbar.shape[0]
    gamma = 1 - (1 - eta) ** tau
    neff = 1 / np.sum(p**2)
    omega = m * r / k - 1
    kth = kappa_theory(gamma, omega, neff)
    Sig = sigma_het(D, p)
    print(f"[5] kappa sweep at rho={k/(m*r):.3f} (theory kappa={kth:.3f}) - Top-k + bidirectional EF")
    for kap in (0.0, 0.25 * kth, 0.5 * kth, kth, 2 * kth, 4 * kth, 8 * kth):
        A = top_eig_rows(Dbar.T @ Dbar - kap * Sig, r)[0]
        v = np.mean([simulate(A, Dbar, D, p, eta=eta, tau=tau, k=k, mode="topk_ef", rounds=rounds, tail=tail, seed=s) for s in range(seeds)])
        print(f"    kappa={kap:6.3f}  final excess={v:.4f}")


def random_instances(n_inst=60, r=4, eta=0.1, tau=4, rho=0.1, seeds=2, rounds=500, tail=250, mode="topk_ef"):
    """Across random instances: does J_theory rank bases correctly, and does CAB
    beat consensus-PCA under the given synchronisation mode?"""
    wins = ties = losses = 0
    rel = []
    corr_x, corr_y = [], []
    for inst in range(n_inst):
        rng = np.random.default_rng(1000 + inst)
        Dbar, D, p, W0 = random_instance(rng)
        m = Dbar.shape[0]
        d = m * r
        k = max(1, int(round(rho * d)))
        gamma = 1 - (1 - eta) ** tau
        neff = 1 / np.sum(p**2)
        kap = kappa_theory(gamma, d / k - 1, neff)
        B, Sig = bases(Dbar, D, p, W0, r, kap, np.random.default_rng(inst))
        sims = {}
        for name, A in B.items():
            sims[name] = np.mean([simulate(A, Dbar, D, p, eta=eta, tau=tau, k=k, mode=mode, rounds=rounds, tail=tail, seed=s) for s in range(seeds)])
            corr_x.append(J_theory(A, Dbar, Sig, kap))
            corr_y.append(sims[name])
        a, b = sims["CAB (k=theory)"], sims["consensus-PCA (k=0)"]
        if abs(a - b) <= 1e-3 * max(b, 1e-12):
            ties += 1
        elif a < b:
            wins += 1
        else:
            losses += 1
        rel.append((b - a) / b)
        others = min(sims["random"], sims["spectral(W0)"])
        rel_other = (others - a) / others
        rel[-1] = (rel[-1], rel_other)
    rel = np.array(rel)
    rx = np.argsort(np.argsort(corr_x))
    ry = np.argsort(np.argsort(corr_y))
    spearman = np.corrcoef(rx, ry)[0, 1]
    print(f"[6] {n_inst} random instances, rho={rho}, gamma={1-(1-eta)**tau:.3f}, mode={mode}")
    print(f"    CAB vs consensus-PCA: wins={wins} ties={ties} losses={losses}; "
          f"median rel. improvement={np.median(rel[:,0]):+.3%}, worst={rel[:,0].min():+.3%}")
    print(f"    CAB vs best(random, spectral): median rel. improvement={np.median(rel[:,1]):+.3%}, worst={rel[:,1].min():+.3%}")
    print(f"    Spearman(J_theory, simulated Top-k+EF final excess) over all bases = {spearman:.3f}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    check_one_round_identity(np.random.default_rng(0))
    check_ky_fan(np.random.default_rng(1))
    check_floor(np.random.default_rng(2))
    compare_bases(np.random.default_rng(3), designed_instance, "designed scenario (label-skew on top consensus directions)")
    kappa_sweep(np.random.default_rng(3), designed_instance)
    print("    note: standard Top-k+EF diverges above gamma*d/K ~ 2.6 for every basis; see dcef_synthetic_check.py")
    # standard EF only in its stable regime (gamma*d/K < 2), so EF divergence does not confound the basis effect
    random_instances(eta=0.03, rho=0.1, mode="topk_ef")
    # delay-consistent EF at an aggressive budget/step where standard EF diverges
    random_instances(eta=0.1, rho=0.05, mode="dcef")
    random_instances(eta=0.1, rho=0.1, mode="dcef")
