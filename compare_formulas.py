"""Compare old vs new curvature formula against 4D numerical truth."""
import numpy as np

alpha = 0.3
h = 1e-5

def s_val(t1, t2):
    return np.cos(t1) + np.cos(t2)

def metric_at(pt):
    t1, p1, t2, p2 = pt
    sv = s_val(t1, t2)
    f = np.exp(2 * alpha * sv)
    return np.diag([f, f * np.sin(t1)**2, f, f * np.sin(t2)**2])

def christoffel_at(pt):
    dg = np.zeros((4, 4, 4))
    for c in range(4):
        ep = pt.copy(); ep[c] += h
        em = pt.copy(); em[c] -= h
        dg[:, :, c] = (metric_at(ep) - metric_at(em)) / (2 * h)
    gi = np.linalg.inv(metric_at(pt))
    G = np.zeros((4, 4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                G[a, b, c] = 0.5 * sum(
                    gi[a, d] * (dg[d, b, c] + dg[d, c, b] - dg[b, c, d])
                    for d in range(4)
                )
    return G

def sectional_K(pt, i, j):
    Gam = christoffel_at(pt)
    dGam = np.zeros((4, 4, 4, 4))
    for d in range(4):
        ep = pt.copy(); ep[d] += h
        em = pt.copy(); em[d] -= h
        dGam[:, :, :, d] = (christoffel_at(ep) - christoffel_at(em)) / (2 * h)
    Rm = np.zeros(4)
    for m in range(4):
        Rm[m] = dGam[m, j, j, i] - dGam[m, j, i, j]
        for e in range(4):
            Rm[m] += Gam[m, i, e] * Gam[e, j, j] - Gam[m, j, e] * Gam[e, j, i]
    g = metric_at(pt)
    R_ij_ij = sum(g[i, m] * Rm[m] for m in range(4))
    denom = g[i, i] * g[j, j] - g[i, j]**2
    return R_ij_ij / denom

tests = [
    ("near poles",  np.array([0.3, 0.5, 0.3, 0.7])),
    ("equators",    np.array([np.pi/2, 0.5, np.pi/2, 0.7])),
    ("mixed",       np.array([0.5, 0.5, 2.0, 0.7])),
    ("south poles", np.array([2.8, 0.5, 2.8, 0.7])),
]

print(f"{'Point':15s} {'4D (truth)':>14s} {'New formula':>14s} {'Old formula':>14s} {'New err':>10s} {'Old err':>10s}")
print("-" * 82)
for name, pt in tests:
    t1 = pt[0]; t2 = pt[2]
    sv = s_val(t1, t2)
    K_4d = sectional_K(pt, 0, 2)
    # New (Brioschi with u=v cancellation): K = alpha*s / e^{2*alpha*s}
    K_new = alpha * sv * np.exp(-2 * alpha * sv)
    # Old formula: K = [alpha*(cos t1 + cos t2) - 2*alpha^2*(sin^2 t1 + sin^2 t2)] / e^{2*alpha*s}
    K_old = (alpha*(np.cos(t1)+np.cos(t2)) - 2*alpha**2*(np.sin(t1)**2+np.sin(t2)**2)) / np.exp(2*alpha*sv)
    print(f"{name:15s} {K_4d:14.8f} {K_new:14.8f} {K_old:14.8f} {abs(K_4d-K_new):10.2e} {abs(K_4d-K_old):10.2e}")
