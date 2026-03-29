"""Verify the mixed sectional curvature of a conformal product metric
on S^2 x S^2 using numerical finite differences on the full 4D Riemann tensor."""
import numpy as np

alpha = 0.3
h = 1e-5

def s_val(t1, t2):
    return np.cos(t1) + np.cos(t2)

def metric_at(pt):
    t1, p1, t2, p2 = pt
    sv = s_val(t1, t2)
    f = np.exp(2 * alpha * sv)
    g = np.diag([f, f * np.sin(t1)**2, f, f * np.sin(t2)**2])
    return g

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

def sectional_curvature(pt, i, j):
    """K(e_i, e_j) of the 4D metric at pt."""
    Gam = christoffel_at(pt)
    dGam = np.zeros((4, 4, 4, 4))
    for d in range(4):
        ep = pt.copy(); ep[d] += h
        em = pt.copy(); em[d] -= h
        dGam[:, :, :, d] = (christoffel_at(ep) - christoffel_at(em)) / (2 * h)
    # R^m_{j i j}
    Rm = np.zeros(4)
    for m in range(4):
        Rm[m] = dGam[m, j, j, i] - dGam[m, j, i, j]
        for e in range(4):
            Rm[m] += Gam[m, i, e] * Gam[e, j, j] - Gam[m, j, e] * Gam[e, j, i]
    g = metric_at(pt)
    R_ij_ij = sum(g[i, m] * Rm[m] for m in range(4))
    denom = g[i, i] * g[j, j] - g[i, j]**2
    return R_ij_ij / denom

# Test points
tests = [
    ("near poles", np.array([0.3, 0.5, 0.3, 0.7])),
    ("equators", np.array([np.pi/2, 0.5, np.pi/2, 0.7])),
    ("mixed", np.array([0.5, 0.5, 2.0, 0.7])),
    ("south poles", np.array([2.8, 0.5, 2.8, 0.7])),
]

print(f"{'Point':15s} {'4D K':>12s} {'Brioschi K':>12s} {'Diff':>12s}")
print("-" * 55)
for name, pt in tests:
    t1, p1, t2, p2 = pt
    K_4d = sectional_curvature(pt, 0, 2)  # K(dt1, dt2)
    sv = s_val(t1, t2)
    K_br = alpha * sv * np.exp(-2 * alpha * sv)
    print(f"{name:15s} {K_4d:12.8f} {K_br:12.8f} {abs(K_4d - K_br):12.2e}")
