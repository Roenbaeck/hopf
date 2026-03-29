"""Verify the feedback's exact formula and K >= 0 for isotropic ell=1 seam.
Uses ANALYTICAL metric computation (no finite differences for metric itself).
"""
import numpy as np

gamma_val = 0.4
h_fd = 1e-5

def seam_and_derivs(t1, p1, t2, p2):
    """Compute s and all needed derivatives analytically."""
    st1, ct1 = np.sin(t1), np.cos(t1)
    st2, ct2 = np.sin(t2), np.cos(t2)
    dp = p1 - p2
    cd, sd = np.cos(dp), np.sin(dp)

    s = st1*st2*cd + ct1*ct2
    # First derivatives
    s_t1 = ct1*st2*cd - st1*ct2
    s_p1 = -st1*st2*sd
    s_t2 = st1*ct2*cd - ct1*st2
    s_p2 = st1*st2*sd

    # Mixed second derivatives (cross-factor = coordinate partials)
    s_t1t2 = ct1*ct2*cd + st1*st2
    s_t1p2 = ct1*st2*sd
    s_p1t2 = -st1*ct2*sd
    s_p1p2 = st1*st2*cd  # note: NOT equal to s in general

    return s, (s_t1, s_p1, s_t2, s_p2), (s_t1t2, s_t1p2, s_p1t2, s_p1p2)

def metric_analytical(t1, p1, t2, p2):
    """Full 4x4 metric g = h + gamma*nabla^2 s, analytical."""
    st1, st2 = np.sin(t1), np.sin(t2)
    s, _, (A, B, C, D) = seam_and_derivs(t1, p1, t2, p2)
    lam = 1 - gamma_val * s

    g = np.zeros((4, 4))
    # Diagonal (conformal on each factor)
    g[0, 0] = lam
    g[1, 1] = lam * st1**2
    g[2, 2] = lam
    g[3, 3] = lam * st2**2
    # Mixed Hessian (cross-factor)
    g[0, 2] = g[2, 0] = gamma_val * A
    g[0, 3] = g[3, 0] = gamma_val * B
    g[1, 2] = g[2, 1] = gamma_val * C
    g[1, 3] = g[3, 1] = gamma_val * D
    return g

def christoffel_first(pt):
    """Christoffel symbols of the first kind via finite differences of analytical metric."""
    G1 = np.zeros((4, 4, 4))
    dg = np.zeros((4, 4, 4))
    for k in range(4):
        ep = pt.copy(); ep[k] += h_fd
        em = pt.copy(); em[k] -= h_fd
        dg[:, :, k] = (metric_analytical(*ep) - metric_analytical(*em)) / (2 * h_fd)
    for K in range(4):
        for I in range(4):
            for J in range(I, 4):
                val = 0.5 * (dg[K, I, J] + dg[K, J, I] - dg[I, J, K])
                G1[K, I, J] = val
                G1[K, J, I] = val
    return G1

def full_riemann_K(pt, i, j):
    """Full sectional curvature K(e_i, e_j) via finite-difference Christoffels."""
    g = metric_analytical(*pt)
    gi = np.linalg.inv(g)

    def get_Gam_up(p):
        G1 = christoffel_first(p)
        gi_loc = np.linalg.inv(metric_analytical(*p))
        return np.einsum('km,mij->kij', gi_loc, G1)

    Gam_up = get_Gam_up(pt)
    dGam = np.zeros((4, 4, 4, 4))
    for d in range(4):
        ep = pt.copy(); ep[d] += h_fd
        em = pt.copy(); em[d] -= h_fd
        dGam[:, :, :, d] = (get_Gam_up(ep) - get_Gam_up(em)) / (2 * h_fd)

    Rm = np.zeros(4)
    for m in range(4):
        Rm[m] = dGam[m, j, j, i] - dGam[m, j, i, j]
        for e in range(4):
            Rm[m] += Gam_up[m, i, e] * Gam_up[e, j, j] - Gam_up[m, j, e] * Gam_up[e, j, i]

    R_ijij = sum(g[i, m] * Rm[m] for m in range(4))
    denom = g[i, i] * g[j, j] - g[i, j]**2
    return R_ijij / denom

def quad_formula_K(pt, i, j):
    """K via quadratic formula: R = g^{KM}(V_K V_M - U_K Z_M)."""
    g = metric_analytical(*pt)
    gi = np.linalg.inv(g)
    G1 = christoffel_first(pt)

    V = G1[:, i, j]
    U = G1[:, i, i]
    Z = G1[:, j, j]

    R_quad = 0.0
    for K in range(4):
        for M in range(4):
            R_quad += gi[K, M] * (V[K]*V[M] - U[K]*Z[M])

    denom = g[i, i] * g[j, j] - g[i, j]**2
    return R_quad / denom

# =============================================
# TEST 1: Exact quadratic formula vs full Riemann
# =============================================
print("="*70)
print("TEST 1: Exact quadratic formula vs full Riemann tensor")
print(f"gamma = {gamma_val}")
print("="*70)
tests = [
    ("generic1", np.array([0.7, 0.3, 1.2, 0.8])),
    ("generic2", np.array([1.0, 1.5, 0.5, 2.1])),
    ("near_pole", np.array([0.2, 0.5, 0.3, 0.7])),
    ("equator", np.array([np.pi/2, 0.5, np.pi/2, 0.7])),
    ("south", np.array([2.5, 0.5, 2.8, 0.7])),
]
print(f"{'Point':12s} {'Full R':>14s} {'Quad R':>14s} {'Diff':>10s}")
print("-" * 55)
for name, pt in tests:
    K_full = full_riemann_K(pt, 0, 2)
    K_quad = quad_formula_K(pt, 0, 2)
    print(f"{name:12s} {K_full:14.8f} {K_quad:14.8f} {abs(K_full-K_quad):10.2e}")

# =============================================
# TEST 2: K >= 0 for all mixed planes?
# =============================================
print()
print("="*70)
print("TEST 2: K >= 0 for all mixed planes (scanning)")
print("="*70)

min_K = float('inf')
min_info = None
n_total = 0
n_negative = 0

for t1 in np.linspace(0.2, np.pi-0.2, 6):
    for t2 in np.linspace(0.2, np.pi-0.2, 6):
        for p1 in [0.0, 1.0]:
            for p2 in [0.0, 1.0]:
                pt = np.array([t1, p1, t2, p2])
                g = metric_analytical(*pt)
                gi = np.linalg.inv(g)
                G1 = christoffel_first(pt)

                for theta in np.linspace(0, np.pi, 10):
                    for phi in np.linspace(0, np.pi, 10):
                        ct, st = np.cos(theta), np.sin(theta)
                        cp, sp = np.cos(phi), np.sin(phi)

                        V = ct*cp*G1[:, 0, 2] + ct*sp*G1[:, 0, 3] + st*cp*G1[:, 1, 2] + st*sp*G1[:, 1, 3]
                        U = ct*ct*G1[:, 0, 0] + 2*ct*st*G1[:, 0, 1] + st*st*G1[:, 1, 1]
                        Z = cp*cp*G1[:, 2, 2] + 2*cp*sp*G1[:, 2, 3] + sp*sp*G1[:, 3, 3]

                        R_num = np.einsum('k,km,m->', V, gi, V) - np.einsum('k,km,m->', U, gi, Z)

                        X_vec = np.array([ct, st, 0, 0])
                        Y_vec = np.array([0, 0, cp, sp])
                        gXX = X_vec @ g @ X_vec
                        gYY = Y_vec @ g @ Y_vec
                        gXY = X_vec @ g @ Y_vec
                        denom = gXX * gYY - gXY**2

                        if abs(denom) < 1e-12:
                            continue

                        K_val = R_num / denom
                        n_total += 1
                        if K_val < min_K:
                            min_K = K_val
                            min_info = (t1, p1, t2, p2, theta, phi)
                        if K_val < -1e-6:
                            n_negative += 1

print(f"Total planes tested: {n_total}")
print(f"Min K found: {min_K:.10e}")
if min_info:
    print(f"  at (t1,p1,t2,p2,theta,phi) = " +
          ",".join(f"{x:.3f}" for x in min_info))
print(f"Planes with K < -1e-6: {n_negative}")
print(f"K >= 0 (within tolerance): {'YES' if n_negative == 0 else 'NO'}")

# =============================================
# TEST 3: Full Riemann for mixed planes check
# =============================================
print()
print("="*70)
print("TEST 3: Full Riemann K for a few mixed planes")
print("="*70)
# At the point with min_K from test 2, compute full K
if min_info:
    pt = np.array(min_info[:4])
    K_full_02 = full_riemann_K(pt, 0, 2)
    K_quad_02 = quad_formula_K(pt, 0, 2)
    print(f"At min-K point, K(e_0,e_2): full={K_full_02:.10f}, quad={K_quad_02:.10f}")
