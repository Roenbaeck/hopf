"""Verify K < 0 for specific mixed planes using full numerical Riemann.
Avoids polar regions to rule out coordinate artifacts.
"""
import numpy as np

gamma_val = 0.4
h_fd = 1e-5

def seam_and_derivs(t1, p1, t2, p2):
    st1, ct1 = np.sin(t1), np.cos(t1)
    st2, ct2 = np.sin(t2), np.cos(t2)
    dp = p1 - p2
    cd, sd = np.cos(dp), np.sin(dp)
    s = st1*st2*cd + ct1*ct2
    A = ct1*ct2*cd + st1*st2
    B = ct1*st2*sd
    C = -st1*ct2*sd
    D = st1*st2*cd
    return s, (A, B, C, D)

def metric_analytical(pt):
    t1, p1, t2, p2 = pt
    st1, st2 = np.sin(t1), np.sin(t2)
    s, (A, B, C, D) = seam_and_derivs(t1, p1, t2, p2)
    lam = 1 - gamma_val * s
    g = np.zeros((4, 4))
    g[0, 0] = lam
    g[1, 1] = lam * st1**2
    g[2, 2] = lam
    g[3, 3] = lam * st2**2
    g[0, 2] = g[2, 0] = gamma_val * A
    g[0, 3] = g[3, 0] = gamma_val * B
    g[1, 2] = g[2, 1] = gamma_val * C
    g[1, 3] = g[3, 1] = gamma_val * D
    return g

def christoffel_first(pt):
    G1 = np.zeros((4, 4, 4))
    dg = np.zeros((4, 4, 4))
    for k in range(4):
        ep = pt.copy(); ep[k] += h_fd
        em = pt.copy(); em[k] -= h_fd
        dg[:, :, k] = (metric_analytical(ep) - metric_analytical(em)) / (2 * h_fd)
    for K in range(4):
        for I in range(4):
            for J in range(I, 4):
                val = 0.5 * (dg[K, I, J] + dg[K, J, I] - dg[I, J, K])
                G1[K, I, J] = val
                G1[K, J, I] = val
    return G1

def full_K_general(pt, X_vec, Y_vec):
    """Full Riemann K(X,Y) via finite-difference Christoffel derivatives."""
    g = metric_analytical(pt)
    gi = np.linalg.inv(g)

    def get_Gam_up(p):
        G1 = christoffel_first(p)
        gi_loc = np.linalg.inv(metric_analytical(p))
        return np.einsum('km,mij->kij', gi_loc, G1)

    Gam_up = get_Gam_up(pt)
    dGam = np.zeros((4, 4, 4, 4))
    for d in range(4):
        ep = pt.copy(); ep[d] += h_fd
        em = pt.copy(); em[d] -= h_fd
        dGam[:, :, :, d] = (get_Gam_up(ep) - get_Gam_up(em)) / (2 * h_fd)

    # R^m_{YXY} for R_{XYXY} = g_{Xm} R^m_{YXY}
    # R^m_{jkl} = dGam^m_{jl,k} - dGam^m_{jk,l} + Gam^m_{ke} Gam^e_{jl} - Gam^m_{le} Gam^e_{jk}
    # For K(X,Y): R_{XYXY} with convention that K = R_{XYXY} / |X^Y|^2
    # We need R(e_i, e_j, e_i, e_j) then compose

    # Compute full R_{ijkl} tensor
    R4 = np.zeros((4, 4, 4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    # R^m_{jkl}
                    Rm = np.zeros(4)
                    for m in range(4):
                        Rm[m] = dGam[m, j, l, k] - dGam[m, j, k, l]
                        for e in range(4):
                            Rm[m] += Gam_up[m, k, e]*Gam_up[e, j, l] - Gam_up[m, l, e]*Gam_up[e, j, k]
                    R4[i, j, k, l] = sum(g[i, m]*Rm[m] for m in range(4))

    # R(X,Y,X,Y) = R_{ijkl} X^i Y^j X^k Y^l
    R_XYXY = np.einsum('ijkl,i,j,k,l', R4, X_vec, Y_vec, X_vec, Y_vec)

    gXX = X_vec @ g @ X_vec
    gYY = Y_vec @ g @ Y_vec
    gXY = X_vec @ g @ Y_vec
    denom = gXX * gYY - gXY**2
    return R_XYXY / denom

def quad_K_general(pt, X_vec, Y_vec):
    """K via quadratic formula: R = g^{KM}(V_K V_M - U_K Z_M)."""
    g = metric_analytical(pt)
    gi = np.linalg.inv(g)
    G1 = christoffel_first(pt)

    # V_K = Gamma_{K,XY} = sum Gamma_{K,ij} X^i Y^j
    V = np.einsum('kij,i,j->k', G1, X_vec, Y_vec)
    U = np.einsum('kij,i,j->k', G1, X_vec, X_vec)
    Z = np.einsum('kij,i,j->k', G1, Y_vec, Y_vec)

    R_quad = np.einsum('k,km,m->', V, gi, V) - np.einsum('k,km,m->', U, gi, Z)

    gXX = X_vec @ g @ X_vec
    gYY = Y_vec @ g @ Y_vec
    gXY = X_vec @ g @ Y_vec
    denom = gXX * gYY - gXY**2
    return R_quad / denom

# Test points far from poles
test_pts = [
    ("mid-lat", np.array([1.0, 0.5, 1.2, 0.8])),
    ("equator_region", np.array([np.pi/2, 0.3, np.pi/2 + 0.3, 1.0])),
    ("varied", np.array([0.8, 2.0, 1.8, 0.5])),
]

print("="*80)
print("Verification: Full Riemann vs Quadratic formula for general mixed planes")
print(f"gamma = {gamma_val}")
print("="*80)

for name, pt in test_pts:
    print(f"\n--- Point: {name} = ({pt[0]:.2f},{pt[1]:.2f},{pt[2]:.2f},{pt[3]:.2f}) ---")
    s = seam_and_derivs(*pt)[0]
    g = metric_analytical(pt)
    eigs = np.linalg.eigvalsh(g)
    print(f"  s = {s:.4f}, lam = {1-gamma_val*s:.4f}, metric eigenvalues: {eigs}")

    # Scan over mixed plane angles
    min_K_full = float('inf')
    min_K_quad = float('inf')
    worst_theta, worst_phi = 0, 0

    for theta in np.linspace(0, np.pi, 20):
        for phi in np.linspace(0, np.pi, 20):
            ct, st = np.cos(theta), np.sin(theta)
            cp, sp = np.cos(phi), np.sin(phi)
            X = np.array([ct, st, 0, 0])
            Y = np.array([0, 0, cp, sp])

            gXX = X @ g @ X
            gYY = Y @ g @ Y
            gXY = X @ g @ Y
            denom = gXX * gYY - gXY**2
            if abs(denom) < 1e-10:
                continue

            Kq = quad_K_general(pt, X, Y)
            if Kq < min_K_quad:
                min_K_quad = Kq
                worst_theta, worst_phi = theta, phi

    print(f"  Min K (quad formula) over mixed planes: {min_K_quad:.8f}")

    # Verify the worst-case with full Riemann
    ct, st = np.cos(worst_theta), np.sin(worst_theta)
    cp, sp = np.cos(worst_phi), np.sin(worst_phi)
    X = np.array([ct, st, 0, 0])
    Y = np.array([0, 0, cp, sp])
    K_full = full_K_general(pt, X, Y)
    K_quad = quad_K_general(pt, X, Y)
    print(f"  Full Riemann at worst plane: {K_full:.8f}")
    print(f"  Quad formula at worst plane: {K_quad:.8f}")
    print(f"  Difference: {abs(K_full - K_quad):.2e}")
