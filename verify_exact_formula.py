"""Verify the feedback's exact formula and K >= 0 for isotropic ell=1 seam.

Tests:
1. Does R_XYXY = g^{KM}(V_K V_M - U_K Z_M) hold EXACTLY?
2. Is K_g(X,Y) >= 0 for ALL mixed planes?
"""
import numpy as np

gamma = 0.4  # large gamma to test non-perturbatively
h_fd = 1e-5   # finite-difference step

def seam(pt):
    """Isotropic seam s = n1 . n2 on S^2 x S^2."""
    t1, p1, t2, p2 = pt
    x1 = np.array([np.sin(t1)*np.cos(p1), np.sin(t1)*np.sin(p1), np.cos(t1)])
    x2 = np.array([np.sin(t2)*np.cos(p2), np.sin(t2)*np.sin(p2), np.cos(t2)])
    return x1 @ x2

def metric_at(pt):
    """4x4 metric g = h + gamma * nabla^2 s in coords (t1,p1,t2,p2)."""
    t1, p1, t2, p2 = pt
    sv = seam(pt)
    lam = 1 - gamma * sv  # conformal factor
    # Background metric h = diag(1, sin^2 t1, 1, sin^2 t2)
    # g|_factor = (1 - gamma*s) * h_k  (conformal on each factor)
    # g_{a alpha} = gamma * s_{;a alpha}  (mixed Hessian)
    # Need mixed Hessian numerically
    g = np.zeros((4, 4))
    g[0, 0] = lam
    g[1, 1] = lam * np.sin(t1)**2
    g[2, 2] = lam
    g[3, 3] = lam * np.sin(t2)**2
    # Mixed Hessian: compute numerically
    for i in range(2):
        for j in range(2, 4):
            ep_i = pt.copy(); ep_i[i] += h_fd
            em_i = pt.copy(); em_i[i] -= h_fd
            ep_j = pt.copy(); ep_j[j] += h_fd
            em_j = pt.copy(); em_j[j] -= h_fd
            ep_ij = pt.copy(); ep_ij[i] += h_fd; ep_ij[j] += h_fd
            em_ij = pt.copy(); em_ij[i] -= h_fd; em_ij[j] -= h_fd
            ep_i_em_j = pt.copy(); ep_i_em_j[i] += h_fd; ep_i_em_j[j] -= h_fd
            em_i_ep_j = pt.copy(); em_i_ep_j[i] -= h_fd; em_i_ep_j[j] += h_fd
            # Mixed partial of s
            s_ij = (seam(ep_ij) - seam(ep_i_em_j) - seam(em_i_ep_j) + seam(em_ij)) / (4*h_fd**2)
            # But we need covariant Hessian, not coordinate Hessian
            # For now use full numerical metric
            pass
    # Actually, let's just build the full metric numerically
    return _full_metric(pt)

def _full_metric(pt):
    """Build full 4x4 metric via finite differences of s."""
    t1, p1, t2, p2 = pt
    # g_IJ = h_IJ + gamma * s_{;IJ}
    # Use numerical second derivatives of s in coordinates,
    # then subtract Christoffel corrections for covariant Hessian.
    # Simpler: use the fact that g_IJ = h_IJ + gamma * (s_{,IJ} - Gamma^K_{IJ} s_{,K})
    # where Gamma^K_{IJ} are background Christoffel symbols.
    
    # Background h and its Christoffels
    st1, ct1 = np.sin(t1), np.cos(t1)
    st2, ct2 = np.sin(t2), np.cos(t2)
    
    h = np.diag([1.0, st1**2, 1.0, st2**2])
    
    # Background Christoffel symbols (S^2 x S^2)
    # Only nonzero: Gamma^0_{11} = -st1*ct1, Gamma^1_{01} = Gamma^1_{10} = ct1/st1
    # and similarly for factor 2
    bg_Gamma = np.zeros((4, 4, 4))
    if abs(st1) > 1e-10:
        bg_Gamma[0, 1, 1] = -st1 * ct1
        bg_Gamma[1, 0, 1] = ct1 / st1
        bg_Gamma[1, 1, 0] = ct1 / st1
    if abs(st2) > 1e-10:
        bg_Gamma[2, 3, 3] = -st2 * ct2
        bg_Gamma[3, 2, 3] = ct2 / st2
        bg_Gamma[3, 3, 2] = ct2 / st2
    
    # First derivatives of s
    ds = np.zeros(4)
    for i in range(4):
        ep = pt.copy(); ep[i] += h_fd
        em = pt.copy(); em[i] -= h_fd
        ds[i] = (seam(ep) - seam(em)) / (2 * h_fd)
    
    # Second coordinate derivatives of s
    d2s = np.zeros((4, 4))
    for i in range(4):
        for j in range(i, 4):
            if i == j:
                ep = pt.copy(); ep[i] += h_fd
                em = pt.copy(); em[i] -= h_fd
                d2s[i, j] = (seam(ep) - 2*seam(pt) + seam(em)) / h_fd**2
            else:
                pp = pt.copy(); pp[i] += h_fd; pp[j] += h_fd
                pm = pt.copy(); pm[i] += h_fd; pm[j] -= h_fd
                mp = pt.copy(); mp[i] -= h_fd; mp[j] += h_fd
                mm = pt.copy(); mm[i] -= h_fd; mm[j] -= h_fd
                d2s[i, j] = (seam(pp) - seam(pm) - seam(mp) + seam(mm)) / (4*h_fd**2)
            d2s[j, i] = d2s[i, j]
    
    # Covariant Hessian: s_{;IJ} = s_{,IJ} - Gamma^K_{IJ} s_{,K}
    cov_hess = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            cov_hess[i, j] = d2s[i, j] - sum(bg_Gamma[k, i, j] * ds[k] for k in range(4))
    
    g = h + gamma * cov_hess
    return g

def christoffel_first_kind(pt):
    """Christoffel symbols of the first kind: Gamma_{K,IJ} = (g_{KI,J} + g_{KJ,I} - g_{IJ,K})/2"""
    G1 = np.zeros((4, 4, 4))  # G1[K, I, J]
    dg = np.zeros((4, 4, 4))  # dg[I, J, K] = g_{IJ,K}
    for k in range(4):
        ep = pt.copy(); ep[k] += h_fd
        em = pt.copy(); em[k] -= h_fd
        dg[:, :, k] = (_full_metric(ep) - _full_metric(em)) / (2 * h_fd)
    for K in range(4):
        for I in range(4):
            for J in range(4):
                G1[K, I, J] = 0.5 * (dg[K, I, J] + dg[K, J, I] - dg[I, J, K])
    return G1

def riemann_4d(pt, i, j):
    """Full 4D sectional curvature K(e_i, e_j) via Christoffel finite differences."""
    Gam = christoffel_first_kind(pt)
    g = _full_metric(pt)
    gi = np.linalg.inv(g)
    
    # V_K = Gamma_{K,ij}, U_K = Gamma_{K,ii}, Z_K = Gamma_{K,jj}
    V = Gam[:, i, j]
    U = Gam[:, i, i]
    Z = Gam[:, j, j]
    
    # Quadratic formula: R = g^{KM}(V_K V_M - U_K Z_M)
    R_quad = 0.0
    for K in range(4):
        for M in range(4):
            R_quad += gi[K, M] * (V[K]*V[M] - U[K]*Z[M])
    
    # Full numerical R via finite-difference Christoffels
    # Gamma^K_{IJ} = g^{KM} Gamma_{M,IJ}
    Gam_up = np.einsum('km,mij->kij', gi, Gam)
    dGam_up = np.zeros((4, 4, 4, 4))
    for d in range(4):
        ep = pt.copy(); ep[d] += h_fd
        em = pt.copy(); em[d] -= h_fd
        G1p = christoffel_first_kind(ep)
        G1m = christoffel_first_kind(em)
        gip = np.linalg.inv(_full_metric(ep))
        gim = np.linalg.inv(_full_metric(em))
        Gamp = np.einsum('km,mij->kij', gip, G1p)
        Gamm = np.einsum('km,mij->kij', gim, G1m)
        dGam_up[:, :, :, d] = (Gamp - Gamm) / (2 * h_fd)
    
    # R^m_{jij}
    Rm = np.zeros(4)
    for m in range(4):
        Rm[m] = dGam_up[m, j, j, i] - dGam_up[m, j, i, j]
        for e in range(4):
            Rm[m] += Gam_up[m, i, e] * Gam_up[e, j, j] - Gam_up[m, j, e] * Gam_up[e, j, i]
    
    R_full = sum(g[i, m] * Rm[m] for m in range(4))
    denom = g[i, i] * g[j, j] - g[i, j]**2
    
    return R_full / denom, R_quad / denom

# Test 1: Does the quadratic formula match the full Riemann?
print("="*70)
print("TEST 1: Exact quadratic formula vs full Riemann tensor")
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
    K_full, K_quad = riemann_4d(pt, 0, 2)
    print(f"{name:12s} {K_full:14.8f} {K_quad:14.8f} {abs(K_full-K_quad):10.2e}")

# Test 2: Is K >= 0 for all mixed planes?
print()
print("="*70)
print("TEST 2: K >= 0 for all mixed planes (scanning angles)")
print("="*70)
min_K = float('inf')
min_info = None
n_total = 0
n_negative = 0

for t1 in np.linspace(0.15, np.pi-0.15, 8):
    for t2 in np.linspace(0.15, np.pi-0.15, 8):
        for p1 in [0.0, 1.0, 2.0]:
            for p2 in [0.0, 1.0, 2.0]:
                pt = np.array([t1, p1, t2, p2])
                g = _full_metric(pt)
                gi = np.linalg.inv(g)
                G1 = christoffel_first_kind(pt)
                
                # Scan over mixed plane angles
                for theta in np.linspace(0, np.pi, 12):
                    for phi in np.linspace(0, np.pi, 12):
                        # Mixed plane: X = cos(theta)*e_0 + sin(theta)*e_1
                        #              Y = cos(phi)*e_2 + sin(phi)*e_3
                        ct, st = np.cos(theta), np.sin(theta)
                        cp, sp = np.cos(phi), np.sin(phi)
                        
                        # V_K = Gamma_{K, X, Y}
                        V = ct*cp*G1[:, 0, 2] + ct*sp*G1[:, 0, 3] + st*cp*G1[:, 1, 2] + st*sp*G1[:, 1, 3]
                        U = ct*ct*G1[:, 0, 0] + 2*ct*st*G1[:, 0, 1] + st*st*G1[:, 1, 1]
                        Z = cp*cp*G1[:, 2, 2] + 2*cp*sp*G1[:, 2, 3] + sp*sp*G1[:, 3, 3]
                        
                        R_num = 0.0
                        for K in range(4):
                            for M in range(4):
                                R_num += gi[K, M] * (V[K]*V[M] - U[K]*Z[M])
                        
                        # Area element
                        X_vec = np.array([ct, st, 0, 0])
                        Y_vec = np.array([0, 0, cp, sp])
                        gXX = X_vec @ g @ X_vec
                        gYY = Y_vec @ g @ Y_vec
                        gXY = X_vec @ g @ Y_vec
                        denom = gXX * gYY - gXY**2
                        
                        K_val = R_num / denom
                        n_total += 1
                        if K_val < min_K:
                            min_K = K_val
                            min_info = (t1, p1, t2, p2, theta, phi)
                        if K_val < -1e-6:
                            n_negative += 1

print(f"Total planes tested: {n_total}")
print(f"Min K found: {min_K:.10f}")
if min_info:
    print(f"  at (t1,p1,t2,p2,theta,phi) = ({min_info[0]:.2f},{min_info[1]:.2f},{min_info[2]:.2f},{min_info[3]:.2f},{min_info[4]:.2f},{min_info[5]:.2f})")
print(f"Planes with K < -1e-6: {n_negative}")
print(f"K >= 0 confirmed: {'YES' if n_negative == 0 else 'NO'}")
