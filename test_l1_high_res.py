#!/usr/bin/env python3
"""
High-resolution test: are ALL mixed curvatures >= 0 for l=1 seams?

Uses N=20 grid (vs N=12 previously) and computes both -K (original convention)
and the corrected K via the angle scan with R_{XYXY}/denom.

Key question: are the K < 0 values from the FD test real or FD artifacts?
"""

import numpy as np
from test_Q_S2xS2 import make_seam_product_harmonics

def compute_all_mixed_K(seam_func, gamma, N=20):
    """Compute mixed curvatures with analytical Hessian + FD Christoffel derivatives."""
    eps_th = 0.15
    th1 = np.linspace(eps_th, np.pi - eps_th, N)
    ph1 = np.linspace(0, 2*np.pi, N, endpoint=False)
    th2 = np.linspace(eps_th, np.pi - eps_th, N)
    ph2 = np.linspace(0, 2*np.pi, N, endpoint=False)
    dxs = [th1[1]-th1[0], ph1[1]-ph1[0], th2[1]-th2[0], ph2[1]-ph2[0]]

    T1, P1, T2, P2 = np.meshgrid(th1, ph1, th2, ph2, indexing='ij')
    shape = T1.shape
    st1, ct1 = np.sin(T1), np.cos(T1)
    st2, ct2 = np.sin(T2), np.cos(T2)

    S, dS, d2S = seam_func(T1, P1, T2, P2)

    H = np.zeros((4, 4) + shape)
    H[0,0] = d2S[0][0]
    H[0,1] = d2S[0][1] - (ct1/st1)*dS[1]
    H[0,2] = d2S[0][2]
    H[0,3] = d2S[0][3]
    H[1,1] = d2S[1][1] + st1*ct1*dS[0]
    H[1,2] = d2S[1][2]
    H[1,3] = d2S[1][3]
    H[2,2] = d2S[2][2]
    H[2,3] = d2S[2][3] - (ct2/st2)*dS[3]
    H[3,3] = d2S[3][3] + st2*ct2*dS[2]
    for i in range(4):
        for j in range(i+1, 4):
            H[j,i] = H[i,j]

    G = gamma * H.copy()
    G[0,0] += 1.0; G[1,1] += st1**2; G[2,2] += 1.0; G[3,3] += st2**2

    def fd(arr, axis):
        return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2*dxs[axis])

    dG = np.zeros((4, 4, 4) + shape)
    for k in range(4):
        for i in range(4):
            for j in range(i, 4):
                dG[i,j,k] = fd(G[i,j], k); dG[j,i,k] = dG[i,j,k]

    Gflat = G.reshape(4, 4, -1).transpose(2, 0, 1)
    Giflat = np.linalg.inv(Gflat)
    Ginv = Giflat.transpose(1, 2, 0).reshape(4, 4, *shape)

    Chr = np.zeros((4, 4, 4) + shape)
    for m in range(4):
        for i in range(4):
            for j in range(i, 4):
                val = np.zeros(shape)
                for k in range(4):
                    val += Ginv[m,k] * (dG[k,i,j] + dG[k,j,i] - dG[i,j,k])
                Chr[m,i,j] = 0.5 * val
                if j > i: Chr[m,j,i] = Chr[m,i,j]

    dChr = np.zeros((4, 4, 4, 4) + shape)
    for l in range(4):
        for m in range(4):
            for i in range(4):
                for j in range(i, 4):
                    dChr[m,i,j,l] = fd(Chr[m,i,j], l)
                    if j > i: dChr[m,j,i,l] = dChr[m,i,j,l]

    def compute_R_ijkl(i, j, k, l):
        """R_{ijkl} with convention K(X,Y) = R_{XYXY}/denom."""
        R_up = {}
        for m in range(4):
            term1 = dChr[m,j,l,k] - dChr[m,j,k,l]
            term3 = np.zeros(shape)
            for p in range(4):
                term3 += Chr[m,k,p]*Chr[p,j,l] - Chr[m,l,p]*Chr[p,j,k]
            R_up[m] = term1 + term3
        R = np.zeros(shape)
        for m in range(4):
            R += G[i,m] * R_up[m]
        return R

    trim = 3  # larger trim for better interior accuracy
    sl = tuple(slice(trim, -trim) for _ in range(4))

    # All R_{aAlphaBBeta} components for mixed sector
    R_mixed = {}
    for i in [0,1]:
        for j in [2,3]:
            for k in [0,1]:
                for l in [2,3]:
                    R_mixed[(i,j,k,l)] = compute_R_ijkl(i, j, k, l)

    # Scan over all mixed planes - use CORRECT K formula: K = R_{XYXY}/denom
    Na, Nb = 72, 72
    angles_a = np.linspace(0, np.pi, Na, endpoint=False)
    angles_b = np.linspace(0, np.pi, Nb, endpoint=False)
    K_min_grid = np.full(shape, np.inf)
    K_max_grid = np.full(shape, -np.inf)
    
    for ia in range(Na):
        ca, sa = np.cos(angles_a[ia]), np.sin(angles_a[ia])
        X = np.array([ca, sa])
        for ib in range(Nb):
            cb, sb = np.cos(angles_b[ib]), np.sin(angles_b[ib])
            Y = np.array([cb, sb])
            
            # R_{XYXY} = R_{ijkl} X^i Y^j X^k Y^l  (CORRECT for K = R_{XYXY}/denom)
            num = np.zeros(shape)
            for ii, i in enumerate([0,1]):
                for jj, j in enumerate([2,3]):
                    for kk, k in enumerate([0,1]):
                        for ll, l in enumerate([2,3]):
                            num += R_mixed[(i,j,k,l)] * X[ii]*Y[jj]*X[kk]*Y[ll]
            
            gXX = np.zeros(shape)
            for ii, i in enumerate([0,1]):
                for kk, k in enumerate([0,1]):
                    gXX += G[i,k] * X[ii]*X[kk]
            gYY = np.zeros(shape)
            for jj, j in enumerate([2,3]):
                for ll, l in enumerate([2,3]):
                    gYY += G[j,l] * Y[jj]*Y[ll]
            gXY = np.zeros(shape)
            for ii, i in enumerate([0,1]):
                for jj, j in enumerate([2,3]):
                    gXY += G[i,j] * X[ii]*Y[jj]
            
            denom = gXX*gYY - gXY**2
            K_val = num / denom
            K_min_grid = np.minimum(K_min_grid, K_val)
            K_max_grid = np.maximum(K_max_grid, K_val)
    
    Kmin_i = K_min_grid[sl]
    Kmax_i = K_max_grid[sl]
    return Kmin_i, Kmax_i

# Convergence test: same seam at different resolutions
print("=" * 70)
print("Convergence test: min K over all mixed planes vs grid resolution")
print("=" * 70)

gamma = 0.05
seams = {
    "z1*z2": [0,0,0, 0,0,0, 0,0,1],
    "x1*x2": [1,0,0, 0,0,0, 0,0,0],
    "x1*y2+z1*z2": [0,1,0, 0,0,0, 0,0,1],
}

for name, c in seams.items():
    sf = make_seam_product_harmonics(c)
    print(f"\n--- {name}, gamma={gamma} ---")
    for N in [12, 16, 20]:
        Kmin, Kmax = compute_all_mixed_K(sf, gamma, N)
        print(f"  N={N:2d}: min(Kmin)={np.min(Kmin):+.6e}, max(Kmin)={np.max(Kmin):+.6e}, "
              f"min(Kmax)={np.min(Kmax):+.6e}, max(Kmax)={np.max(Kmax):+.6e}")

# Random l=1 seams at N=20
print("\n" + "=" * 70)
print("Random l=1 seams at N=20, gamma=0.05")
print("=" * 70)

rng = np.random.default_rng(42)
for trial in range(10):
    c = rng.normal(size=9).tolist()
    sf = make_seam_product_harmonics(c)
    Kmin, Kmax = compute_all_mixed_K(sf, gamma, 20)
    min_Kmin = np.min(Kmin)
    max_Kmin = np.max(Kmin)
    print(f"  Trial {trial+1:2d}: min(Kmin)={min_Kmin:+.6e}, max(Kmin)={max_Kmin:+.6e}, "
          f"all_mixed_K>=0: {min_Kmin >= -1e-10}")

# Also check: what gamma range works?
print("\n" + "=" * 70)
print("gamma scan for x1*x2 at N=20")
print("=" * 70)
sf = make_seam_product_harmonics([1,0,0, 0,0,0, 0,0,0])
for gam in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]:
    Kmin, Kmax = compute_all_mixed_K(sf, gam, 20)
    print(f"  gamma={gam:.2f}: min(Kmin)={np.min(Kmin):+.6e}, max(Kmin)={np.max(Kmin):+.6e}")
