#!/usr/bin/env python3
"""
Compute ALL four mixed sectional curvatures for l=1 vs l=2 seams on S^2 x S^2.

Mixed planes:
  K_02 = K(d/dth1, d/dth2)
  K_03 = K(d/dth1, d/dph2) 
  K_12 = K(d/dph1, d/dth2)
  K_13 = K(d/dph1, d/dph2)

And the minimum over all mixed unit-vector planes (X in T_1, Y in T_2).
"""

import numpy as np
from test_Q_S2xS2 import make_seam_product_harmonics

def build_all_mixed_curvatures(seam_func, gamma, N=12):
    """Compute all R_{aAlphaBBeta} and sectional curvatures for mixed planes."""
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

    # Covariant Hessian
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

    # Full Riemann tensor R_{ijkl} (lowered) for mixed components
    # R_{abcd} = g_{am} R^m_{bcd}
    # R^m_{bcd} = dChr^m_{bd}/dx^c - dChr^m_{bc}/dx^d + Chr^m_{cp}*Chr^p_{bd} - Chr^m_{dp}*Chr^p_{bc}
    
    def compute_R_ijkl(i, j, k, l):
        """Compute R_{ijkl} = g_{im} R^m_{jkl}."""
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

    # Compute all mixed Riemann components R_{a,alpha,b,beta}
    # Factor 1 indices: {0,1} (th1, ph1)
    # Factor 2 indices: {2,3} (th2, ph2)
    
    trim = 2
    sl = tuple(slice(trim, -trim) for _ in range(4))

    # Coordinate-basis sectional curvatures K(e_i, e_j) = R_{ijji} / (g_{ii}*g_{jj} - g_{ij}^2)
    mixed_pairs = [(0,2), (0,3), (1,2), (1,3)]  # all mixed planes
    
    print(f"  gamma = {gamma}")
    for (a, al) in mixed_pairs:
        R_val = compute_R_ijkl(a, al, al, a)  # R_{a,alpha,alpha,a} for sectional curv
        denom = G[a,a]*G[al,al] - G[a,al]**2
        K = R_val / denom
        Ki = K[sl]
        print(f"  K({a},{al}): [{np.min(Ki):.6e}, {np.max(Ki):.6e}]  K<=0: {np.max(Ki) <= 1e-10}")

    # Minimum over all mixed unit-vector planes at each point
    # K(X,Y) where X = cos(a)*e_0^{on} + sin(a)*e_1^{on}, Y = cos(b)*e_2^{on} + sin(b)*e_3^{on}
    # e_i^{on} = orthonormal frame
    # 
    # Compute the 2x2 mixed curvature matrix M_{(a,alpha),(b,beta)} in orthonormal frame
    # Then minimize over decomposable tensors X tensor Y.
    
    # Build orthonormal frame from coordinate basis within each factor
    # Factor 1: e_0, e_1 = normalized th1, ph1 (Gram-Schmidt)
    # For S^2 coord basis: g_1 = [[1, 0],[0, sin^2 th1]] + gamma*H corrections
    # Just normalize each coord vector
    
    # Actually simpler: compute R_{aAlphaBBeta} for all 2x2 combinations
    # in orthonormal frame, then scan over angles.
    
    # Mixed curvature in ON frame:
    # M_{ij} where i = (a_idx=0or1) and j = (alpha_idx=0or1)
    # M_{00} = R(hat_0, hat_2, hat_2, hat_0) etc.
    
    # Build ON frame via Cholesky of each 2x2 block
    G1 = np.array([[G[0,0], G[0,1]], [G[1,0], G[1,1]]])  # (2,2,N,N,N,N)
    G2 = np.array([[G[2,2], G[2,3]], [G[3,2], G[3,3]]])
    
    # For the ON frame, we need transformations e_hat = L^{-1} e_coord
    # where G_block = L L^T. Then R_{hat} = L^{-T} R L^{-1} suitably.
    
    # For 4 mixed planes in ON frame:
    # R^{ON}_{A Alpha B Beta} = sum L1^{-1}_{Aa} L2^{-1}_{Alpha,alpha} L1^{-1}_{Bb} L2^{-1}_{Beta,beta} R_{a alpha b beta}
    
    # This is complex. Let me just scan over angles numerically.
    Na, Nb = 36, 36
    angles_a = np.linspace(0, np.pi, Na)  # half circle suffices for unit vectors
    angles_b = np.linspace(0, np.pi, Nb)
    
    # We need R_{ijkl} for all i,k in {0,1} and j,l in {2,3}
    # K(X,Y) = R_{ijkl} X^i Y^j X^k Y^l / (g_{ik}X^i X^k * g_{jl}Y^j Y^l - (g_{ij}X^i Y^j)^2)
    # where X = (X0, X1, 0, 0) and Y = (0, 0, Y2, Y3)
    
    # Pre-compute needed R_{ijkl}
    R_mixed = {}
    for i in [0,1]:
        for j in [2,3]:
            for k in [0,1]:
                for l in [2,3]:
                    R_mixed[(i,j,k,l)] = compute_R_ijkl(i, j, k, l)
    
    K_min_grid = np.full(shape, np.inf)
    K_max_grid = np.full(shape, -np.inf)
    
    for ia in range(Na):
        ca, sa = np.cos(angles_a[ia]), np.sin(angles_a[ia])
        X = np.array([ca, sa])  # components in coords 0,1
        for ib in range(Nb):
            cb, sb = np.cos(angles_b[ib]), np.sin(angles_b[ib])
            Y = np.array([cb, sb])  # components in coords 2,3
            
            # R_{ijkl} X^i Y^j X^k Y^l
            num = np.zeros(shape)
            for ii, i in enumerate([0,1]):
                for jj, j in enumerate([2,3]):
                    for kk, k in enumerate([0,1]):
                        for ll, l in enumerate([2,3]):
                            num += R_mixed[(i,j,k,l)] * X[ii]*Y[jj]*X[kk]*Y[ll]
            
            # g_{ik}X^i X^k
            gXX = np.zeros(shape)
            for ii, i in enumerate([0,1]):
                for kk, k in enumerate([0,1]):
                    gXX += G[i,k] * X[ii]*X[kk]
            
            # g_{jl}Y^j Y^l
            gYY = np.zeros(shape)
            for jj, j in enumerate([2,3]):
                for ll, l in enumerate([2,3]):
                    gYY += G[j,l] * Y[jj]*Y[ll]
            
            # g_{ij}X^i Y^j
            gXY = np.zeros(shape)
            for ii, i in enumerate([0,1]):
                for jj, j in enumerate([2,3]):
                    gXY += G[i,j] * X[ii]*Y[jj]
            
            denom = gXX*gYY - gXY**2
            K_val = num / denom
            
            K_min_grid = np.minimum(K_min_grid, K_val)
            K_max_grid = np.maximum(K_max_grid, K_val)
    
    K_min_interior = K_min_grid[sl]
    K_max_interior = K_max_grid[sl]
    print(f"  Min mixed K over all planes: [{np.min(K_min_interior):.6e}, {np.max(K_min_interior):.6e}]")
    print(f"  Max mixed K over all planes: [{np.min(K_max_interior):.6e}, {np.max(K_max_interior):.6e}]")
    print(f"  ALL mixed K <= 0: {np.max(K_max_interior) <= 1e-10}")

    return K_max_interior

print("=" * 70)
print("All mixed curvatures for l=1 seams")
print("=" * 70)

gamma = 0.05

seams_l1 = {
    "x1*x2": [1,0,0, 0,0,0, 0,0,0],
    "z1*z2": [0,0,0, 0,0,0, 0,0,1],
    "x1*y2+z1*z2": [0,1,0, 0,0,0, 0,0,1],
}

for name, c in seams_l1.items():
    print(f"\n--- l=1 seam: {name} ---")
    build_all_mixed_curvatures(make_seam_product_harmonics(c), gamma)

# Random l=1 seams
rng = np.random.default_rng(42)
print("\n--- Random l=1 seams ---")
for trial in range(5):
    c = rng.normal(size=9).tolist()
    print(f"\nRandom l=1 #{trial+1}:")
    K_max = build_all_mixed_curvatures(make_seam_product_harmonics(c), gamma)

print("\n" + "=" * 70)
print("All mixed curvatures for l=2 seams (for comparison)")  
print("=" * 70)

# For l=2, we need the l=2 seam builder
# Using finite differences as in test_Q_S2xS2_l2.py
from test_Q_S2xS2_l2 import make_seam_general

# l=2 basis functions on S^2
def Y20(th, ph): return 3*np.cos(th)**2 - 1  # proportional to Y_2^0
def Y21c(th, ph): return np.sin(th)*np.cos(th)*np.cos(ph)  # ~ Re(Y_2^1) 
def Y21s(th, ph): return np.sin(th)*np.cos(th)*np.sin(ph)  # ~ Im(Y_2^1)
def Y22c(th, ph): return np.sin(th)**2*np.cos(2*ph)  # ~ Re(Y_2^2)
def Y22s(th, ph): return np.sin(th)**2*np.sin(2*ph)  # ~ Im(Y_2^2)

l2_basis = [Y20, Y21c, Y21s, Y22c, Y22s]

# A simple l=2 seam: Y21c(unit1) * Y21c(unit2) = xz * xz
print(f"\n--- l=2 seam: xz*xz ---")
# This uses the general builder with finite differences
# We need a compatible interface. Let me build inline.

def build_l2_mixed_curvatures(seam_func_simple, gamma, N=12):
    """For a seam defined as s(th1,ph1,th2,ph2) using finite differences."""
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

    S = seam_func_simple(T1, P1, T2, P2)

    def fd(arr, axis):
        return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2*dxs[axis])
    def fd2(arr, axis):
        return (np.roll(arr, -1, axis=axis) - 2*arr + np.roll(arr, 1, axis=axis)) / (dxs[axis]**2)
    def fd_cross(arr, ax1, ax2):
        return fd(fd(arr, ax1), ax2)

    # Coordinate 2nd derivatives
    d2S = [[None]*4 for _ in range(4)]
    for i in range(4):
        d2S[i][i] = fd2(S, i)
        for j in range(i+1, 4):
            d2S[i][j] = fd_cross(S, i, j)
            d2S[j][i] = d2S[i][j]
    dS = [fd(S, i) for i in range(4)]

    # Covariant Hessian
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

    trim = 2
    sl = tuple(slice(trim, -trim) for _ in range(4))

    mixed_pairs = [(0,2), (0,3), (1,2), (1,3)]
    print(f"  gamma = {gamma}")
    for (a, al) in mixed_pairs:
        R_val = compute_R_ijkl(a, al, al, a)
        denom = G[a,a]*G[al,al] - G[a,al]**2
        K = R_val / denom
        Ki = K[sl]
        print(f"  K({a},{al}): [{np.min(Ki):.6e}, {np.max(Ki):.6e}]  K<=0: {np.max(Ki) <= 1e-10}")

    # Scan over all mixed planes
    R_mixed = {}
    for i in [0,1]:
        for j in [2,3]:
            for k in [0,1]:
                for l in [2,3]:
                    R_mixed[(i,j,k,l)] = compute_R_ijkl(i, j, k, l)

    Na, Nb = 36, 36
    angles_a = np.linspace(0, np.pi, Na)
    angles_b = np.linspace(0, np.pi, Nb)
    K_max_grid = np.full(shape, -np.inf)
    
    for ia in range(Na):
        ca, sa = np.cos(angles_a[ia]), np.sin(angles_a[ia])
        X = np.array([ca, sa])
        for ib in range(Nb):
            cb, sb = np.cos(angles_b[ib]), np.sin(angles_b[ib])
            Y = np.array([cb, sb])
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
            K_max_grid = np.maximum(K_max_grid, K_val)
    
    K_max_interior = K_max_grid[sl]
    print(f"  Max mixed K over all planes: [{np.min(K_max_interior):.6e}, {np.max(K_max_interior):.6e}]")
    print(f"  ALL mixed K <= 0: {np.max(K_max_interior) <= 1e-10}")


# l=2 seam: xz * xz
def xz_xz(t1, p1, t2, p2):
    return np.sin(t1)*np.cos(t1)*np.cos(p1) * np.sin(t2)*np.cos(t2)*np.cos(p2)
build_l2_mixed_curvatures(xz_xz, gamma)

# Random l=2 seam
print(f"\n--- Random l=2 seam ---")
rng2 = np.random.default_rng(123)
def make_random_l2():
    c = rng2.normal(size=(5, 5))
    def sf(t1, p1, t2, p2):
        b1 = [f(t1, p1) for f in l2_basis]
        b2 = [f(t2, p2) for f in l2_basis]
        s = np.zeros_like(t1)
        for i in range(5):
            for j in range(5):
                s += c[i,j] * b1[i] * b2[j]
        return s
    return sf
build_l2_mixed_curvatures(make_random_l2(), gamma)
