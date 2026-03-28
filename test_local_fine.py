#!/usr/bin/env python3
"""
High-accuracy curvature at specific points using analytical l=1 Hessian
with a LOCAL fine grid. Avoids cascaded FD issues.

For l=1: Hess(s) is known exactly. Only metric DERIVATIVES use FD,
and only Christoffel DERIVATIVES use FD. Two levels total.
"""

import numpy as np
from test_Q_S2xS2 import make_seam_product_harmonics

def compute_K_at_point(seam_func, gamma, center, h=1e-4, scan_angles=180):
    """Compute min/max mixed K at a specific point using local fine FD.
    
    center: (th1, ph1, th2, ph2)
    h: FD step size (small = more accurate but more roundoff)
    """
    t10, p10, t20, p20 = center
    
    # Build a 5-point stencil in each direction centered at the target point
    # Need enough points for second derivatives of Christoffels
    N = 7  # 7 points per direction
    
    th1 = t10 + h * np.arange(-(N//2), N//2+1)
    ph1 = p10 + h * np.arange(-(N//2), N//2+1)
    th2 = t20 + h * np.arange(-(N//2), N//2+1)
    ph2 = p20 + h * np.arange(-(N//2), N//2+1)
    
    T1, P1, T2, P2 = np.meshgrid(th1, ph1, th2, ph2, indexing='ij')
    shape = T1.shape
    st1, ct1 = np.sin(T1), np.cos(T1)
    st2, ct2 = np.sin(T2), np.cos(T2)
    
    S, dS, d2S = seam_func(T1, P1, T2, P2)
    
    # Covariant Hessian (exact)
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
    
    # FD for metric derivatives (using the uniform h spacing)
    def fd(arr, axis):
        return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2*h)
    
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
    
    # Extract values at the CENTER point
    c = N // 2  # center index
    ci = (c, c, c, c)
    
    G0 = G[:, :, c, c, c, c]
    Chr0 = Chr[:, :, :, c, c, c, c]
    dChr0 = dChr[:, :, :, :, c, c, c, c]
    
    def compute_R(i, j, k, l):
        val = 0
        for m in range(4):
            Rm = dChr0[m,j,l,k] - dChr0[m,j,k,l]
            for p in range(4):
                Rm += Chr0[m,k,p]*Chr0[p,j,l] - Chr0[m,l,p]*Chr0[p,j,k]
            val += G0[i,m] * Rm
        return val
    
    # All mixed R_{aAlphaBBeta}
    R_mixed = np.zeros((2,2,2,2))
    for ii, i in enumerate([0,1]):
        for jj, j in enumerate([2,3]):
            for kk, k in enumerate([0,1]):
                for ll, l in enumerate([2,3]):
                    R_mixed[ii,jj,kk,ll] = compute_R(i, j, k, l)
    
    # Scan over mixed planes
    K_min = np.inf; K_max = -np.inf
    best_min_angle = None; best_max_angle = None
    
    for ia in range(scan_angles):
        ca, sa = np.cos(np.pi*ia/scan_angles), np.sin(np.pi*ia/scan_angles)
        X = np.array([ca, sa])
        for ib in range(scan_angles):
            cb, sb = np.cos(np.pi*ib/scan_angles), np.sin(np.pi*ib/scan_angles)
            Y = np.array([cb, sb])
            
            num = sum(R_mixed[ii,jj,kk,ll]*X[ii]*Y[jj]*X[kk]*Y[ll]
                      for ii in range(2) for jj in range(2) for kk in range(2) for ll in range(2))
            gXX = sum(G0[i,k]*X[ii]*X[kk] for ii,i in enumerate([0,1]) for kk,k in enumerate([0,1]))
            gYY = sum(G0[j,l]*Y[jj]*Y[ll] for jj,j in enumerate([2,3]) for ll,l in enumerate([2,3]))
            gXY = sum(G0[i,j]*X[ii]*Y[jj] for ii,i in enumerate([0,1]) for jj,j in enumerate([2,3]))
            denom = gXX*gYY - gXY**2
            K_val = num/denom
            if K_val < K_min:
                K_min = K_val; best_min_angle = (ia, ib)
            if K_val > K_max:
                K_max = K_val; best_max_angle = (ia, ib)
    
    # Also compute coordinate-plane K values
    K_coord = {}
    for (a, al, name) in [(0,2,"th1,th2"), (0,3,"th1,ph2"), (1,2,"ph1,th2"), (1,3,"ph1,ph2")]:
        R = compute_R(a, al, a, al)
        denom = G0[a,a]*G0[al,al] - G0[a,al]**2
        K_coord[name] = R / denom if abs(denom) > 1e-15 else float('nan')
    
    return K_min, K_max, K_coord

gamma = 0.1

print("=" * 70)
print(f"Ultra-precise mixed curvatures, gamma={gamma}")
print("=" * 70)

points = [
    (np.pi/4, np.pi/4, np.pi/4, np.pi/4),
    (np.pi/3, np.pi/6, np.pi/4, np.pi/3),
    (np.pi/2, np.pi/4, np.pi/2, np.pi/4),
    (1.0, 0.5, 1.2, 0.8),
    (0.8, 1.5, 0.6, 2.0),
]

for seam_name, coeffs in [("z1z2", [0,0,0,0,0,0,0,0,1]), 
                            ("x1x2", [1,0,0,0,0,0,0,0,0]),
                            ("x1y2+z1z2", [0,1,0,0,0,0,0,0,1])]:
    sf = make_seam_product_harmonics(coeffs)
    print(f"\n--- Seam: {seam_name} ---")
    
    for pt in points:
        K_min, K_max, K_coord = compute_K_at_point(sf, gamma, pt, h=1e-4)
        t1, p1, t2, p2 = pt
        print(f"  ({t1:.2f},{p1:.2f},{t2:.2f},{p2:.2f}): Kmin={K_min:+.8e}, Kmax={K_max:+.8e}")
        for name, val in K_coord.items():
            print(f"    K({name}) = {val:+.8e}")

# Convergence in h
print(f"\n{'='*70}")
print("h-convergence test at (pi/4, pi/4, pi/4, pi/4)")
print(f"{'='*70}")
for seam_name, coeffs in [("z1z2", [0,0,0,0,0,0,0,0,1]), ("x1x2", [1,0,0,0,0,0,0,0,0])]:
    sf = make_seam_product_harmonics(coeffs)
    pt = (np.pi/4, np.pi/4, np.pi/4, np.pi/4)
    print(f"\n--- {seam_name} ---")
    for h in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]:
        K_min, K_max, _ = compute_K_at_point(sf, gamma, pt, h=h, scan_angles=90)
        print(f"  h={h:.0e}: Kmin={K_min:+.8e}, Kmax={K_max:+.8e}")
