#!/usr/bin/env python3
"""
Test: does the full-rank l=1 seam s = x1x2 + y1y2 + z1z2 = n1.n2
achieve K > 0 for ALL mixed planes everywhere?

If so, this would mean the seam metric has STRICTLY POSITIVE curvature
(within-factor K ≈ 1, mixed K > 0), which would have major implications
for the Hopf conjecture.

Also test rank-2 and random l=1 seams.
"""

import numpy as np
from test_Q_S2xS2 import make_seam_product_harmonics

def compute_K_at_point(seam_func, gamma, center, h=1e-4, scan_angles=180):
    """Compute min/max mixed K at a specific point using local fine FD."""
    t10, p10, t20, p20 = center
    N = 7
    
    th1 = t10 + h * np.arange(-(N//2), N//2+1)
    ph1 = p10 + h * np.arange(-(N//2), N//2+1)
    th2 = t20 + h * np.arange(-(N//2), N//2+1)
    ph2 = p20 + h * np.arange(-(N//2), N//2+1)
    
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
    
    c = N // 2
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
    
    R_mixed = np.zeros((2,2,2,2))
    for ii, i in enumerate([0,1]):
        for jj, j in enumerate([2,3]):
            for kk, k in enumerate([0,1]):
                for ll, l in enumerate([2,3]):
                    R_mixed[ii,jj,kk,ll] = compute_R(i, j, k, l)
    
    K_min = np.inf; K_max = -np.inf
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
            K_min = min(K_min, K_val)
            K_max = max(K_max, K_val)
    
    return K_min, K_max

gamma = 0.1

# Dense sampling of points on S^2 x S^2
points = []
for t1 in np.linspace(0.2, np.pi-0.2, 6):
    for p1 in np.linspace(0, 2*np.pi, 6, endpoint=False):
        for t2 in np.linspace(0.2, np.pi-0.2, 6):
            for p2 in np.linspace(0, 2*np.pi, 6, endpoint=False):
                points.append((t1, p1, t2, p2))

# Test full-rank seam: s = x1x2 + y1y2 + z1z2
print("=" * 70)
print(f"Full-rank seam: s = x1x2 + y1y2 + z1z2 = n1.n2, gamma={gamma}")
print("=" * 70)

sf = make_seam_product_harmonics([1,0,0, 0,1,0, 0,0,1])
K_mins = []
K_maxs = []
for i, pt in enumerate(points):
    K_min, K_max = compute_K_at_point(sf, gamma, pt, h=1e-4, scan_angles=90)
    K_mins.append(K_min)
    K_maxs.append(K_max)
    if K_min < 1e-9:
        t1, p1, t2, p2 = pt
        print(f"  SMALL K_min at ({t1:.2f},{p1:.2f},{t2:.2f},{p2:.2f}): {K_min:+.4e}, K_max={K_max:+.4e}")

print(f"\n  Overall: K_min_range = [{min(K_mins):+.6e}, {max(K_mins):+.6e}]")
print(f"           K_max_range = [{min(K_maxs):+.6e}, {max(K_maxs):+.6e}]")
print(f"  ALL mixed K >= 0?: {min(K_mins) >= -1e-9}")
print(f"  ALL mixed K > 0?:  {min(K_mins) > 1e-9}")

# Also test rank-1: z1z2 (should have K=0 places)
print(f"\n{'='*70}")
print(f"Rank-1 seam: z1z2, gamma={gamma}")
print(f"{'='*70}")
sf = make_seam_product_harmonics([0,0,0, 0,0,0, 0,0,1])
K_mins = []
for i, pt in enumerate(points):
    K_min, K_max = compute_K_at_point(sf, gamma, pt, h=1e-4, scan_angles=90)
    K_mins.append(K_min)
print(f"  K_min_range = [{min(K_mins):+.6e}, {max(K_mins):+.6e}]")
print(f"  Has K ≈ 0 places: {min(K_mins) < 1e-7}")

# Also check within-factor curvature to make sure it stays positive 
print(f"\n{'='*70}")
print(f"Within-factor curvature K(e_0,e_1) for n1.n2 seam")
print(f"{'='*70}")
sf = make_seam_product_harmonics([1,0,0, 0,1,0, 0,0,1])
for pt in [(np.pi/4, np.pi/4, np.pi/4, np.pi/4),
           (0.3, 0.5, 0.3, 0.5),
           (np.pi/2, np.pi/2, np.pi/2, np.pi/2)]:
    t10, p10, t20, p20 = pt
    N = 7; h = 1e-4
    th1 = t10 + h * np.arange(-(N//2), N//2+1)
    ph1 = p10 + h * np.arange(-(N//2), N//2+1)
    th2 = t20 + h * np.arange(-(N//2), N//2+1)
    ph2 = p20 + h * np.arange(-(N//2), N//2+1)
    T1, P1, T2, P2 = np.meshgrid(th1, ph1, th2, ph2, indexing='ij')
    shape = T1.shape; st1, ct1 = np.sin(T1), np.cos(T1); st2, ct2 = np.sin(T2), np.cos(T2)
    S, dS, d2S = sf(T1, P1, T2, P2)
    H = np.zeros((4, 4) + shape)
    H[0,0] = d2S[0][0]; H[0,1] = d2S[0][1] - (ct1/st1)*dS[1]
    H[1,1] = d2S[1][1] + st1*ct1*dS[0]
    H[0,2] = d2S[0][2]; H[0,3] = d2S[0][3]; H[1,2] = d2S[1][2]; H[1,3] = d2S[1][3]
    H[2,2] = d2S[2][2]; H[2,3] = d2S[2][3] - (ct2/st2)*dS[3]; H[3,3] = d2S[3][3] + st2*ct2*dS[2]
    for i in range(4):
        for j in range(i+1, 4):
            H[j,i] = H[i,j]
    G = gamma * H.copy()
    G[0,0] += 1; G[1,1] += st1**2; G[2,2] += 1; G[3,3] += st2**2
    def fd(arr, axis):
        return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2*h)
    dG = np.zeros((4,4,4)+shape)
    for k in range(4):
        for i in range(4):
            for j in range(i,4):
                dG[i,j,k] = fd(G[i,j], k); dG[j,i,k] = dG[i,j,k]
    Gf = G.reshape(4,4,-1).transpose(2,0,1)
    Gi = np.linalg.inv(Gf).transpose(1,2,0).reshape(4,4,*shape)
    Chr = np.zeros((4,4,4)+shape)
    for m in range(4):
        for i in range(4):
            for j in range(i,4):
                v = np.zeros(shape)
                for k in range(4):
                    v += Gi[m,k]*(dG[k,i,j]+dG[k,j,i]-dG[i,j,k])
                Chr[m,i,j] = 0.5*v
                if j>i: Chr[m,j,i] = Chr[m,i,j]
    dChr = np.zeros((4,4,4,4)+shape)
    for l in range(4):
        for m in range(4):
            for i in range(4):
                for j in range(i,4):
                    dChr[m,i,j,l] = fd(Chr[m,i,j],l)
                    if j>i: dChr[m,j,i,l] = dChr[m,i,j,l]
    c = N//2
    G0=G[:,:,c,c,c,c]; Chr0=Chr[:,:,:,c,c,c,c]; dChr0=dChr[:,:,:,:,c,c,c,c]
    # R_{0101}
    R_0101 = 0
    for m in range(4):
        Rm = dChr0[m,1,1,0] - dChr0[m,1,0,1]
        for p in range(4):
            Rm += Chr0[m,0,p]*Chr0[p,1,1] - Chr0[m,1,p]*Chr0[p,1,0]
        R_0101 += G0[0,m] * Rm
    denom = G0[0,0]*G0[1,1] - G0[0,1]**2
    K_within = R_0101/denom
    print(f"  ({pt[0]:.2f},{pt[1]:.2f},{pt[2]:.2f},{pt[3]:.2f}): K_within = {K_within:.6f}")
