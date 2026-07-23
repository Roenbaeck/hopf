#!/usr/bin/env python3
"""
Systematic study: K_min as a function of s = n1.n2 for the full-rank seam.
This maps out exactly how K_min depends on the coupling strength.

Also: verify that for GENERAL (σ1,σ2,σ3) l=1 seams, the K=0 locus
is determined by the number of nonzero singular values.
"""

import numpy as np
from test_Q_S2xS2 import make_seam_product_harmonics

def compute_Kmin_at_point(seam_func, gamma, center, h=1e-4, scan_angles=60):
    """Compute min/max mixed K at a specific point."""
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
    def compute_R(i,j,k,l):
        val = 0
        for m in range(4):
            Rm = dChr0[m,j,l,k]-dChr0[m,j,k,l]
            for p in range(4):
                Rm += Chr0[m,k,p]*Chr0[p,j,l]-Chr0[m,l,p]*Chr0[p,j,k]
            val += G0[i,m]*Rm
        return val
    R_mixed = np.zeros((2,2,2,2))
    for ii,i in enumerate([0,1]):
        for jj,j in enumerate([2,3]):
            for kk,k in enumerate([0,1]):
                for ll,l in enumerate([2,3]):
                    R_mixed[ii,jj,kk,ll] = compute_R(i,j,k,l)
    K_min=np.inf; K_max=-np.inf
    for ia in range(scan_angles):
        ca,sa = np.cos(np.pi*ia/scan_angles), np.sin(np.pi*ia/scan_angles)
        X = np.array([ca, sa])
        for ib in range(scan_angles):
            cb,sb = np.cos(np.pi*ib/scan_angles), np.sin(np.pi*ib/scan_angles)
            Y = np.array([cb, sb])
            num = sum(R_mixed[ii,jj,kk,ll]*X[ii]*Y[jj]*X[kk]*Y[ll]
                      for ii in range(2) for jj in range(2) for kk in range(2) for ll in range(2))
            gXX = sum(G0[i,k]*X[ii]*X[kk] for ii,i in enumerate([0,1]) for kk,k in enumerate([0,1]))
            gYY = sum(G0[j,l]*Y[jj]*Y[ll] for jj,j in enumerate([2,3]) for ll,l in enumerate([2,3]))
            gXY = sum(G0[i,j]*X[ii]*Y[jj] for ii,i in enumerate([0,1]) for jj,j in enumerate([2,3]))
            denom = gXX*gYY - gXY**2
            K_val = num/denom
            K_min=min(K_min,K_val); K_max=max(K_max,K_val)
    return K_min, K_max

gamma = 0.1

# === Part 1: K_min vs s for n1.n2 seam ===
print("="*70)
print("K_min vs s = n1.n2 for the full-rank seam")
print("="*70)

sf = make_seam_product_harmonics([1,0,0, 0,1,0, 0,0,1])

# Use (θ₁, φ₁) = (π/4, 0) fixed, and vary θ₂ to sweep s
# n₁ = (sin(π/4), 0, cos(π/4)) = (√2/2, 0, √2/2)
# n₂ = (sinθ₂, 0, cosθ₂) when φ₂=0
# s = sin(π/4)sinθ₂ + cos(π/4)cosθ₂ = cos(π/4 - θ₂)

t1, p1 = np.pi/4, 0.0
print(f"\nFixed: (θ₁,φ₁) = ({t1:.4f}, {p1:.4f})")
print(f"{'θ₂':>8s}  {'s=n1·n2':>10s}  {'K_min':>12s}  {'K_max':>12s}")
print("-"*50)

for t2 in np.linspace(0.3, np.pi-0.3, 15):
    p2 = 0.0
    s_val = np.cos(t1 - t2)
    Kmin, Kmax = compute_Kmin_at_point(sf, gamma, (t1, p1, t2, p2))
    print(f"{t2:8.4f}  {s_val:+10.6f}  {Kmin:+12.4e}  {Kmax:+12.4e}")

# Also sweep with φ₂ ≠ 0 to get s near 0
print(f"\nSweep φ₂ with θ₁=θ₂=π/2 to get s = cos(φ₂):")
t1, p1, t2 = np.pi/2, 0.0, np.pi/2
print(f"{'φ₂':>8s}  {'s':>10s}  {'K_min':>12s}  {'K_max':>12s}")
print("-"*50)
for p2 in np.linspace(0.1, np.pi, 10):
    s_val = np.cos(p2)  # at equator with φ₁=0: s = cos(φ₁-φ₂) = cos(φ₂)
    Kmin, Kmax = compute_Kmin_at_point(sf, gamma, (t1, p1, t2, p2))
    print(f"{p2:8.4f}  {s_val:+10.6f}  {Kmin:+12.4e}  {Kmax:+12.4e}")

# === Part 2: Different rank seams ===
print("\n"+"="*70)
print("K_min for different-rank l=1 seams at generic point (1.0, 2.0, 1.5, 3.0)")
print("="*70)

pt = (1.0, 2.0, 1.5, 3.0)
seam_configs = [
    ("z1z2 (rank 1)", [0,0,0, 0,0,0, 0,0,1]),
    ("x1x2 (rank 1)", [1,0,0, 0,0,0, 0,0,0]),
    ("x1x2+y1y2 (rank 2)", [1,0,0, 0,1,0, 0,0,0]),
    ("x1y2+z1z2 (rank 2)", [0,1,0, 0,0,0, 0,0,1]),
    ("n1.n2 (rank 3)", [1,0,0, 0,1,0, 0,0,1]),
    ("2*x1x2+y1y2+z1z2 (rank 3, unequal σ)", [2,0,0, 0,1,0, 0,0,1]),
]

for name, coeffs in seam_configs:
    sf = make_seam_product_harmonics(coeffs)
    Kmin, Kmax = compute_Kmin_at_point(sf, gamma, pt)
    print(f"  {name:>40s}: Kmin={Kmin:+.4e}, Kmax={Kmax:+.4e}")

# === Part 3: Scaling of K_min with s^2 ===
print("\n"+"="*70)
print("Does K_min scale as s² (coupling strength)?")
print("="*70)
print("At θ₂ points giving different |s| values:")

t1, p1 = np.pi/4, 0.0
data_s = []
data_Kmin = []
for t2 in np.linspace(0.5, np.pi-0.5, 12):
    p2 = 0.0
    s_val = np.cos(t1 - t2)
    Kmin, Kmax = compute_Kmin_at_point(sf, gamma, (t1, p1, t2, p2))
    data_s.append(s_val)
    data_Kmin.append(Kmin)

# Group by similar |s| and see if K_min ~ s^2 or |s| or something else
sf = make_seam_product_harmonics([1,0,0, 0,1,0, 0,0,1])
print(f"{'s':>10s}  {'K_min':>12s}  {'K_min/s²':>12s}  {'K_min/|s|':>12s}")
print("-"*52)
for t2 in np.linspace(0.4, np.pi-0.4, 15):
    p2 = 0.0
    s_val = np.cos(t1 - t2)
    if abs(s_val) < 0.01:
        continue
    Kmin, Kmax = compute_Kmin_at_point(sf, gamma, (t1, p1, t2, p2))
    ratio_s2 = Kmin / s_val**2 if abs(s_val) > 0.01 else np.nan
    ratio_abs = Kmin / abs(s_val) if abs(s_val) > 0.01 else np.nan
    print(f"{s_val:+10.6f}  {Kmin:+12.4e}  {ratio_s2:+12.4e}  {ratio_abs:+12.4e}")
