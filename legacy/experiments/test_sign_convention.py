#!/usr/bin/env python3
"""
Verify sign convention: compute within-factor curvature K(e_th1, e_ph1)
for the background S^2 x S^2 metric (gamma=0). Must give K = +1.
"""

import numpy as np
from test_Q_S2xS2 import make_seam_product_harmonics

N = 14
eps_th = 0.15
th1 = np.linspace(eps_th, np.pi - eps_th, N)
ph1 = np.linspace(0, 2*np.pi, N, endpoint=False)
th2 = np.linspace(eps_th, np.pi - eps_th, N)
ph2 = np.linspace(0, 2*np.pi, N, endpoint=False)
dxs = [th1[1]-th1[0], ph1[1]-ph1[0], th2[1]-th2[0], ph2[1]-ph2[0]]

T1, P1, T2, P2 = np.meshgrid(th1, ph1, th2, ph2, indexing='ij')
shape = T1.shape
st1 = np.sin(T1)
st2 = np.sin(T2)

# Background metric (gamma=0)
G = np.zeros((4, 4) + shape)
G[0,0] = 1.0; G[1,1] = st1**2; G[2,2] = 1.0; G[3,3] = st2**2

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

# Method 1: R_{0110} (what the original code computes for "R_0202")
# Rm = dChr[m,0,2,2] - dChr[m,2,2,0] + Chr[m,2,p]*Chr[p,0,2] - Chr[m,0,p]*Chr[p,2,2]
# But adapted for (0,1) pair:
R_0110 = np.zeros(shape)
for m in range(4):
    Rm = dChr[m,0,1,1] - dChr[m,1,1,0]
    for p in range(4):
        Rm += Chr[m,1,p]*Chr[p,0,1] - Chr[m,0,p]*Chr[p,1,1]
    R_0110 += G[0,m] * Rm
denom_01 = G[0,0]*G[1,1] - G[0,1]**2

# Method 2: compute_R_ijkl(0, 1, 1, 0)  [same as method 1]
# = g_{0m} R^m_{1,1,0} 
# R^m_{j=1,k=1,l=0} = dChr[m,1,0,1] - dChr[m,1,1,0] + Chr[m,1,p]Chr[p,1,0] - Chr[m,0,p]Chr[p,1,1]

R_up_m110 = np.zeros((4,) + shape)
for m in range(4):
    R_up_m110[m] = dChr[m,1,0,1] - dChr[m,1,1,0]
    for p in range(4):
        R_up_m110[m] += Chr[m,1,p]*Chr[p,1,0] - Chr[m,0,p]*Chr[p,1,1]
R_0110_v2 = np.zeros(shape)
for m in range(4):
    R_0110_v2 += G[0,m] * R_up_m110[m]

# Method 3: R_{0101} = g_{0m} R^m_{1,0,1}
R_up_m101 = np.zeros((4,) + shape)
for m in range(4):
    R_up_m101[m] = dChr[m,1,1,0] - dChr[m,1,0,1]
    for p in range(4):
        R_up_m101[m] += Chr[m,0,p]*Chr[p,1,1] - Chr[m,1,p]*Chr[p,1,0]
R_0101 = np.zeros(shape)
for m in range(4):
    R_0101 += G[0,m] * R_up_m101[m]

trim = 2
sl = tuple(slice(trim, -trim) for _ in range(4))

print("Sign convention check on background S^2 x S^2:")
print(f"  R_0110 / denom = {np.mean(R_0110[sl] / denom_01[sl]):.6f}  (should be +1 or -1)")
print(f"  R_0110_v2 / denom = {np.mean(R_0110_v2[sl] / denom_01[sl]):.6f}")
print(f"  R_0101 / denom = {np.mean(R_0101[sl] / denom_01[sl]):.6f}")
print(f"  (K = +1 for round S^2 is the correct answer)")
print()

# Also check: what does the original code's R_0202 formula give for the (0,1) plane?
# The original code: Rm = dChr[m,0,2,2] - dChr[m,2,2,0] + Chr[m,2,p]Chr[p,0,2] - Chr[m,0,p]Chr[p,2,2]
# This is R^m_{j=0,k=2,l=0}? Let me match:
# dChr[m,0,2,2] = ∂_2 Γ^m_{02} and dChr[m,2,2,0] = ∂_0 Γ^m_{22}
# R^m_{j,k,l} = ∂_k Γ^m_{jl} - ∂_l Γ^m_{jk} + ...
# j=2, k=2, l=0: ∂_2 Γ^m_{20} - ∂_0 Γ^m_{22} + Γ^m_{2p}Γ^p_{20} - Γ^m_{0p}Γ^p_{22}
# Since Γ^m_{20} = Γ^m_{02}, this matches. So the original computes R^m_{220}.
# Then R_0220 = g_{0m} R^m_{220}

# For the within-factor plane (0,1), adapting the same formula:
# R^m_{110} = ∂_1 Γ^m_{10} - ∂_0 Γ^m_{11} + Γ^m_{1p}Γ^p_{10} - Γ^m_{0p}Γ^p_{11}
# This matches Method 1. So Method 1 computes R_{0110}.

K_method1 = R_0110[sl] / denom_01[sl]
K_method3 = R_0101[sl] / denom_01[sl]
print(f"  Method 1 (R_0110/denom, as in original code): mean = {np.mean(K_method1):.6f}")
print(f"  Method 3 (R_0101/denom): mean = {np.mean(K_method3):.6f}")
print(f"  R_0101 = -R_0110: check = {np.allclose(R_0101[sl], -R_0110[sl])}")

# For the MIXED plane (0,2):
R_0220 = np.zeros(shape)
for m in range(4):
    Rm = dChr[m,0,2,2] - dChr[m,2,2,0]
    for p in range(4):
        Rm += Chr[m,2,p]*Chr[p,0,2] - Chr[m,0,p]*Chr[p,2,2]
    R_0220 += G[0,m] * Rm
denom_02 = G[0,0]*G[2,2] - G[0,2]**2
K_mixed = R_0220[sl] / denom_02[sl]
print(f"\n  Mixed K(0,2) on background (should be 0): mean = {np.mean(K_mixed):.6e}")
print(f"  range: [{np.min(K_mixed):.6e}, {np.max(K_mixed):.6e}]")

print("\n  CONCLUSION:")
if np.mean(K_method1) > 0:
    print("  R_0110/denom = +1: the code convention gives CORRECT K")
    print("  So K_mix <= 0 in original test IS correct: mixed curvature is non-positive.")
else:
    print("  R_0110/denom = -1: the code convention gives -K")
    print("  So K_mix <= 0 means actual K >= 0 (non-negative), sign was flipped!")
