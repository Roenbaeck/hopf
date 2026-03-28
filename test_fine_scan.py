#!/usr/bin/env python3
"""
Fine angular scan to check if K_min = 0 at generic points for n1.n2 seam.
If SO(3) acts transitively on level sets of s, then K_min must be 
the same at all points with the same s. Check if the earlier K_min = 8.7e-7 
was just insufficient angular resolution.
"""

import numpy as np
from test_Q_S2xS2 import make_seam_product_harmonics

def compute_Kmin_fine(seam_func, gamma, center, h=1e-4, scan_angles=360):
    """Compute min/max mixed K with very fine angular scan."""
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
        for j in range(i+1, 4): H[j,i] = H[i,j]
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
    
    # Build the curvature as a bilinear form on mixed planes
    # K(X,Y) = R_{XYXY} / (|X|^2|Y|^2 - <X,Y>^2)
    # where X in T_1 (indices 0,1), Y in T_2 (indices 2,3)
    
    R_mixed = np.zeros((2,2,2,2))
    for ii,i in enumerate([0,1]):
        for jj,j in enumerate([2,3]):
            for kk,k in enumerate([0,1]):
                for ll,l in enumerate([2,3]):
                    R_mixed[ii,jj,kk,ll] = compute_R(i,j,k,l)
    
    # Also extract metric blocks
    G11 = G0[:2,:2].copy()  # factor 1 metric
    G22 = G0[2:,2:].copy()  # factor 2 metric
    G12 = G0[:2,2:].copy()  # cross metric
    
    # More efficient: compute eigenvalues of the curvature operator
    # Build the matrix Q_{(ab)} such that K = X^T Q X / (norm terms)
    # where X = [cos α, sin α] parameterizes T_1 direction
    # and Y = [cos β, sin β] parameterizes T_2 direction
    
    K_min = np.inf; K_max = -np.inf
    best_angles = None
    
    angles = np.linspace(0, np.pi, scan_angles, endpoint=False)
    
    for ia, alpha in enumerate(angles):
        ca, sa = np.cos(alpha), np.sin(alpha)
        X = np.array([ca, sa])
        gXX = X @ G11 @ X
        
        for ib, beta in enumerate(angles):
            cb, sb = np.cos(beta), np.sin(beta)
            Y = np.array([cb, sb])
            
            gYY = Y @ G22 @ Y
            gXY = X @ G12 @ Y
            
            num = sum(R_mixed[ii,jj,kk,ll]*X[ii]*Y[jj]*X[kk]*Y[ll]
                      for ii in range(2) for jj in range(2) for kk in range(2) for ll in range(2))
            denom = gXX*gYY - gXY**2
            K_val = num / denom
            
            if K_val < K_min:
                K_min = K_val
                best_angles = (alpha, beta)
            K_max = max(K_max, K_val)
    
    return K_min, K_max, best_angles, G0, R_mixed, G11, G22, G12

gamma = 0.1
sf = make_seam_product_harmonics([1,0,0, 0,1,0, 0,0,1])

print("="*70)
print("FINE angular scan: K_min for n1.n2 seam")
print("="*70)

# Test at the generic point with different angular resolutions
pt = (1.0, 2.0, 1.5, 3.0)
s_val = (np.sin(1)*np.cos(2)*np.sin(1.5)*np.cos(3) + 
         np.sin(1)*np.sin(2)*np.sin(1.5)*np.sin(3) + 
         np.cos(1)*np.cos(1.5))
print(f"\nPoint ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}, {pt[3]:.1f}), s = {s_val:.6f}")
print(f"{'n_angles':>10s}  {'K_min':>14s}  {'α_best':>10s}  {'β_best':>10s}")
print("-"*50)

for n_ang in [30, 60, 90, 180, 360, 720, 1440]:
    Kmin, Kmax, best, G0, Rm, G11, G22, G12 = compute_Kmin_fine(sf, gamma, pt, scan_angles=n_ang)
    print(f"{n_ang:10d}  {Kmin:+14.6e}  {np.degrees(best[0]):10.2f}  {np.degrees(best[1]):10.2f}")

# Also test at the "symmetric" point where K_min should be exactly 0
# θ₂ = pi/4 + acos(s_val) to get same s value at (π/4, 0, θ₂, 0)
import scipy.optimize
t2_target = np.pi/4 + np.arccos(s_val)  # might give s ≈ s_val
pt2 = (np.pi/4, 0, t2_target, 0)
s_check = np.cos(np.pi/4 - t2_target)
print(f"\nSymmetric point ({pt2[0]:.4f}, {pt2[1]:.4f}, {pt2[2]:.4f}, {pt2[3]:.4f}), s = {s_check:.6f}")
for n_ang in [30, 60, 180, 720]:
    Kmin, Kmax, best, *_ = compute_Kmin_fine(sf, gamma, pt2, scan_angles=n_ang)
    print(f"{n_ang:10d}  {Kmin:+14.6e}  {np.degrees(best[0]):10.2f}  {np.degrees(best[1]):10.2f}")

# Optimization-based approach: find true K_min analytically
print("\n" + "="*70)
print("Eigenvalue approach: find K_min as eigenvalue problem")
print("="*70)

# At the generic point, compute the curvature operator and find its eigenvalues
pt = (1.0, 2.0, 1.5, 3.0)
Kmin, Kmax, best, G0, R_mixed, G11, G22, G12 = compute_Kmin_fine(sf, gamma, pt, scan_angles=720)

# The curvature K(X,Y) = R(X,Y,X,Y) / (|X|²|Y|² - <X,Y>²)
# For fixed X, K(X,Y) is minimized/maximized as a generalized eigenvalue problem in Y
# For the absolute minimum, we need to optimize over both X and Y simultaneously

# Use scipy.optimize to find the true minimum
from scipy.optimize import minimize

def neg_K(angles):
    alpha, beta = angles
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    X = np.array([ca, sa])
    Y = np.array([cb, sb])
    gXX = X @ G11 @ X
    gYY = Y @ G22 @ Y
    gXY = X @ G12 @ Y
    num = sum(R_mixed[ii,jj,kk,ll]*X[ii]*Y[jj]*X[kk]*Y[ll]
              for ii in range(2) for jj in range(2) for kk in range(2) for ll in range(2))
    denom = gXX*gYY - gXY**2
    return num/denom  # minimize K (not negative K, we want minimum)

# Multi-start optimization
results = []
for a0 in np.linspace(0, np.pi, 20):
    for b0 in np.linspace(0, np.pi, 20):
        res = minimize(neg_K, [a0, b0], method='Nelder-Mead', 
                       options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 10000})
        results.append((res.fun, res.x))

min_result = min(results, key=lambda x: x[0])
print(f"\nGeneric point ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}, {pt[3]:.1f}), s = {s_val:.6f}")
print(f"  Optimized K_min = {min_result[0]:+.10e}")
print(f"  At angles (α, β) = ({np.degrees(min_result[1][0]):.4f}°, {np.degrees(min_result[1][1]):.4f}°)")

# Repeat at symmetric point
Kmin2, Kmax2, best2, G02, Rm2, G11_2, G22_2, G12_2 = compute_Kmin_fine(sf, gamma, pt2, scan_angles=720)

def neg_K2(angles):
    alpha, beta = angles
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    X = np.array([ca, sa])
    Y = np.array([cb, sb])
    gXX = X @ G11_2 @ X
    gYY = Y @ G22_2 @ Y
    gXY = X @ G12_2 @ Y
    num = sum(Rm2[ii,jj,kk,ll]*X[ii]*Y[jj]*X[kk]*Y[ll]
              for ii in range(2) for jj in range(2) for kk in range(2) for ll in range(2))
    denom = gXX*gYY - gXY**2
    return num/denom

results2 = []
for a0 in np.linspace(0, np.pi, 20):
    for b0 in np.linspace(0, np.pi, 20):
        res = minimize(neg_K2, [a0, b0], method='Nelder-Mead',
                       options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 10000})
        results2.append((res.fun, res.x))

min_result2 = min(results2, key=lambda x: x[0])
print(f"\nSymmetric point ({pt2[0]:.4f}, {pt2[1]:.4f}, {pt2[2]:.4f}, {pt2[3]:.4f}), s = {s_check:.6f}")
print(f"  Optimized K_min = {min_result2[0]:+.10e}")
print(f"  At angles (α, β) = ({np.degrees(min_result2[1][0]):.4f}°, {np.degrees(min_result2[1][1]):.4f}°)")

# Key question: are these the same (as SO(3) symmetry predicts)?
print(f"\n  ΔK_min = {abs(min_result[0] - min_result2[0]):.4e}")
print(f"  Same within FD tolerance? {abs(min_result[0] - min_result2[0]) < 1e-5}")
