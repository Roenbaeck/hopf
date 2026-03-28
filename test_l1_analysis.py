#!/usr/bin/env python3
"""
Analyze why l=1 seams give K <= 0 on S^2 x S^2.

KEY PROPERTY of l=1 spherical harmonics f in {x, y, z}:
  nabla^2 f = -f * g_{S^2}   (Hessian is conformal)

For a product seam s = sum a_{ij} f_i(unit1) * g_j(unit2):

  H_{ab}    = -s * (g_1)_{ab}       (within factor 1)
  H_{alpha,beta} = -s * (g_2)_{alpha,beta}  (within factor 2)
  H_{a,alpha} = sum a_{ij} (nabla_a f_i)(nabla_alpha g_j)  (mixed)

So the seam metric g = h + gamma*H has:
  g_{ab}    = (1 - gamma*s) * (g_1)_{ab}     -- conformal on factor 1
  g_{alpha,beta} = (1 - gamma*s) * (g_2)_{alpha,beta} -- conformal on factor 2
  g_{a,alpha} = gamma * sum a_{ij} (nabla_a f_i)(nabla_alpha g_j)  -- cross

This is a CONFORMALLY WARPED product with cross terms.
Let u = 1 - gamma*s. Then:
  g = u * (g_1 + g_2) + gamma * M
where M_{a,alpha} is the mixed part.

The mixed curvature K(e_a, e_alpha) of such a metric has a known structure.
Let's derive it.
"""

import numpy as np
import sympy as sp
from sympy import symbols, cos, sin, sqrt, simplify, diff, Matrix, Rational

print("=" * 70)
print("Analytical structure of l=1 Hessian seam on S^2 x S^2")
print("=" * 70)

# First, let's verify the Hessian property numerically
print("\n--- Verify nabla^2(Y_1^m) = -Y_1^m * g on S^2 ---")
print("For f = sin(theta)cos(phi) = x, on S^2 with metric dth^2 + sin^2(th) dph^2:")
print("  H_{th,th} = f_{,thth}  = -sin(th)cos(ph) = -f")  
print("  H_{th,ph} = f_{,thph} - cot(th)*f_{,ph} = -cos(th)sin(ph) - cot(th)*(-sin(th)sin(ph))")
print("            = -cos(th)sin(ph) + cos(th)sin(ph) = 0")
print("  H_{ph,ph} = f_{,phph} + sin(th)cos(th)*f_{,th} = -sin(th)cos(ph) + sin(th)cos(th)*cos(th)cos(ph)")
print("            = -sin(th)cos(ph) + cos^2(th)*sin(th)*cos(ph)/sin(th)... wait let me redo")
print()
print("  H_{ph,ph} = f_{,phph} + sin(th)cos(th)*f_{,th}")
print("  f_{,ph} = -sin(th)sin(ph),  f_{,phph} = -sin(th)cos(ph)")
print("  f_{,th} = cos(th)cos(ph)")
print("  H_{ph,ph} = -sin(th)cos(ph) + sin(th)cos(th)*cos(th)cos(ph)")
print("            = cos(ph)[-sin(th) + sin(th)cos^2(th)]")
print("            = cos(ph)*sin(th)*[-1 + cos^2(th)]")
print("            = -cos(ph)*sin(th)*sin^2(th)")
print("            = -f * sin^2(th) = -f * g_{ph,ph}")
print()
print("  So H = -f * diag(1, sin^2(th)) = -f * g_{S^2}. Confirmed!")

print()
print("--- Structure of l=1 seam metric ---")
print()
print("For s = sum a_{ij} f_i(unit1) * h_j(unit2), the metric g = h + gamma*Hess(s):")
print()
print("  g_{ab} = (1 - gamma*s) * (h_1)_{ab}          [conformal on factor 1]")
print("  g_{alpha,beta} = (1 - gamma*s) * (h_2)_{alpha,beta}  [conformal on factor 2]")
print("  g_{a,alpha} = gamma * sum a_{ij} (nabla_a f_i)(nabla_alpha h_j)  [mixed]")
print()
print("Let u = 1 - gamma*s and M_{a,alpha} = gamma * sum a_{ij} (nabla_a f_i)(nabla_alpha h_j)")
print()
print("The full 4x4 metric is:")
print("  g = [[u*h_1,  M], [M^T, u*h_2]]")
print()
print("For small gamma, u ≈ 1 and M = O(gamma), so this is a perturbation of the product.")

print()
print("--- Mixed curvature of conformally warped product ---")
print()
print("For g = u*(h_1 + h_2) (pure conformal, no mixed terms):")
print("  K_mix(e_a, e_alpha) = -(1/2u) * [e_a(e_a(u)) + e_alpha(e_alpha(u))]")
print("                       + (1/4u^2) * [|nabla u|^2]")
print("  (This is the standard formula for conformal change of product metric.)")
print("  Since u = 1 - gamma*s and s = sum a_{ij}*f_i*h_j:")
print("  nabla_a nabla_a u = -gamma * nabla_a nabla_a s = -gamma*(-s) = gamma*s")
print("  (using the l=1 Hessian property within factor 1)")
print("  Similarly nabla_alpha nabla_alpha u = gamma*s (within factor 2)")
print("  So the e_a e_a(u) + e_alpha e_alpha(u) terms are related to gamma*s.")
print()
print("But we also have the mixed block M, which modifies the curvature.")
print("The question: does the M contribution make K more negative or can it make K positive?")

print()
print("=" * 70)
print("Numerical experiment: decompose K into conformal and cross parts")
print("=" * 70)

# Import the builder from previous test
from test_Q_S2xS2 import make_seam_product_harmonics, build_metric_and_curvature

N = 12
trim = 2
sl = tuple(slice(trim, -trim) for _ in range(4))

# Test: compare full K to pure conformal K (no cross terms)
# For a pure conformal metric g = u*h, the mixed curvature is known:
# K_mix = -1/(2u) * (Delta_1(u)/h_1 + Delta_2(u)/h_2) ... but this is for orthonormal frames

# Let me compute K with and without the cross block M

def build_metric_variants(seam_func, gamma, N=12):
    """Build full metric and diagonal-only metric to isolate cross-term effect."""
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

    # Verify: for l=1, H_{ab} should be -s * g1_{ab} and H_{alpha,beta} = -s * g2_{alpha,beta}
    print(f"  Verify conformal: H[0,0] / (-S) = {np.mean(H[0,0][sl] / (-S[sl])):.6f} (should be ~1.0 = g1_00)")
    print(f"  Verify conformal: H[1,1] / (-S * sin^2(th1)) = {np.mean(H[1,1][sl] / (-S[sl] * st1[sl]**2)):.6f} (should be ~1.0)")
    print(f"  Verify conformal: H[2,2] / (-S) = {np.mean(H[2,2][sl] / (-S[sl])):.6f} (should be ~1.0 = g2_00)")
    print(f"  Verify conformal: H[3,3] / (-S * sin^2(th2)) = {np.mean(H[3,3][sl] / (-S[sl] * st2[sl]**2)):.6f} (should be ~1.0)")
    
    # Check that mixed terms H_{0,2}, H_{0,3}, H_{1,2}, H_{1,3} are NOT -s*h
    print(f"  Mixed H[0,2] range: [{np.min(H[0,2][sl]):.4e}, {np.max(H[0,2][sl]):.4e}]")
    print(f"  Mixed H[0,3] range: [{np.min(H[0,3][sl]):.4e}, {np.max(H[0,3][sl]):.4e}]")
    
    # Full metric
    G_full = gamma * H.copy()
    G_full[0,0] += 1.0; G_full[1,1] += st1**2; G_full[2,2] += 1.0; G_full[3,3] += st2**2

    # Diagonal-only metric (set cross terms to zero)
    G_diag = np.zeros_like(G_full)
    G_diag[0,0] = G_full[0,0]; G_diag[0,1] = G_full[0,1]; G_diag[1,0] = G_full[1,0]; G_diag[1,1] = G_full[1,1]
    G_diag[2,2] = G_full[2,2]; G_diag[2,3] = G_full[2,3]; G_diag[3,2] = G_full[3,2]; G_diag[3,3] = G_full[3,3]
    # Cross terms G_diag[0,2], etc. = 0

    # For l=1, the diagonal part is g = u*h where u = 1 - gamma*s
    # Verify:
    u = 1 - gamma * S
    print(f"  G_diag[0,0] vs u*1: max diff = {np.max(np.abs(G_diag[0,0][sl] - u[sl])):.4e}")
    print(f"  G_diag[1,1] vs u*sin^2: max diff = {np.max(np.abs(G_diag[1,1][sl] - u[sl]*st1[sl]**2)):.4e}")

    def compute_K(G, dxs, shape):
        def fd(arr, axis):
            return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2*dxs[axis])
        dG = np.zeros((4, 4, 4) + shape)
        for k in range(4):
            for i in range(4):
                for j in range(i, 4):
                    dG[i,j,k] = fd(G[i,j], k); dG[j,i,k] = dG[i,j,k]
        Gf = G.reshape(4, 4, -1).transpose(2, 0, 1)
        Gif = np.linalg.inv(Gf)
        Ginv = Gif.transpose(1, 2, 0).reshape(4, 4, *shape)
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
        R_0202 = np.zeros(shape)
        for m in range(4):
            Rm = dChr[m,0,2,2] - dChr[m,2,2,0]
            for p in range(4):
                Rm += Chr[m,2,p]*Chr[p,0,2] - Chr[m,0,p]*Chr[p,2,2]
            R_0202 += G[0,m] * Rm
        denom = G[0,0]*G[2,2] - G[0,2]**2
        return R_0202 / denom

    K_full = compute_K(G_full, dxs, shape)
    K_diag = compute_K(G_diag, dxs, shape)

    return K_full, K_diag, S, u


gamma = 0.05  # not too small so cross terms are visible

print("\n--- l=1 seam: x1*x2 ---")
sf = make_seam_product_harmonics([1,0,0, 0,0,0, 0,0,0])
K_full, K_diag, S, u = build_metric_variants(sf, gamma, N)
Kf, Kd = K_full[sl], K_diag[sl]
print(f"  Full K:     [{np.min(Kf):.6e}, {np.max(Kf):.6e}]")
print(f"  Diag K:     [{np.min(Kd):.6e}, {np.max(Kd):.6e}]")
print(f"  K is all <= 0? Full: {np.max(Kf) <= 1e-12}, Diag: {np.max(Kd) <= 1e-12}")

print("\n--- l=1 seam: x1*y2 + z1*z2 ---")
sf = make_seam_product_harmonics([0,1,0, 0,0,0, 0,0,1])
K_full, K_diag, S, u = build_metric_variants(sf, gamma, N)
Kf, Kd = K_full[sl], K_diag[sl]
print(f"  Full K:     [{np.min(Kf):.6e}, {np.max(Kf):.6e}]")
print(f"  Diag K:     [{np.min(Kd):.6e}, {np.max(Kd):.6e}]")
print(f"  K is all <= 0? Full: {np.max(Kf) <= 1e-12}, Diag: {np.max(Kd) <= 1e-12}")

print("\n--- l=1 random seam ---")
rng = np.random.default_rng(42)
c = rng.normal(size=9).tolist()
sf = make_seam_product_harmonics(c)
K_full, K_diag, S, u = build_metric_variants(sf, gamma, N)
Kf, Kd = K_full[sl], K_diag[sl]
print(f"  Full K:     [{np.min(Kf):.6e}, {np.max(Kf):.6e}]")
print(f"  Diag K:     [{np.min(Kd):.6e}, {np.max(Kd):.6e}]")
print(f"  K is all <= 0? Full: {np.max(Kf) <= 1e-12}, Diag: {np.max(Kd) <= 1e-12}")

# For the diagonal (conformal) case, the curvature should follow the known formula:
# K_mix(e_0, e_2) for g = u*h on product S^2 x S^2:
# K_mix = -u_{;00}/(2u) - u_{;22}/(2u) + ... cross gradient terms
# But on S^2 x S^2, for u = 1 - gamma*s with Hess(s)_{within} = -s*h:
#   nabla_0 nabla_0 u = -gamma * nabla_0 nabla_0 s = -gamma * (-s * 1) = gamma*s
# in the theta direction (g1_{00} = 1).
# So u_{;00} = gamma*s, u_{;22} = gamma*s.
# The conformal curvature formula for product:
#   K_mix = -(u_{;00} + u_{;22})/(2u) + |nabla u|^2 /(4u^2) (for DIM=4)
# Wait, the precise formula depends on normalization. Let me use the standard one.

# For g = e^{2phi} h on n-manifold:
#   K_g(X,Y) = K_h(X,Y) - (nabla_X nabla_X phi + nabla_Y nabla_Y phi)
#              + (X(phi)^2 + Y(phi)^2) - |nabla phi|^2
# where everything is in h-orthonormal frames.

# For u = e^{2phi}, phi = (1/2)ln(u) = (1/2)ln(1-gamma*s) ≈ -gamma*s/2
# K_h_mix = 0 on S^2 x S^2
# nabla_X nabla_X phi ≈ -gamma/2 * nabla_X nabla_X s + O(gamma^2)
# For l=1 within factor: nabla_X nabla_X s = -s*g_XX = -s if X is unit
# So nabla_X nabla_X phi ≈ gamma*s/2
# Similarly nabla_Y nabla_Y phi ≈ gamma*s/2
# X(phi) ≈ -gamma/2 * X(s), Y(phi) ≈ -gamma/2 * Y(s)
# |nabla phi|^2 ≈ gamma^2/4 * |nabla s|^2

# K_g ≈ -(gamma*s/2 + gamma*s/2) + gamma^2/4*(X(s)^2 + Y(s)^2) - gamma^2/4*|nabla s|^2
#      = -gamma*s + gamma^2/4*(X(s)^2 + Y(s)^2 - |nabla s|^2)
#      ≤ -gamma*s  (since X(s)^2 + Y(s)^2 ≤ |nabla s|^2)

# Wait, this gives K ≈ -gamma*s, which is not necessarily ≤ 0!
# If s > 0 somewhere, K < 0 there; if s < 0, K > 0.
# But the conformal metric requires u = 1-gamma*s > 0, which for gamma>0
# means s < 1/gamma. The sign of s is unrestricted.

# Hmm, but the NUMERICAL results show K ≤ 0 for the FULL metric, not just diagonal.
# Let me check what's happening more carefully.

print("\n" + "=" * 70)
print("Conformal curvature formula check")
print("=" * 70)

# For the diagonal-only case (pure conformal g = u*h), compare numerical K
# with the analytical formula.
sf = make_seam_product_harmonics([1,0,0, 0,0,0, 0,0,0])  # s = x1*x2
eps_th = 0.15
th1 = np.linspace(eps_th, np.pi - eps_th, N)
ph1 = np.linspace(0, 2*np.pi, N, endpoint=False)
th2 = np.linspace(eps_th, np.pi - eps_th, N)
ph2 = np.linspace(0, 2*np.pi, N, endpoint=False)
T1, P1, T2, P2 = np.meshgrid(th1, ph1, th2, ph2, indexing='ij')
st1, ct1 = np.sin(T1), np.cos(T1)
st2, ct2 = np.sin(T2), np.cos(T2)

S_val = st1*np.cos(P1) * st2*np.cos(P2)  # s = x1*x2
u_val = 1 - gamma * S_val
phi_val = 0.5 * np.log(u_val)  # u = e^{2 phi}

# Gradient components of s in coordinate basis:
# s,0 = cos(th1)cos(ph1)*sin(th2)cos(ph2)
# s,1 = -sin(th1)sin(ph1)*sin(th2)cos(ph2)
# s,2 = sin(th1)cos(ph1)*cos(th2)cos(ph2)
# s,3 = -sin(th1)cos(ph1)*sin(th2)sin(ph2)
dS_0 = ct1*np.cos(P1) * st2*np.cos(P2)
dS_1 = -st1*np.sin(P1) * st2*np.cos(P2)
dS_2 = st1*np.cos(P1) * ct2*np.cos(P2)
dS_3 = -st1*np.cos(P1) * st2*np.sin(P2)

# |grad s|^2_h = (s,0)^2 + (s,1)^2/sin^2(th1) + (s,2)^2 + (s,3)^2/sin^2(th2)
grad_s_sq = dS_0**2 + dS_1**2/st1**2 + dS_2**2 + dS_3**2/st2**2

# phi = ln(u)/2, so phi,i = -gamma/(2u) * s,i
# phi_{;00} = d/dth1 (phi,0) = -gamma/(2u) * s,00 + gamma/(2u^2)*gamma*s,0^2 ... complex
# More directly: use the formula for conformal product K_mix.

# For g = u*h on M1 x M2 (dim 4), with K_h(X,Y) = 0 for mixed:
# In dim n=4 with g = u*h, using f = sqrt(u):
#   K_g(X,Y) = K_h(X,Y) - 1/(2u) * (u_{;XX} + u_{;YY}) + 1/(4u^2) * (u_X^2 + u_Y^2)
# Actually the precise formula for K of conformal metric g_{ij} = u * h_{ij}:
# Write u = e^{2φ}. Then (for orthonormal frame in h):
#   K_g(e_i, e_j) = e^{-2φ}[K_h(e_i,e_j) - (φ_{;ii} + φ_{;jj}) - (n-2)|∇φ|^2 + (φ_i^2 + φ_j^2)]
# Wait, different sources have different conventions. Let me use the formula
# for g_ij = e^{2φ} h_ij in dimension n:
# Sec_g(X,Y) = e^{-2φ}[Sec_h(X,Y) - (Hess_h φ(X,X) + Hess_h φ(Y,Y)) 
#                       + |dφ(X)|^2 + |dφ(Y)|^2 - |∇φ|^2_h]

# For dim n=4, product metric, K_h=0 for mixed, φ = ln(u)/2:
# Sec_g(e_0, e_2) = e^{-2φ}[-(Hess φ)_{00} - (Hess φ)_{22} + φ_0^2 + φ_2^2 - |∇φ|^2]

# φ = -gamma*s/2 + O(gamma^2)
# Hess φ = -gamma/2 * Hess s + O(gamma^2) = -gamma/2 * (-s*h_within) = gamma*s/2 * h_within
# So (Hess φ)_{00} = gamma*s/2 (for e_0 = d/dtheta1, unit vector in h)
#    (Hess φ)_{22} = gamma*s/2
# φ_0 = -gamma*s_0/2, φ_2 = -gamma*s_2/2
# |∇φ|^2 = gamma^2/4 * |∇s|^2

# K_g ≈ (1+gamma*s)*[-gamma*s - gamma^2/4*(s_0^2 + s_2^2 - |∇s|^2)]... hmm

# This is getting messy. Let me just verify: what fraction of K comes from
# the cross terms? If K_diag ≤ 0 already, the cross terms just add to negativity.
# If K_diag changes sign but K_full ≤ 0, then the cross terms enforce negativity.

print("\n--- Sign diagnostics ---")
print(f"  s range: [{np.min(S_val[sl]):.4f}, {np.max(S_val[sl]):.4f}]")
print(f"  u range: [{np.min(u_val[sl]):.4f}, {np.max(u_val[sl]):.4f}]")

print("\n" + "=" * 70)
print("More systematic test: K for l=1 at various gamma values")
print("=" * 70)

for gam in [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]:
    sf = make_seam_product_harmonics([1,0,0, 0,0,0, 0,0,0])
    K, dV, shape = build_metric_and_curvature(sf, gam, N)
    Ki = K[sl]
    print(f"  gamma={gam:.2f}: K in [{np.min(Ki):.6e}, {np.max(Ki):.6e}], K<=0: {np.max(Ki) <= 1e-10}")

print("\n--- Same for random l=1 seam (larger coefficients) ---")
c = rng.normal(size=9)
c = (c / np.max(np.abs(c))).tolist()  # normalize
sf = make_seam_product_harmonics(c)
for gam in [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]:
    K, dV, shape = build_metric_and_curvature(sf, gam, N)
    Ki = K[sl]
    metric_ok = "PD ok" if gam < 0.95 else "PD?"
    print(f"  gamma={gam:.2f}: K in [{np.min(Ki):.6e}, {np.max(Ki):.6e}], K<=0: {np.max(Ki) <= 1e-10}")

print("\n" + "=" * 70)
print("Key test: what makes l>=2 different?")
print("For l=2, nabla^2 Y_2^m != -c * Y_2^m * g. The Hessian is NOT conformal.")
print("=" * 70)

# For f = xz = sin(th)cos(ph)cos(th), the Hessian on S^2:
# H_{th,th} = f_{,thth} = d/dth[cos(2th)cos(ph)] = -2sin(2th)cos(ph)
# But -f*g_{00} = -sin(th)cos(ph)cos(th)*1 = -sin(th)cos(th)cos(ph)
# These are NOT equal: -2sin(2th) = -4sin(th)cos(th) != -sin(th)cos(th)
# So the l=2 Hessian has a traceless part.

print("\nFor f = xz = sin(th)cos(th)cos(ph) (l=2):")
print("  H_{th,th} = -2sin(2th)cos(ph) = -4sin(th)cos(th)cos(ph)")
print("  -f*g_{th,th} = -sin(th)cos(th)cos(ph)")
print("  Ratio: H_{thth}/(-f) = 4 != 1. NOT conformal.")
print()
print("For l=2: nabla^2 Y_2^m = -l(l+1) Y_2^m * g + (traceless part)")
print("         nabla^2 Y_2^m = -6 Y_2^m * g + traceless symmetric tensor")
print("The traceless part is what breaks the K <= 0 property.")
print("Specifically, the diagonal blocks g_{ab} and g_{alpha,beta} are no longer")
print("conformal to the background — they have anisotropy from the traceless Hessian.")
