"""
Verify the totally geodesic S¹×S¹ argument from FEEDBACK.

Key claims to verify:
1. R_P × R_P preserves the seam s = x·y for any plane P
2. The induced metric on N_P = (S²∩P)×(S²∩P) ≅ S¹×S¹ is flat
3. The K=0 plane matches the weak SVD direction
4. Whether the argument extends to distinct-σ seams
"""
import numpy as np
import sympy as sp

print("=" * 60)
print("PART 1: Verify induced metric on N_P is flat (symbolic)")
print("=" * 60)

t1, t2, gamma = sp.symbols('theta_1 theta_2 gamma', real=True)

# On S¹×S¹ (great circle × great circle in the xy-plane, say):
# x = (cos θ₁, sin θ₁, 0), y = (cos θ₂, sin θ₂, 0)
# s = x·y = cos(θ₁-θ₂)

s = sp.cos(t1 - t2)

# Background metric on S¹×S¹: h = dθ₁² + dθ₂² (flat torus)
# No Christoffel symbols on the flat torus.
# Hessian = ordinary second derivatives:
H = sp.Matrix([
    [sp.diff(s, t1, t1), sp.diff(s, t1, t2)],
    [sp.diff(s, t2, t1), sp.diff(s, t2, t2)]
])

print(f"s = cos(θ₁ - θ₂)")
print(f"H[0,0] = {sp.simplify(H[0,0])}")
print(f"H[0,1] = {sp.simplify(H[0,1])}")
print(f"H[1,1] = {sp.simplify(H[1,1])}")

# Induced metric g = I + γH
G = sp.eye(2) + gamma * H
print(f"\nInduced metric:")
print(f"g[0,0] = {sp.simplify(G[0,0])}")
print(f"g[0,1] = {sp.simplify(G[0,1])}")
print(f"g[1,1] = {sp.simplify(G[1,1])}")

# Change to (u, v) = (θ₁-θ₂, θ₁+θ₂)
u, v = sp.symbols('u v', real=True)
# θ₁ = (u+v)/2, θ₂ = (v-u)/2
# dθ₁ = (du+dv)/2, dθ₂ = (dv-du)/2

# Jacobian: dθ/d(u,v)
J = sp.Matrix([[sp.Rational(1,2), sp.Rational(1,2)],
               [sp.Rational(-1,2), sp.Rational(1,2)]])

# g in (u,v) coords: g_uv = J^T G J, but G has θ₁,θ₂ dependence
# Substitute s = cos(u) in G
G_u = G.subs(t1 - t2, u)
# More carefully:
G_explicit = sp.Matrix([
    [1 + gamma*(-sp.cos(u)), gamma*sp.cos(u)],
    [gamma*sp.cos(u), 1 + gamma*(-sp.cos(u))]
])

G_uv = J.T * G_explicit * J
G_uv_simplified = sp.simplify(G_uv)
print(f"\nMetric in (u,v) coordinates:")
print(f"g_uu = {sp.simplify(G_uv_simplified[0,0])}")
print(f"g_uv = {sp.simplify(G_uv_simplified[0,1])}")
print(f"g_vv = {sp.simplify(G_uv_simplified[1,1])}")

# This should be diagonal with g_uu depending only on u, g_vv = 1/2
# => product metric => flat!
is_diagonal = sp.simplify(G_uv_simplified[0,1]) == 0
g_vv_const = sp.simplify(sp.diff(G_uv_simplified[1,1], u)) == 0

print(f"\nDiagonal? g_uv = 0: {is_diagonal}")
print(f"g_vv constant in u? {g_vv_const}")
if is_diagonal and g_vv_const:
    print("✓ Product metric in (u,v) => Gaussian curvature = 0 identically!")
else:
    print("✗ NOT a product metric")

# Double-check: compute Gaussian curvature of g_uu(u) du² + g_vv dv²
g_uu = G_uv_simplified[0,0]
g_vv = G_uv_simplified[1,1]
# K = -1/(2√(g_uu g_vv)) [∂_u(∂_u(g_vv)/√(g_uu g_vv)) + ∂_v(∂_v(g_uu)/√(g_uu g_vv))]
# Since g_vv doesn't depend on u, and g_uu doesn't depend on v:
# ∂_u(g_vv) = 0 and ∂_v(g_uu) = 0
# => K = 0 trivially.
print("  (∂_u g_vv = 0 and ∂_v g_uu = 0 => K = 0 by Brioschi)")

print("\n" + "=" * 60)
print("PART 2: Verify coupling matrix on the K=0 plane")
print("=" * 60)

# At a generic point x = (cos α, sin α, 0), y = (cos β, sin β, 0)
# with x ≠ ±y, the normal to P = span(x,y) is n = (0,0,1).
# The in-plane tangents: e₁ = n × x = (-sin α, cos α, 0)
#                         f₁ = n × y = (-sin β, cos β, 0)
# M(e₁, f₁) = e₁ · f₁ = sin α sin β + cos α cos β = cos(α-β) = s
# The out-of-plane: e₂ = n = (0,0,1), f₂ = n = (0,0,1)
# M(e₂,f₂) = 1

alpha, beta = sp.symbols('alpha beta', real=True)
e1 = sp.Matrix([- sp.sin(alpha), sp.cos(alpha), 0])
f1 = sp.Matrix([- sp.sin(beta), sp.cos(beta), 0])
n = sp.Matrix([0, 0, 1])

M_e1_f1 = e1.dot(f1)
M_n_n = n.dot(n)
M_e1_n = e1.dot(n)  # should be 0
M_n_f1 = n.dot(f1)  # should be 0

s_val = sp.cos(alpha - beta)

print(f"M(e₁,f₁) = {sp.simplify(M_e1_f1)} = cos(α-β) = s? {sp.simplify(M_e1_f1 - s_val) == 0}")
print(f"M(e₂,f₂) = {M_n_n}")
print(f"M(e₁,f₂) = {M_e1_n} (should be 0)")
print(f"M(e₂,f₁) = {M_n_f1} (should be 0)")
print("✓ Coupling matrix = diag(s, 1) => weak direction is (e₁, f₁) with σ = |s|")

print("\n" + "=" * 60)
print("PART 3: Does the reflection argument extend to distinct-σ?")
print("=" * 60)

# For s = σ₁x₁x₂ + σ₂y₁y₂ + σ₃z₁z₂, we need R_P^T Σ R_P = Σ
# where Σ = diag(σ₁, σ₂, σ₃)
# R_P is a reflection (orthogonal with det = -1)

sig1, sig2, sig3 = sp.symbols('sigma_1 sigma_2 sigma_3', positive=True)
Sigma = sp.diag(sig1, sig2, sig3)

# Reflection across xy-plane: R = diag(1, 1, -1)
R_xy = sp.diag(1, 1, -1)
check_xy = sp.simplify(R_xy * Sigma * R_xy - Sigma)
print(f"R_xy commutes with Σ? {check_xy == sp.zeros(3)}")

# Reflection across a general plane with normal n = (a,b,c):
# R_n = I - 2 n n^T
a, b, c = sp.symbols('a b c', real=True)
n_vec = sp.Matrix([a, b, c])
R_n = sp.eye(3) - 2 * n_vec * n_vec.T  # assuming |n|=1

# R_n^T Σ R_n = Σ requires (I - 2nnT)Σ(I - 2nnT) = Σ
# Expanding: Σ - 2nnTΣ - 2ΣnnT + 4n(nTΣn)nT = Σ
# => -2nnTΣ - 2ΣnnT + 4n(nTΣn)nT = 0
# => -2(nTΣ) - 2(Σn)nT + 4(nTΣn)nnT = 0 ... complex

# Try specific non-coordinate plane for distinct σ:
# n = (1/√2, 1/√2, 0), R reflects across the plane x=y
sig_vals = {sig1: 3, sig2: 2, sig3: 1}  # distinct values
n_test = sp.Matrix([1, 1, 0]) / sp.sqrt(2)
R_test = sp.eye(3) - 2 * n_test * n_test.T
Sigma_num = Sigma.subs(sig_vals)
commutator = sp.simplify(R_test * Sigma_num * R_test - Sigma_num)
print(f"\nDistinct σ = (3,2,1), reflection across x=y plane:")
print(f"R Σ R - Σ = {commutator}")
preserves = commutator == sp.zeros(3)
print(f"Preserves seam? {preserves}")

# Coordinate plane reflections always work:
print(f"\nCoordinate plane reflections (always work for diagonal Σ):")
for name, R in [("xy-plane (z→-z)", sp.diag(1,1,-1)), 
                ("xz-plane (y→-y)", sp.diag(1,-1,1)),
                ("yz-plane (x→-x)", sp.diag(-1,1,1))]:
    comm = sp.simplify(R * Sigma * R - Sigma)
    print(f"  {name}: preserves seam? {comm == sp.zeros(3)}")

print(f"\nFor distinct σ, only 3 coordinate-plane reflections work.")
print(f"Each gives a single totally geodesic S¹×S¹ (equator × equator).")
print(f"These 3 tori don't cover all points of S²×S².")

# Check: does every point of S²×S² lie on at least one of these tori?
# Torus 1 (z=0): points with z₁=0 AND z₂=0
# Torus 2 (y=0): points with y₁=0 AND y₂=0  
# Torus 3 (x=0): points with x₁=0 AND x₂=0
# A generic point like (1,1,1)/√3 × (1,0,0) lies on NONE of these.
print(f"\nExample: point x=(1,1,1)/√3, y=(1,0,0)")
print(f"  z₁=1/√3 ≠ 0, y₁=1/√3 ≠ 0, x₁=1/√3 ≠ 0")
print(f"  → lies on NONE of the 3 totally geodesic tori")
print(f"  → reflection argument does NOT cover this point for distinct σ")

print("\n" + "=" * 60)
print("PART 4: Numerical verification of K=0 on N_P")
print("=" * 60)

# Verify using the full 4D curvature computation that K = 0
# for the tangent plane of N_P at various generic points

import sys
sys.path.insert(0, '/Users/lars/Documents/GitHub/hopf')
from test_Q_S2xS2 import make_seam_product_harmonics

def compute_K_for_plane(seam_func, gamma_val, t1, p1, t2, p2, e1_4d, e2_4d, h=1e-5):
    """Compute K for a specific mixed plane at a point."""
    coords = np.array([t1, p1, t2, p2])
    
    st1, ct1 = np.sin(t1), np.cos(t1)
    st2, ct2 = np.sin(t2), np.cos(t2)
    h_bg = np.diag([1.0, st1**2, 1.0, st2**2])
    
    def metric_at(c):
        st1_, ct1_ = np.sin(c[0]), np.cos(c[0])
        st2_, ct2_ = np.sin(c[2]), np.cos(c[2])
        h_ = np.diag([1.0, st1_**2, 1.0, st2_**2])
        s_, ds_, d2s_ = seam_func(c[0], c[1], c[2], c[3])
        H_ = np.zeros((4,4))
        H_[0,0] = d2s_[0][0]
        H_[0,1] = d2s_[0][1] - (ct1_/st1_)*ds_[1]
        H_[0,2] = d2s_[0][2]
        H_[0,3] = d2s_[0][3]
        H_[1,0] = H_[0,1]
        H_[1,1] = d2s_[1][1] + st1_*ct1_*ds_[0]
        H_[1,2] = d2s_[1][2]
        H_[1,3] = d2s_[1][3]
        H_[2,0] = H_[0,2]
        H_[2,1] = H_[1,2]
        H_[2,2] = d2s_[2][2]
        H_[2,3] = d2s_[2][3] - (ct2_/st2_)*ds_[3]
        H_[3,0] = H_[0,3]
        H_[3,1] = H_[1,3]
        H_[3,2] = H_[2,3]
        H_[3,3] = d2s_[3][3] + st2_*ct2_*ds_[2]
        return h_ + gamma_val * H_
    
    G0 = metric_at(coords)
    
    # dG/dx^k
    dG = np.zeros((4,4,4))
    for k in range(4):
        dt_p = np.zeros(4); dt_p[k] = h
        dt_m = np.zeros(4); dt_m[k] = -h
        dG[:,:,k] = (metric_at(coords+dt_p) - metric_at(coords+dt_m)) / (2*h)
    
    Ginv = np.linalg.inv(G0)
    
    Gamma = np.zeros((4,4,4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                Gamma[i,j,k] = 0.5 * sum(Ginv[i,l] * (dG[l,j,k] + dG[l,k,j] - dG[j,k,l]) for l in range(4))
    
    def christoffel_at(dt):
        c = coords + dt
        Gc = metric_at(c)
        Gc_inv = np.linalg.inv(Gc)
        dGc = np.zeros((4,4,4))
        for k in range(4):
            dtp = np.zeros(4); dtp[k] = h
            dtm = np.zeros(4); dtm[k] = -h
            dGc[:,:,k] = (metric_at(c+dtp) - metric_at(c+dtm)) / (2*h)
        Gam = np.zeros((4,4,4))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    Gam[i,j,k] = 0.5 * sum(Gc_inv[i,l] * (dGc[l,j,k] + dGc[l,k,j] - dGc[j,k,l]) for l in range(4))
        return Gam
    
    dGamma = np.zeros((4,4,4,4))
    for l in range(4):
        dt_p = np.zeros(4); dt_p[l] = h
        dt_m = np.zeros(4); dt_m[l] = -h
        dGamma[:,:,:,l] = (christoffel_at(dt_p) - christoffel_at(dt_m)) / (2*h)
    
    R = np.zeros((4,4,4,4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    R[i,j,k,l] = dGamma[i,j,l,k] - dGamma[i,j,k,l]
                    for m in range(4):
                        R[i,j,k,l] += Gamma[i,k,m]*Gamma[m,j,l] - Gamma[i,l,m]*Gamma[m,j,k]
    
    Riem = np.zeros((4,4,4,4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    Riem[i,j,k,l] = sum(G0[i,m]*R[m,j,k,l] for m in range(4))
    
    num = sum(Riem[a,b,c,d] * e1_4d[a] * e2_4d[b] * e1_4d[c] * e2_4d[d]
              for a in range(4) for b in range(4) for c in range(4) for d in range(4))
    g11 = e1_4d @ G0 @ e1_4d
    g22 = e2_4d @ G0 @ e2_4d
    g12 = e1_4d @ G0 @ e2_4d
    denom = g11*g22 - g12**2
    
    return num/denom if abs(denom) > 1e-15 else 0.0

# Full-rank seam s = n₁·n₂
coeffs_full = [1,0,0, 0,1,0, 0,0,1]
seam_full = make_seam_product_harmonics(coeffs_full)

# Test at several generic points
# Point: x = (sinα cosβ, sinα sinβ, cosα) parameterized by (α, β)
# The tangent to N_P at this point:
# If P = span(x, y), n = x×y / |x×y|
# e₁ = n×x (tangent to S² at x, in-plane)
# In spherical coords, this tangent has components (dθ, dφ)

gamma_val = 0.1
h_fd = 1e-5

print("\nFull-rank seam s = n₁·n₂, γ = 0.1:")
print(f"{'Point':>40} {'s':>8} {'K(N_P plane)':>14} {'K_min(all)':>14}")

test_points = [
    (0.8, 0.0, 1.5, 0.0),
    (1.0, 0.5, 2.0, 1.3),
    (0.5, 1.0, 1.2, 2.5),
    (1.2, 0.3, 0.7, 1.8),
    (np.pi/4, 0.0, np.pi/3, 0.0),
]

for t1_v, p1_v, t2_v, p2_v in test_points:
    # Compute the embedding coordinates
    x = np.array([np.sin(t1_v)*np.cos(p1_v), np.sin(t1_v)*np.sin(p1_v), np.cos(t1_v)])
    y = np.array([np.sin(t2_v)*np.cos(p2_v), np.sin(t2_v)*np.sin(p2_v), np.cos(t2_v)])
    s_val = x @ y
    
    # Normal to P = span(x,y)
    n_vec = np.cross(x, y)
    n_norm = np.linalg.norm(n_vec)
    if n_norm < 1e-10:
        print(f"  ({t1_v:.1f},{p1_v:.1f},{t2_v:.1f},{p2_v:.1f}): x ≈ ±y, skip")
        continue
    n_vec = n_vec / n_norm
    
    # In-plane tangent at x: e₁ = n×x
    e1_3d = np.cross(n_vec, x)
    e1_3d = e1_3d / np.linalg.norm(e1_3d)
    
    # In-plane tangent at y: f₁ = n×y
    f1_3d = np.cross(n_vec, y)
    f1_3d = f1_3d / np.linalg.norm(f1_3d)
    
    # Convert to spherical coordinate tangent vectors
    # For x = (sinθ cosφ, sinθ sinφ, cosθ):
    # ∂/∂θ = (cosθ cosφ, cosθ sinφ, -sinθ)
    # ∂/∂φ = (-sinθ sinφ, sinθ cosφ, 0)
    
    ct1, st1 = np.cos(t1_v), np.sin(t1_v)
    cp1, sp1 = np.cos(p1_v), np.sin(p1_v)
    ct2, st2 = np.cos(t2_v), np.sin(t2_v)
    cp2, sp2 = np.cos(p2_v), np.sin(p2_v)
    
    dtheta1 = np.array([ct1*cp1, ct1*sp1, -st1])
    dphi1 = np.array([-st1*sp1, st1*cp1, 0])
    
    dtheta2 = np.array([ct2*cp2, ct2*sp2, -st2])
    dphi2 = np.array([-st2*sp2, st2*cp2, 0])
    
    # Express e₁ in (∂θ₁, ∂φ₁) basis:
    # e₁ · ∂θ₁ gives θ-component, but we need to account for metric
    # In the orthonormal basis, ∂θ has unit length, ∂φ has length sinθ
    e1_th1 = e1_3d @ dtheta1  # coeff of ∂/∂θ₁
    e1_ph1 = e1_3d @ dphi1    # coeff of sinθ₁ · (∂/∂φ₁ / sinθ₁)
    # Actually dphi1 already has magnitude sinθ₁
    # e₁ = a ∂/∂θ₁ + b ∂/∂φ₁ => e₁_3d = a·dtheta1 + b·dphi1
    # Solve: [dtheta1 | dphi1]^T [dtheta1 | dphi1] [a; b] = [dtheta1 | dphi1]^T e1_3d
    A_mat = np.column_stack([dtheta1, dphi1])
    coeffs_e1 = np.linalg.lstsq(A_mat, e1_3d, rcond=None)[0]
    
    A_mat2 = np.column_stack([dtheta2, dphi2])
    coeffs_f1 = np.linalg.lstsq(A_mat2, f1_3d, rcond=None)[0]
    
    # 4D tangent vectors: (∂θ₁, ∂φ₁, ∂θ₂, ∂φ₂)
    e1_4d = np.array([coeffs_e1[0], coeffs_e1[1], 0, 0])
    e2_4d = np.array([0, 0, coeffs_f1[0], coeffs_f1[1]])
    
    K_NP = compute_K_for_plane(seam_full, gamma_val, t1_v, p1_v, t2_v, p2_v, e1_4d, e2_4d, h=h_fd)
    
    # Also compute K_min over all mixed planes
    K_min = 1e30
    n_ang = 90
    for i in range(n_ang):
        a_ang = np.pi * i / n_ang
        v1 = np.array([np.cos(a_ang), np.sin(a_ang), 0, 0])
        for j in range(n_ang):
            b_ang = np.pi * j / n_ang
            v2 = np.array([0, 0, np.cos(b_ang), np.sin(b_ang)])
            K_val = compute_K_for_plane(seam_full, gamma_val, t1_v, p1_v, t2_v, p2_v, v1, v2, h=h_fd)
            K_min = min(K_min, K_val)
    
    print(f"  ({t1_v:.1f},{p1_v:.1f},{t2_v:.1f},{p2_v:.1f})  s={s_val:+.4f}  K(N_P)={K_NP:+.4e}  K_min={K_min:+.4e}")

print("\n" + "=" * 60)
print("PART 5: Does argument work for distinct σ = (3,2,1)?")
print("=" * 60)

# For distinct σ, only coordinate-plane reflections work
# Fixed point of R_{xy} × R_{xy}: equator × equator (z₁=z₂=0)
# Test: is the equatorial torus still flat?

# s = 3x₁x₂ + 2y₁y₂ + z₁z₂ at z=0:
# s|_{z=0} = 3cos(φ₁)cos(φ₂) + 2sin(φ₁)sin(φ₂)
# On the equator (θ=π/2): s = 3cosφ₁cosφ₂ + 2sinφ₁sinφ₂

coeffs_distinct = [3,0,0, 0,2,0, 0,0,1]
seam_distinct = make_seam_product_harmonics(coeffs_distinct)

# Point on equatorial torus: θ₁ = θ₂ = π/2, φ₁ = 0.5, φ₂ = 1.3
t1_eq, t2_eq = np.pi/2, np.pi/2
p1_eq, p2_eq = 0.5, 1.3

# The N_P tangent for the xy-plane is ∂/∂φ₁ and ∂/∂φ₂
e1_eq = np.array([0, 1, 0, 0])  # ∂/∂φ₁
e2_eq = np.array([0, 0, 0, 1])  # ∂/∂φ₂

K_eq = compute_K_for_plane(seam_distinct, gamma_val, t1_eq, p1_eq, t2_eq, p2_eq, e1_eq, e2_eq, h=h_fd)
print(f"Distinct σ=(3,2,1), equatorial torus, K(∂φ₁, ∂φ₂) = {K_eq:+.6e}")

# But at a generic (non-equatorial) point, no torus passes through:
t1_gen, p1_gen, t2_gen, p2_gen = 1.0, 0.5, 2.0, 1.3
K_min_gen = 1e30
n_ang = 90
for i in range(n_ang):
    a_ang = np.pi * i / n_ang
    v1 = np.array([np.cos(a_ang), np.sin(a_ang), 0, 0])
    for j in range(n_ang):
        b_ang = np.pi * j / n_ang
        v2 = np.array([0, 0, np.cos(b_ang), np.sin(b_ang)])
        K_val = compute_K_for_plane(seam_distinct, gamma_val, t1_gen, p1_gen, t2_gen, p2_gen, v1, v2, h=h_fd)
        K_min_gen = min(K_min_gen, K_val)

print(f"Distinct σ=(3,2,1), generic point, K_min = {K_min_gen:+.6e}")
print(f"  (No totally geodesic torus passes through generic points for distinct σ)")
