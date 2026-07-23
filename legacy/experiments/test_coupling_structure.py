#!/usr/bin/env python3
"""
Verify the geometric structure of the off-diagonal coupling block M = dn1 · dn2^T
for the full-rank l=1 seam s = n1.n2.

Theory: in orthonormal frames for each S^2 factor, M has singular values (1, |cos α|)
where cos α = n1 · n2 = s.

When s = 0 (n1 ⊥ n2): M has rank 1 → one uncoupled direction → K = 0 for that plane
When s = ±1 (n1 = ±n2): M has singular values (1,1) → coupling is conformal...
  but the diagonal block is (1-γs) = 1∓γ, so the full metric is a product → K = 0

Also verify: for RANK-1 seam z1z2, the off-diagonal M always has rank ≤ 1,
which explains why K(θ1,θ2) = 0 everywhere.
"""

import numpy as np

def jacobians(theta, phi):
    """Jacobians of (x,y,z) = n(theta,phi) w.r.t. (theta, phi)."""
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    # dn/dtheta = (ct*cp, ct*sp, -st)
    # dn/dphi = (-st*sp, st*cp, 0)
    J = np.array([[ct*cp, ct*sp, -st],
                   [-st*sp, st*cp, 0.0]])
    return J

def compute_M_svd(t1, p1, t2, p2):
    """Compute off-diagonal coupling M = J1 · J2^T and its SVD."""
    J1 = jacobians(t1, p1)
    J2 = jacobians(t2, p2)
    M = J1 @ J2.T  # 2x2 matrix in coordinate basis
    
    # Convert to orthonormal frame: divide rows by sqrt(h_aa)
    # h1 = diag(1, sin²θ₁), h2 = diag(1, sin²θ₂)
    st1, st2 = np.sin(t1), np.sin(t2)
    M_ortho = np.array([[M[0,0], M[0,1]/st2],
                         [M[1,0]/st1, M[1,1]/(st1*st2)]])
    
    U, svals, Vh = np.linalg.svd(M_ortho)
    return M, M_ortho, svals

print("=" * 70)
print("Singular values of M' (orthonormal frame) vs |s| = |n1·n2|")
print("Prediction: singular values = (1, |cos α|) where s = cos α")
print("=" * 70)

np.random.seed(42)
print(f"{'s':>10s}  {'σ₁':>10s}  {'σ₂':>10s}  {'|s|':>10s}  {'σ₂=|s|?':>10s}")
print("-" * 55)

for _ in range(20):
    t1 = np.random.uniform(0.3, np.pi - 0.3)
    p1 = np.random.uniform(0, 2*np.pi)
    t2 = np.random.uniform(0.3, np.pi - 0.3)
    p2 = np.random.uniform(0, 2*np.pi)
    
    n1 = np.array([np.sin(t1)*np.cos(p1), np.sin(t1)*np.sin(p1), np.cos(t1)])
    n2 = np.array([np.sin(t2)*np.cos(p2), np.sin(t2)*np.sin(p2), np.cos(t2)])
    s = n1 @ n2
    
    M, M_ortho, svals = compute_M_svd(t1, p1, t2, p2)
    match = abs(svals[1] - abs(s)) < 1e-10
    print(f"{s:+10.6f}  {svals[0]:10.6f}  {svals[1]:10.6f}  {abs(s):10.6f}  {'✓' if match else '✗'}")

# Special cases
print("\nSpecial cases:")
special = [
    (np.pi/4, np.pi/4, np.pi/4, np.pi/4, "n1=n2"),
    (np.pi/2, 0, np.pi/2, np.pi/2, "n1⊥n2"),
    (np.pi/4, 0, 3*np.pi/4, np.pi, "n1=-n2"),
    (np.pi/3, 0, np.pi/3, np.pi/2, "generic"),
]
for t1, p1, t2, p2, label in special:
    n1 = np.array([np.sin(t1)*np.cos(p1), np.sin(t1)*np.sin(p1), np.cos(t1)])
    n2 = np.array([np.sin(t2)*np.cos(p2), np.sin(t2)*np.sin(p2), np.cos(t2)])
    s = n1 @ n2
    M, M_ortho, svals = compute_M_svd(t1, p1, t2, p2)
    print(f"  {label:>10s}: s={s:+.6f}, σ=(${svals[0]:.6f}, {svals[1]:.6f}), |s|={abs(s):.6f}")

# Now verify: for z1z2 (rank-1 seam), the coupling matrix has rank ≤ 1 ALWAYS
print("\n" + "=" * 70)
print("z1z2 seam: off-diagonal coupling matrix")
print("=" * 70)
# s = z1*z2 = cos θ1 cos θ2
# ∂s/∂θ1 = -sin θ1 cos θ2, ∂s/∂φ1 = 0
# ∂s/∂θ2 = -cos θ1 sin θ2, ∂s/∂φ2 = 0
# Hessian cross block = ∂²s/∂(θ1,φ1)∂(θ2,φ2)
# H_02 = ∂²s/∂θ1∂θ2 = sin θ1 sin θ2
# H_03 = ∂²s/∂θ1∂φ2 = 0
# H_12 = ∂²s/∂φ1∂θ2 = 0
# H_13 = ∂²s/∂φ1∂φ2 = 0
# So the cross Hessian is [[sinθ1 sinθ2, 0], [0, 0]] → ALWAYS rank 1!
print("Cross Hessian for z1z2:")
print("  H_cross = [[sin θ₁ sin θ₂, 0], [0, 0]]")
print("  Always rank 1 (or 0 at poles)")
print("  → The off-diagonal block γ H_cross has rank ≤ 1")
print("  → (θ₁,θ₂) direction IS the only coupled direction in the cross block")
print()

# For z1z2: verify cross Hessian at a test point
for t1, p1, t2, p2 in [(np.pi/4, np.pi/4, np.pi/4, np.pi/4),
                         (1.0, 2.0, 1.5, 3.0)]:
    st1, st2 = np.sin(t1), np.sin(t2)
    H_cross = np.array([[st1*st2, 0], [0, 0]])
    print(f"  At ({t1:.2f},{p1:.2f},{t2:.2f},{p2:.2f}):")
    print(f"    H_cross = [[{H_cross[0,0]:.4f}, {H_cross[0,1]:.4f}], [{H_cross[1,0]:.4f}, {H_cross[1,1]:.4f}]]")
    print(f"    rank = {np.linalg.matrix_rank(H_cross)}")

# For n1.n2: verify cross Hessian is the same as J1 · J2^T
print("\n" + "=" * 70)
print("n1.n2 seam: verify cross Hessian = J1 · J2^T")
print("=" * 70)
# s = n1·n2 = sin θ₁ cos φ₁ sin θ₂ cos φ₂ + sin θ₁ sin φ₁ sin θ₂ sin φ₂ + cos θ₁ cos θ₂
# = sin θ₁ sin θ₂ cos(φ₁-φ₂) + cos θ₁ cos θ₂
# ∂²s/∂θ₁∂θ₂ = cos θ₁ cos θ₂ cos(φ₁-φ₂) - sin θ₁ sin θ₂
# ∂²s/∂θ₁∂φ₂ = cos θ₁ sin θ₂ sin(φ₁-φ₂)   [wait, -sin or +sin?]
# Let me just compute J1 · J2^T and compare with analytic Hessian

for t1, p1, t2, p2 in [(np.pi/4, 1.0, np.pi/3, 2.0),
                         (1.0, 2.0, 1.5, 3.0)]:
    J1 = jacobians(t1, p1)
    J2 = jacobians(t2, p2)
    M = J1 @ J2.T
    
    # Analytic Hessian of s = n1·n2 w.r.t. (θ₁,φ₁,θ₂,φ₂)
    # Using partial derivatives
    st1, ct1, sp1, cp1 = np.sin(t1), np.cos(t1), np.sin(p1), np.cos(p1)
    st2, ct2, sp2, cp2 = np.sin(t2), np.cos(t2), np.sin(p2), np.cos(p2)
    dp = p1 - p2
    cdp, sdp = np.cos(dp), np.sin(dp)
    
    # s = st1*st2*cdp + ct1*ct2
    # Cross Hessian (raw, before covariant correction):
    d2s_02 = ct1*ct2*cdp - st1*st2  # ∂²s/∂θ₁∂θ₂
    d2s_03 = -ct1*st2*sdp           # ∂²s/∂θ₁∂φ₂
    d2s_12 = -st1*ct2*sdp           # ∂²s/∂φ₁∂θ₂  (wait, let me compute)
    # ∂s/∂φ₁ = -st1*st2*sdp (since d/dφ₁ cos(φ₁-φ₂) = -sin(φ₁-φ₂))
    # ∂²(∂s/∂φ₁)/∂θ₂ = -st1*ct2*sdp
    # ∂²(∂s/∂φ₁)/∂φ₂ = -st1*st2*cdp  (since d/dφ₂(-sdp) = cdp... wait)
    # d/dφ₂ [-sin(φ₁-φ₂)] = cos(φ₁-φ₂)
    # so ∂²s/∂φ₁∂φ₂ = st1*st2*cdp
    d2s_13 = st1*st2*cdp
    
    H_cross = np.array([[d2s_02, d2s_03], [d2s_12, d2s_13]])
    
    print(f"\n  At ({t1:.2f},{p1:.2f},{t2:.2f},{p2:.2f}):")
    print(f"    J1·J2^T = [[{M[0,0]:.6f}, {M[0,1]:.6f}], [{M[1,0]:.6f}, {M[1,1]:.6f}]]")
    print(f"    H_cross = [[{H_cross[0,0]:.6f}, {H_cross[0,1]:.6f}], [{H_cross[1,0]:.6f}, {H_cross[1,1]:.6f}]]")
    print(f"    Match?  {np.allclose(M, H_cross)}")

# Summary of mechanism
print("\n" + "=" * 70)
print("MECHANISM SUMMARY")
print("=" * 70)
print("""
For the full-rank seam s = n₁·n₂:

1. The off-diagonal coupling block in orthonormal frames has
   singular values (1, |s|) where s = n₁·n₂.

2. At s = 0 (n₁ ⊥ n₂): one singular value vanishes → rank 1
   → one mixed plane is completely decoupled → K = 0 for that plane

3. At s = ±1 (n₁ = ±n₂): the metric becomes a product metric
   k₁(S²) × k₂(S²) for different scale factors
   → ALL mixed K = 0

4. At generic s ∈ (-1,1)\\{0}: both singular values > 0
   → all mixed planes are coupled → K > 0

The zero locus of K is the set:
  {s = 0} (codim 1 in S²×S²) × {specific plane} (codim 1 in Gr(2,4))
  ∪ {s = ±1} (codim 2 in S²×S²) × {all mixed planes}

For rank-1 seams (like z₁z₂):
  The cross Hessian always has rank ≤ 1
  → K = 0 in the decoupled plane at EVERY point
  → K > 0 only for planes that include the coupled direction
""")
