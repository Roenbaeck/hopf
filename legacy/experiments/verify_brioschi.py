"""
Verify FEEDBACK contributions:
1. Brioschi proof: K=0 identically for rank-1 equivariant ℓ=1 seam
2. (δΓ)² cancellation: K = O(γ³) for all ℓ=1 seams
"""
import sympy as sp

print("=" * 60)
print("PART 1: Brioschi proof for s = cos(θ₁)cos(θ₂)")
print("=" * 60)

t1, t2, gamma = sp.symbols('theta_1 theta_2 gamma', real=True)

s = sp.cos(t1) * sp.cos(t2)

# Orbit metric components for g = h + γ∇²s
# Within-factor: conformal Hessian gives g_kk = (1 - γs) on each factor
# Cross: η = γ ∂²s/∂θ₁∂θ₂
lam = 1 - gamma * s  # λ₁ = λ₂ = 1 - γs (same for both factors)
eta = gamma * sp.diff(s, t1, t2)

print(f"s = {s}")
print(f"λ = {lam}")
print(f"η = {eta}")

# Verify the key derivative identities
d1_lam = sp.diff(lam, t1)
d2_eta = sp.diff(eta, t2)
d2_lam = sp.diff(lam, t2)
d1_eta = sp.diff(eta, t1)

print(f"\n∂₁λ = {sp.simplify(d1_lam)}")
print(f"∂₂η = {sp.simplify(d2_eta)}")
print(f"∂₁λ - ∂₂η = {sp.simplify(d1_lam - d2_eta)}")

print(f"\n∂₂λ = {sp.simplify(d2_lam)}")
print(f"∂₁η = {sp.simplify(d1_eta)}")
print(f"∂₂λ - ∂₁η = {sp.simplify(d2_lam - d1_eta)}")

# Now compute Brioschi formula directly
det_g = lam**2 - eta**2  # λ₁ = λ₂ = λ
sqrt_det = sp.sqrt(det_g)

# Numerators in Brioschi formula
num1 = d1_lam - d2_eta  # λ_{2,1} - η_{,2} = ∂₁λ - ∂₂η
num2 = d2_lam - d1_eta  # λ_{1,2} - η_{,1} = ∂₂λ - ∂₁η

print(f"\nBrioschi numerator 1 (λ_{{2,1}} - η_{{,2}}): {sp.simplify(num1)}")
print(f"Brioschi numerator 2 (λ_{{1,2}} - η_{{,1}}): {sp.simplify(num2)}")

if sp.simplify(num1) == 0 and sp.simplify(num2) == 0:
    print("\n✓ BOTH numerators vanish identically => K = 0 for ALL θ₁, θ₂, γ")
    print("  The Brioschi proof is VERIFIED symbolically.")
else:
    print("\n✗ Numerators do NOT vanish — checking full formula...")
    K = -1/(2*sqrt_det) * (sp.diff(num1/sqrt_det, t1) + sp.diff(num2/sqrt_det, t2))
    K_simplified = sp.simplify(K)
    print(f"  K = {K_simplified}")

# Algebraic source: s + ∂₂²s = 0
print(f"\nAlgebraic source: s + ∂₂²s = {sp.simplify(s + sp.diff(s, t2, t2))}")
print(f"Algebraic source: s + ∂₁²s = {sp.simplify(s + sp.diff(s, t1, t1))}")

print("\n" + "=" * 60)
print("PART 1b: Verify for general rank-1 ℓ=1 seams")
print("=" * 60)

# General rank-1: s = f(θ₁)g(θ₂) where f, g are ℓ=1 harmonics
# Using f = a·cos(θ₁) + b·sin(θ₁), g = c·cos(θ₂) + d·sin(θ₂)
a, b, c, d = sp.symbols('a b c d', real=True)
f1 = a * sp.cos(t1) + b * sp.sin(t1)
f2 = c * sp.cos(t2) + d * sp.sin(t2)
s_gen = f1 * f2

lam_gen = 1 - gamma * s_gen
eta_gen = gamma * sp.diff(s_gen, t1, t2)

num1_gen = sp.diff(lam_gen, t1) - sp.diff(eta_gen, t2)
num2_gen = sp.diff(lam_gen, t2) - sp.diff(eta_gen, t1)

print(f"General rank-1: s = ({f1})({f2})")
print(f"∂₁λ - ∂₂η = {sp.simplify(num1_gen)}")
print(f"∂₂λ - ∂₁η = {sp.simplify(num2_gen)}")

if sp.simplify(num1_gen) == 0 and sp.simplify(num2_gen) == 0:
    print("✓ K = 0 for ALL rank-1 ℓ=1 seams (general coefficients)")
else:
    print("✗ Does NOT vanish for general rank-1")

print("\n" + "=" * 60)
print("PART 2: (δΓ)² cancellation — Christoffel symbols")
print("=" * 60)

# For a general ℓ=1 product seam on S²×S², verify the 
# cross-factor Christoffel symbol formula:
#   δΓ^a_{bα} = -(γ/2) δ^a_b ∂_α s
#
# The full derivation uses third derivatives of s.
# For ℓ=1: ∇³s reduces to first derivatives via conformal Hessian.
#
# We verify by explicit computation on the metric.

# Use coords (θ₁, θ₂) for the orbit space (equivariant case)
# Full seam with distinct singular values:
sig1, sig2, sig3 = sp.symbols('sigma_1 sigma_2 sigma_3', positive=True)

# s = σ₁ x₁x₂ + σ₂ y₁y₂ + σ₃ z₁z₂
# In spherical coords:
# x = sinθ cosφ, y = sinθ sinφ, z = cosθ
# At φ₁ = φ₂ = 0 (by equivariance the orbit space analysis holds):
# s = σ₁ sinθ₁ sinθ₂ + σ₃ cosθ₁ cosθ₂
# (The σ₂ term involves cosφ and disappears in equivariant reduction at φ=0)
# Actually for the general non-equivariant case we need all 4 coords.

# Let's verify the (δΓ)² claim numerically instead.
# For a general ℓ=1 seam, compute K at various γ values and check scaling.

print("\nSwitching to numerical verification of K = O(γ³)...")
print("(See PART 3 below)")

print("\n" + "=" * 60)
print("PART 3: K = O(γ³) scaling test")
print("=" * 60)

import numpy as np
import sys
sys.path.insert(0, '/Users/lars/Documents/GitHub/hopf')

# Import the curvature computation machinery
from test_Q_S2xS2 import build_metric_and_curvature

def compute_Kmax_mixed(theta1, theta2, gamma_val, seam_coeffs, h=1e-5):
    """Compute maximum mixed K at a point for given γ."""
    g, Riem = build_metric_and_curvature(theta1, theta2, gamma_val, seam_coeffs, h=h)
    
    # Mixed curvature: span of (v1, v2) where v1 ∈ T_1, v2 ∈ T_2
    # Coordinates: (θ₁, φ₁, θ₂, φ₂) = indices (0,1,2,3)
    # The code computes R_{XYYX}/denom = -K (sign convention!)
    K_values = []
    n_angles = 60
    for i in range(n_angles):
        alpha = np.pi * i / n_angles
        e1 = np.array([np.cos(alpha), np.sin(alpha), 0, 0])
        for j in range(n_angles):
            beta = np.pi * j / n_angles
            e2 = np.array([0, 0, np.cos(beta), np.sin(beta)])
            
            num = sum(Riem[a,b,c,d] * e1[a] * e2[b] * e2[c] * e1[d]
                      for a in range(4) for b in range(4)
                      for c in range(4) for d in range(4))
            
            g1 = sum(g[a,b] * e1[a] * e1[b] for a in range(4) for b in range(4))
            g2 = sum(g[a,b] * e2[a] * e2[b] for a in range(4) for b in range(4))
            g12 = sum(g[a,b] * e1[a] * e2[b] for a in range(4) for b in range(4))
            denom = g1 * g2 - g12**2
            
            if abs(denom) > 1e-15:
                # Code computes R_{XYYX}/denom = -K, so K = -num/denom
                K = -num / denom
                K_values.append(K)
    
    return max(K_values), min(K_values)

# Test point
theta1, theta2 = 1.0, 2.0

# Full-rank seam: s = n₁·n₂ (σ₁=σ₂=σ₃=1)
seam_full = [(1, 0, 1, 0), (0, 1, 0, 1), (0, 0, 0, 0)]  # Will construct below

# Actually, let's use the seam_coeffs format from test_Q_S2xS2
# seam_coeffs = [(l1, m1, l2, m2, coeff), ...]
# For s = z₁z₂ = Y₁⁰·Y₁⁰ (up to normalization)
seam_z1z2 = [(1, 0, 1, 0, 1.0)]

# For s = x₁x₂ + y₁y₂ + z₁z₂ = n₁·n₂ 
# x = sinθ cosφ ~ Re(Y₁¹), y = sinθ sinφ ~ Im(Y₁¹), z = cosθ ~ Y₁⁰
# In terms of spherical harmonics:
# x₁x₂ ~ Re(Y₁¹)₁ Re(Y₁¹)₂
# y₁y₂ ~ Im(Y₁¹)₁ Im(Y₁¹)₂
# Need to check what format the code expects...

# Let me check what test_Q_S2xS2 expects
from test_Q_S2xS2 import seam_function, seam_gradient

# Test γ-scaling for z₁z₂ seam
print("Testing K scaling with γ for s = z₁z₂:")
gammas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
for gam in gammas:
    Kmax, Kmin = compute_Kmax_mixed(theta1, theta2, gam, seam_z1z2, h=1e-5)
    print(f"  γ={gam:.2f}: Kmax={Kmax:+.6e}, Kmin={Kmin:+.6e}, Kmax/γ²={Kmax/gam**2:+.6e}, Kmax/γ³={Kmax/gam**3:+.6e}")

# Also test a rank-2 seam (x₁x₂ + y₁y₂)
# x₁x₂ = sinθ₁cosφ₁ · sinθ₂cosφ₂ 
# y₁y₂ = sinθ₁sinφ₁ · sinθ₂sinφ₂
# x₁x₂ + y₁y₂ = sinθ₁sinθ₂(cosφ₁cosφ₂ + sinφ₁sinφ₂) = sinθ₁sinθ₂cos(φ₁-φ₂)
# In terms of Y₁^m: Y₁^{±1} ∝ sinθ e^{±iφ}
# Re(Y₁¹)₁·Re(Y₁¹)₂ + Im(Y₁¹)₁·Im(Y₁¹)₂ 
#   = sinθ₁cosφ₁·sinθ₂cosφ₂ + sinθ₁sinφ₁·sinθ₂sinφ₂ = sinθ₁sinθ₂cos(φ₁-φ₂)
seam_xy = [(1, 1, 1, 1, 1.0)]  # This encodes Re(Y₁¹)·Re(Y₁¹) + Im(Y₁¹)·Im(Y₁¹)

print("\nTesting K scaling with γ for s = x₁x₂ + y₁y₂:")
for gam in gammas:
    Kmax, Kmin = compute_Kmax_mixed(theta1, theta2, gam, seam_xy, h=1e-5)
    print(f"  γ={gam:.2f}: Kmax={Kmax:+.6e}, Kmin={Kmin:+.6e}, Kmax/γ²={Kmax/gam**2:+.6e}, Kmax/γ³={Kmax/gam**3:+.6e}")

# Full-rank: n₁·n₂ = x₁x₂ + y₁y₂ + z₁z₂
seam_full_rank = [(1, 0, 1, 0, 1.0), (1, 1, 1, 1, 1.0)]
print("\nTesting K scaling with γ for s = n₁·n₂ (full rank):")
for gam in gammas:
    Kmax, Kmin = compute_Kmax_mixed(theta1, theta2, gam, seam_full_rank, h=1e-5)
    print(f"  γ={gam:.2f}: Kmax={Kmax:+.6e}, Kmin={Kmin:+.6e}, Kmax/γ²={Kmax/gam**2:+.6e}, Kmax/γ³={Kmax/gam**3:+.6e}")

# For comparison: ℓ=2 seam (should be O(γ²))
seam_l2 = [(2, 0, 2, 0, 1.0)]
print("\nTesting K scaling with γ for ℓ=2 seam (should be O(γ²)):")
for gam in gammas:
    Kmax, Kmin = compute_Kmax_mixed(theta1, theta2, gam, seam_l2, h=1e-5)
    print(f"  γ={gam:.2f}: Kmax={Kmax:+.6e}, Kmin={Kmin:+.6e}, Kmax/γ²={Kmax/gam**2:+.6e}, Kmax/γ³={Kmax/gam**3:+.6e}")
