"""
Verify the totally geodesic torus argument for distinct-sigma ℓ=1 seams.

For s = σ₁x₁x₂ + σ₂y₁y₂ + σ₃z₁z₂ with the reflection (x,y,z)->(x,y,-z),
the fixed set is {z=0}×{z=0} = equator × equator.

Parametrise: x_k = (cos φ_k, sin φ_k, 0), so
  s|_T² = σ₁ cos φ₁ cos φ₂ + σ₂ sin φ₁ sin φ₂

The metric on the torus is g|_T² = h|_T² + γ ∇² s|_T².

Check: is the intrinsic curvature of (T², g|_T²) identically zero?
If not, does ∫K dA = 0 by Gauss-Bonnet?
"""
import numpy as np
from sympy import *

phi1, phi2, gamma_s = symbols('phi1 phi2 gamma', real=True)
sig1, sig2, sig3 = symbols('sigma1 sigma2 sigma3', real=True, positive=True)

# Seam restricted to equator×equator (z=0)
s = sig1 * cos(phi1) * cos(phi2) + sig2 * sin(phi1) * sin(phi2)
print("s|_T² =", s)

# Background metric on equator×equator: dφ₁² + dφ₂²
# (sin²θ_k = 1 at equator)
# Covariant Hessian restricted to the torus:
# On S² at equator (θ=π/2), the coordinate φ parametrises the equator.
# The metric is sin²θ dφ² = dφ² at equator.
# The covariant Hessian of s restricted to the torus is just d²s/dφ_k dφ_l
# (the Christoffel symbol on the equator circle is zero: Γ^φ_φφ = 0
#  because the metric sin²θ is constant along φ at fixed θ=π/2).

H11 = diff(s, phi1, 2)  # ∂²s/∂φ₁²
H12 = diff(s, phi1, phi2)  # ∂²s/∂φ₁∂φ₂
H22 = diff(s, phi2, 2)  # ∂²s/∂φ₂²

print("H₁₁ =", H11)
print("H₁₂ =", H12)
print("H₂₂ =", H22)

# Wait - we need the FULL covariant Hessian, not just in φ directions.
# At the equator θ=π/2, the Hessian of s on S² has the covariant form:
# (∇²s)_φφ = s_φφ + sin θ cos θ s_θ
# But at θ=π/2: sin θ cos θ = 0, so (∇²s)_φφ = s_φφ ✓
# For the cross terms: (∇²s)_φ₁φ₂ = s_φ₁φ₂ (no Christoffel connection between factors) ✓

# Metric on torus: g = (1 + γH₁₁)dφ₁² + 2γH₁₂ dφ₁dφ₂ + (1 + γH₂₂)dφ₂²
g11 = 1 + gamma_s * H11
g12 = gamma_s * H12
g22 = 1 + gamma_s * H22

print("\ng₁₁ =", g11)
print("g₁₂ =", g12)
print("g₂₂ =", g22)

# Gaussian curvature via Brioschi formula for 2D metric g(φ₁,φ₂)
det_g = g11*g22 - g12**2
det_g_simplified = simplify(det_g)
print("\ndet g =", det_g_simplified)

# Brioschi: K = -1/(2√(det g)) [ ∂₁((g₂₂,₁ - g₁₂,₂)/√(det g)) + ∂₂((g₁₁,₂ - g₁₂,₁)/√(det g)) ]
g22_1 = diff(g22, phi1)
g12_2 = diff(g12, phi2)
g11_2 = diff(g11, phi2)
g12_1 = diff(g12, phi1)

num1 = g22_1 - g12_2
num2 = g11_2 - g12_1

print("\nBrioschi numerator 1 (g₂₂,₁ - g₁₂,₂) =", simplify(num1))
print("Brioschi numerator 2 (g₁₁,₂ - g₁₂,₁) =", simplify(num2))

# If both numerators are zero, K = 0 identically!
print("\nnum1 == 0?", simplify(num1).equals(S(0)))
print("num2 == 0?", simplify(num2).equals(S(0)))

# Let's also expand to be sure
print("\nnum1 expanded:", expand(num1))
print("num2 expanded:", expand(num2))

# For the isotropic case σ₁ = σ₂ = σ₃ = 1:
print("\n=== Isotropic case σ₁=σ₂=1 ===")
s_iso = cos(phi1)*cos(phi2) + sin(phi1)*sin(phi2)
print("s_iso =", trigsimp(s_iso), "= cos(φ₁ - φ₂)")

# For general σ: check if it's still a product (u,v coordinates)
u = phi1 - phi2
v = phi1 + phi2
print("\n=== Change to u = φ₁-φ₂, v = φ₁+φ₂ ===")
s_uv = s.subs(phi1, (u+v)/2).subs(phi2, (v-u)/2)
s_uv = trigsimp(s_uv)
print("s(u,v) =", s_uv)

# Also: explicit computation for σ₁=3, σ₂=2, σ₃=1 (distinct)
print("\n=== Explicit: σ₁=3, σ₂=2, σ₃=1 ===")
s_explicit = 3*cos(phi1)*cos(phi2) + 2*sin(phi1)*sin(phi2)
H11_e = diff(s_explicit, phi1, 2)
H12_e = diff(s_explicit, phi1, phi2)
H22_e = diff(s_explicit, phi2, 2)
g11_e = 1 + gamma_s*H11_e
g12_e = gamma_s*H12_e
g22_e = 1 + gamma_s*H22_e

num1_e = diff(g22_e, phi1) - diff(g12_e, phi2)
num2_e = diff(g11_e, phi2) - diff(g12_e, phi1)
print("num1 =", expand(num1_e))
print("num2 =", expand(num2_e))
