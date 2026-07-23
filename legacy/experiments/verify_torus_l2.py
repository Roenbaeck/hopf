"""
Verify the torus argument for ℓ≥2 seams.

For s = Y₂⁰(x₁)Y₂⁰(x₂) = (3z₁²-1)(3z₂²-1),
the reflection (x,y,z) -> (x,-y,z) preserves s.
Fixed set: {y=0} × {y=0} = great circle × great circle = T².

Parametrise the great circle {y=0} by theta: x = sin θ, z = cos θ.
s|_T² = (3cos²θ₁ - 1)(3cos²θ₂ - 1).

Check: does K = 0 on this torus, or just ∫K = 0?
"""
import sympy as sp

theta1, theta2, gamma = sp.symbols('theta1 theta2 gamma', real=True)

# Y₂⁰ restricted to {y=0}: f(θ) = 3cos²θ - 1
f1 = 3*sp.cos(theta1)**2 - 1
f2 = 3*sp.cos(theta2)**2 - 1
s = f1 * f2

print("s|_T² =", s)
print("s simplified =", sp.trigsimp(s))

# Background metric on the great circle: dθ²
# Covariant Hessian on S² restricted to the great circle {y=0, φ=0}:
# At φ=0 on S², the covariant Hessian in the θ direction is just d²s/dθ²
# (since Γ^θ_θθ = 0 on S²)
# BUT we need to be careful about the φ-direction.
# On the great circle {y=0}, we only have the θ-direction.
# So the torus T² has coords (θ₁, θ₂) and background metric dθ₁² + dθ₂².

# The seam metric on T²:
# g_θ₁θ₁ = 1 + γ ∂²s/∂θ₁²
# g_θ₁θ₂ = γ ∂²s/∂θ₁∂θ₂  
# g_θ₂θ₂ = 1 + γ ∂²s/∂θ₂²

H11 = sp.diff(s, theta1, 2)
H12 = sp.diff(s, theta1, theta2)  
H22 = sp.diff(s, theta2, 2)

# Brioschi numerators for g = (1+γH₁₁, γH₁₂; γH₁₂, 1+γH₂₂)
g11 = 1 + gamma * H11
g12 = gamma * H12
g22 = 1 + gamma * H22

num1 = sp.diff(g22, theta1) - sp.diff(g12, theta2)
num2 = sp.diff(g11, theta2) - sp.diff(g12, theta1)

print("\nBrioschi num1 =", sp.expand(num1))
print("Brioschi num2 =", sp.expand(num2))

# This is a separable seam s = f(θ₁)g(θ₂), so by the separability
# theorem, both numerators should be zero!
print("\nnum1 == 0?", sp.simplify(num1) == 0)
print("num2 == 0?", sp.simplify(num2) == 0)

# Now try a NON-separable ℓ=2 seam on a torus.
# s = Y₂¹(x₁) Y₂¹(x₂) = (x₁z₁)(x₂z₂) = sin θ₁ cos θ₁ cos φ₁ · sin θ₂ cos θ₂ cos φ₂
# Under (x,y,z) -> (x,y,-z): z_k -> -z_k, so x_k z_k -> -x_k z_k
# s -> (-x₁z₁)(-x₂z₂) = s. Invariant! ✓
# Fixed set: {z=0} × {z=0} = equator × equator

phi1, phi2 = sp.symbols('phi1 phi2', real=True)
# At equator θ=π/2: x = cos φ, y = sin φ, z = 0
# s = (cos φ₁ · 0)(cos φ₂ · 0) = 0

print("\n=== Y₂¹ × Y₂¹ on equator torus ===")
print("s|_T² = 0 (seam vanishes on the fixed torus)")
print("So g|_T² = h|_T² (round product metric on T²)")
print("K = 0 trivially!")

# Try the reflection (x,y,z) -> (-x,y,z) for Y₂¹ = xz
# Under x -> -x: xz -> -xz. s = (xz)(xz) -> (-xz)(-xz) = s. Invariant! ✓
# Fixed set: {x=0} × {x=0}. Great circle in y-z plane.
# Parametrise: y = sin θ, z = cos θ (with x=0).
# s|_T² = (0 · cos θ₁)(0 · cos θ₂) = 0. Again zero!

print("\n=== Y₂¹ × Y₂¹ on {x=0} torus ===")
print("s|_T² = 0 again (xz vanishes when x=0)")

# What about Y₂² = x²-y²?
# Under (x,y,z) -> (x,y,-z): x²-y² -> x²-y². Invariant! ✓
# Fixed set: {z=0} × {z=0} = equator × equator
# s|_T² = (cos²φ₁ - sin²φ₁)(cos²φ₂ - sin²φ₂) = cos(2φ₁)cos(2φ₂)

print("\n=== Y₂² × Y₂² on equator torus ===")
s_Y22 = sp.cos(2*phi1) * sp.cos(2*phi2)
print("s|_T² =", s_Y22, "(separable!)")

H11_22 = sp.diff(s_Y22, phi1, 2)
H12_22 = sp.diff(s_Y22, phi1, phi2)
H22_22 = sp.diff(s_Y22, phi2, 2)

g11_22 = 1 + gamma * H11_22
g12_22 = gamma * H12_22
g22_22 = 1 + gamma * H22_22

num1_22 = sp.diff(g22_22, phi1) - sp.diff(g12_22, phi2)
num2_22 = sp.diff(g11_22, phi2) - sp.diff(g12_22, phi1)

print("Brioschi num1 =", sp.expand(num1_22))
print("Brioschi num2 =", sp.expand(num2_22))
print("Both zero (separable):", sp.simplify(num1_22)==0 and sp.simplify(num2_22)==0)
