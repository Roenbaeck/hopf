"""
Verify/refute the (δΓ)² cancellation claim from FEEDBACK.

FEEDBACK claims:
1. The ONLY nonzero cross-factor Christoffel symbols at first order are:
   δΓ^a_{bα} = -(γ/2) δ^a_b ∂_α s
   δΓ^α_{bβ} = -(γ/2) δ^α_β ∂_b s

2. But what about Γ^a_{αβ} and Γ^α_{ab}? Let's check these.

We verify symbolically using SymPy.
"""
import sympy as sp

t1, t2, p1, p2, gamma = sp.symbols('theta_1 theta_2 phi_1 phi_2 gamma', real=True)

# ℓ=1 seam: s = cos(θ₁)cos(θ₂)
s = sp.cos(t1) * sp.cos(t2)

# Coordinates: (θ₁, φ₁, θ₂, φ₂) = (x⁰, x¹, x², x³)
coords = [t1, p1, t2, p2]

# Background metric h = diag(1, sin²θ₁, 1, sin²θ₂)
st1, st2 = sp.sin(t1), sp.sin(t2)
ct1, ct2 = sp.cos(t1), sp.cos(t2)

h = sp.Matrix([
    [1, 0, 0, 0],
    [0, st1**2, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, st2**2]
])

# Covariant Hessian of s on S²×S²
# ∇²s = ∂²s - Γ^k_{ij} ∂_k s
# Background Christoffels: Γ^0_{11} = -sinθ₁cosθ₁, Γ^1_{01} = cosθ₁/sinθ₁
#                          Γ^2_{33} = -sinθ₂cosθ₂, Γ^3_{23} = cosθ₂/sinθ₂

ds = [sp.diff(s, c) for c in coords]
d2s = [[sp.diff(s, coords[i], coords[j]) for j in range(4)] for i in range(4)]

H = sp.zeros(4, 4)
for i in range(4):
    for j in range(4):
        H[i, j] = d2s[i][j]

# Subtract Christoffel corrections
H[0, 1] -= (ct1/st1) * ds[1]
H[1, 0] -= (ct1/st1) * ds[1]
H[1, 1] += st1*ct1 * ds[0]
H[2, 3] -= (ct2/st2) * ds[3]
H[3, 2] -= (ct2/st2) * ds[3]
H[3, 3] += st2*ct2 * ds[2]

print("Covariant Hessian H:")
for i in range(4):
    for j in range(4):
        val = sp.simplify(H[i,j])
        if val != 0:
            print(f"  H[{i},{j}] = {val}")

# Full metric g = h + γH
G = h + gamma * H

print("\nFull metric g = h + γH:")
for i in range(4):
    for j in range(4):
        val = sp.simplify(G[i,j])
        if val != 0:
            print(f"  g[{i},{j}] = {val}")

# Compute Christoffel symbols of g at first order in γ
# Γ^i_{jk} = (1/2) g^{il} (∂_j g_{lk} + ∂_k g_{lj} - ∂_l g_{jk})

# At first order, g^{il} ≈ h^{il}
h_inv = h.inv()

# Metric derivatives
dG = [[[sp.diff(G[i,j], coords[k]) for k in range(4)] for j in range(4)] for i in range(4)]

# Christoffel symbols (first order in γ)
def christoffel_first_order(i, j, k):
    """Compute Γ^i_{jk} at first order in γ."""
    val = sp.S(0)
    for l in range(4):
        val += h_inv[i, l] * (dG[l][k][j] + dG[l][j][k] - dG[j][k][l])
    return sp.Rational(1, 2) * val

print("\n" + "=" * 60)
print("Cross-factor Christoffel symbols at first order in γ")
print("=" * 60)

# Factor 1 indices: 0, 1 (θ₁, φ₁)
# Factor 2 indices: 2, 3 (θ₂, φ₂)
factor1 = [0, 1]
factor2 = [2, 3]

print("\nType 1: Γ^a_{bα} (upper=factor1, lower=factor1×factor2)")
for a in factor1:
    for b in factor1:
        for alpha in factor2:
            val = sp.simplify(christoffel_first_order(a, b, alpha))
            if val != 0:
                # Check if it matches -(γ/2) δ^a_b ∂_α s
                expected = -gamma/2 * (1 if a == b else 0) * ds[alpha]
                diff = sp.simplify(val - expected)
                match = "✓" if diff == 0 else f"✗ diff={diff}"
                print(f"  Γ^{a}_{{{b}{alpha}}} = {val}  {match}")

print("\nType 2: Γ^α_{bβ} (upper=factor2, lower=factor1×factor2)")
for alpha in factor2:
    for b in factor1:
        for beta in factor2:
            val = sp.simplify(christoffel_first_order(alpha, b, beta))
            if val != 0:
                expected = -gamma/2 * (1 if alpha == beta else 0) * ds[b]
                diff = sp.simplify(val - expected)
                match = "✓" if diff == 0 else f"✗ diff={diff}"
                print(f"  Γ^{alpha}_{{{b}{beta}}} = {val}  {match}")

print("\nType 3: Γ^a_{αβ} (upper=factor1, lower=factor2×factor2)")
print("  (FEEDBACK claims these are zero — let's check)")
for a in factor1:
    for alpha in factor2:
        for beta in range(alpha, 4):  # β ≥ α
            if beta in factor2:
                val = sp.simplify(christoffel_first_order(a, alpha, beta))
                if val != 0:
                    print(f"  Γ^{a}_{{{alpha}{beta}}} = {val}  ← NONZERO!")
                else:
                    print(f"  Γ^{a}_{{{alpha}{beta}}} = 0")

print("\nType 4: Γ^α_{ab} (upper=factor2, lower=factor1×factor1)")
print("  (FEEDBACK claims these are zero — let's check)")
for alpha in factor2:
    for a in factor1:
        for b in range(a, 2):  # b ≥ a
            if b in factor1:
                val = sp.simplify(christoffel_first_order(alpha, a, b))
                if val != 0:
                    print(f"  Γ^{alpha}_{{{a}{b}}} = {val}  ← NONZERO!")
                else:
                    print(f"  Γ^{alpha}_{{{a}{b}}} = 0")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("If Type 3 or Type 4 are nonzero, the (δΓ)² cancellation")
print("argument is incomplete — it missed these terms.")
print("This would explain why K = O(γ²), not O(γ³), numerically.")
