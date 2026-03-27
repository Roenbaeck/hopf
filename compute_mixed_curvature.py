"""
Compute the mixed sectional curvature R_{1313} for the O(2)xO(2)-equivariant
product seam metric on S^2 x S^2.

Metric: g = lambda1 dtheta1^2 + mu1 sin^2(theta1) dphi1^2
          + lambda2 dtheta2^2 + mu2 sin^2(theta2) dphi2^2
          + 2*eta dtheta1 dtheta2

where lambda1, lambda2, mu1, mu2, eta are functions of (theta1, theta2).

Coordinates: x^0 = theta1, x^1 = phi1, x^2 = theta2, x^3 = phi2
"""

import sympy as sp
from sympy import symbols, sin, cos, Function, Matrix, simplify, diff, Rational
from sympy import Symbol, sqrt, factor, collect, expand

# Coordinates
theta1, theta2 = symbols('theta1 theta2')
phi1, phi2 = symbols('phi1 phi2')

# Metric coefficient functions (depend on theta1, theta2)
lam1 = Function('lambda1')(theta1, theta2)
lam2 = Function('lambda2')(theta1, theta2)
mu1 = Function('mu1')(theta1, theta2)
mu2 = Function('mu2')(theta1, theta2)
eta = Function('eta')(theta1, theta2)

# Coordinates list
coords = [theta1, phi1, theta2, phi2]
n = 4

# Build metric tensor g_{ij}
# Indices: 0=theta1, 1=phi1, 2=theta2, 3=phi2
g = Matrix.zeros(n, n)
g[0, 0] = lam1
g[1, 1] = mu1 * sin(theta1)**2
g[2, 2] = lam2
g[3, 3] = mu2 * sin(theta2)**2
g[0, 2] = eta
g[2, 0] = eta

print("Metric tensor g_{ij}:")
sp.pprint(g)
print()

# Inverse metric
print("Computing inverse metric...")
ginv = g.inv()
ginv = sp.simplify(ginv)
print("Inverse metric g^{ij}:")
sp.pprint(ginv)
print()

# Christoffel symbols Gamma^i_{jk} = (1/2) g^{il} (g_{lj,k} + g_{lk,j} - g_{jk,l})
print("Computing Christoffel symbols...")

def christoffel(i, j, k):
    """Compute Gamma^i_{jk}"""
    val = 0
    for l in range(n):
        val += Rational(1, 2) * ginv[i, l] * (
            diff(g[l, j], coords[k]) +
            diff(g[l, k], coords[j]) -
            diff(g[j, k], coords[l])
        )
    return val

# Precompute all Christoffel symbols
Gamma = [[[None for _ in range(n)] for _ in range(n)] for _ in range(n)]
for i in range(n):
    for j in range(n):
        for k in range(j, n):  # symmetric in j,k
            val = christoffel(i, j, k)
            Gamma[i][j][k] = val
            Gamma[i][k][j] = val

# Riemann tensor R^i_{jkl} = d_k Gamma^i_{jl} - d_l Gamma^i_{jk}
#                             + Gamma^i_{km} Gamma^m_{jl} - Gamma^i_{lm} Gamma^m_{jk}
print("Computing R^0_{202} (= R^theta1_{theta2,theta1,theta2})...")

def riemann_updown(i, j, k, l):
    """Compute R^i_{jkl}"""
    val = diff(Gamma[i][j][l], coords[k]) - diff(Gamma[i][j][k], coords[l])
    for m in range(n):
        val += Gamma[i][k][m] * Gamma[m][j][l] - Gamma[i][l][m] * Gamma[m][j][k]
    return val

# We want R_{0202} = g_{0i} R^i_{202}
# = g_{00} R^0_{202} + g_{02} R^2_{202}    (only these since g is block-ish)

print("Computing R^0_{202}...")
R0_202 = riemann_updown(0, 2, 0, 2)
R0_202 = sp.simplify(R0_202)
print("R^0_{202} =")
sp.pprint(R0_202)
print()

print("Computing R^2_{202}...")
R2_202 = riemann_updown(2, 2, 0, 2)
R2_202 = sp.simplify(R2_202)
print("R^2_{202} =")
sp.pprint(R2_202)
print()

# R_{0202} = g_{00} R^0_{202} + g_{02} R^2_{202}
# But we also need contributions from other components
# R_{0202} = sum_i g_{0i} R^i_{202}
# = g_{00} R^0_{202} + g_{01} R^1_{202} + g_{02} R^2_{202} + g_{03} R^3_{202}
# g_{01} = 0, g_{03} = 0, so:
# R_{0202} = g_{00} R^0_{202} + g_{02} R^2_{202}
#          = lambda1 * R^0_{202} + eta * R^2_{202}

R_0202 = g[0, 0] * R0_202 + g[0, 2] * R2_202
R_0202 = sp.simplify(R_0202)
print("R_{0202} = R_{theta1,theta2,theta1,theta2} =")
sp.pprint(R_0202)
print()

# The denominator for sectional curvature: g_{00}g_{22} - g_{02}^2
denom = g[0, 0] * g[2, 2] - g[0, 2]**2
denom_simplified = sp.simplify(denom)
print("Denominator (lambda1*lambda2 - eta^2) =")
sp.pprint(denom_simplified)
print()

# Mixed sectional curvature K = R_0202 / denom
K_mix = R_0202 / denom
K_mix = sp.simplify(K_mix)
print("Mixed sectional curvature K(d_theta1, d_theta2) = R_{0202} / (lambda1*lambda2 - eta^2):")
sp.pprint(K_mix)
print()

# ===================================================================
# Now specialize to the DIAGONAL case (eta=0) for verification
# ===================================================================
print("=" * 60)
print("SPECIALISATION: eta = 0 (diagonal case)")
print("=" * 60)

# Also set mu1 = mu1(theta1), mu2 = mu2(theta2) for simplicity?
# No, keep general first. Just set eta = 0.
R_0202_diag = R_0202.subs(eta, 0)
R_0202_diag = sp.simplify(R_0202_diag)
print("R_{0202} with eta=0:")
sp.pprint(R_0202_diag)
print()

K_mix_diag = R_0202_diag / (lam1 * lam2)
K_mix_diag = sp.simplify(K_mix_diag)
print("K_mix with eta=0:")
sp.pprint(K_mix_diag)
print()

# ===================================================================
# Further specialise: mu1 = lambda1, mu2 = lambda2 (conformal product)
# ===================================================================
print("=" * 60)
print("SPECIALISATION: conformal product (mu_k = lambda_k, eta=0)")
print("=" * 60)

R_0202_conf = R_0202_diag.subs(mu1, lam1).subs(mu2, lam2)
R_0202_conf = sp.simplify(R_0202_conf)
print("R_{0202} conformal:")
sp.pprint(R_0202_conf)
print()

K_mix_conf = R_0202_conf / (lam1 * lam2)
K_mix_conf = sp.simplify(K_mix_conf)
print("K_mix conformal:")
sp.pprint(K_mix_conf)
print()

print("=" * 60)
print("FULL FORMULA for R_{0202} (with eta):")
print("=" * 60)
print()

# Try to collect terms to see the structure
# The key question: can we write R_0202 in a form that reveals
# it must be <= 0 at the maximum of log(lambda1*lambda2 - eta^2)?
R_0202_expanded = sp.expand(R_0202)
print("R_{0202} expanded:")
sp.pprint(R_0202_expanded)
