#!/usr/bin/env python3
"""
Symbolic computation of mixed curvature for the simplest l=1 seam:
  s = cos(theta1) * cos(theta2)  (= z1 * z2)

Metric structure:
  g = h + gamma * Hess(s)
  
With l=1 conformal property:
  g_00 = 1 - gamma*s = u
  g_11 = u * sin^2(th1)
  g_22 = u
  g_33 = u * sin^2(th2)
  g_02 = gamma * sin(th1) * sin(th2)   [the cross term]
  all others = 0

Goal: compute K(d/dth1, d/dth2) exactly and show K <= 0.
"""

import sympy as sp
from sympy import symbols, cos, sin, sqrt, simplify, diff, Matrix, Rational, Function, trigsimp, factor, expand

th1, th2, gam = symbols('theta_1 theta_2 gamma', real=True, positive=True)

s1, c1 = sin(th1), cos(th1)
s2, c2 = sin(th2), cos(th2)

# Seam function
s = c1 * c2
u = 1 - gam * s

# 4x4 metric tensor g[i,j], coordinates = (th1, ph1, th2, ph2)
# Due to phi-independence of s = z1*z2, the metric is block-diagonal
# in (th1,th2) and (ph1,ph2) and also independent of ph1, ph2.
# This means Christoffels involving phi are simple.

# Actually, let's exploit the simplification: R_{0202} for this metric
# depends only on the (th1, th2) 2d subspace plus the warping of phi1, phi2.
# But since we're computing full 4D sectional curvature, we need all 4D.

# Let me set up the full 4x4 metric symbolically.
# Coords: x0=th1, x1=ph1, x2=th2, x3=ph2

n = 4
g = sp.zeros(n, n)
g[0, 0] = u
g[1, 1] = u * s1**2
g[2, 2] = u
g[3, 3] = u * s2**2
g[0, 2] = gam * s1 * s2
g[2, 0] = gam * s1 * s2

coords = [th1, symbols('phi_1'), th2, symbols('phi_2')]

print("Metric tensor g:")
sp.pprint(g)

# Inverse metric
ginv = g.inv()
print("\nInverse metric g^{ij}:")
ginv_simplified = sp.zeros(n, n)
for i in range(n):
    for j in range(n):
        ginv_simplified[i, j] = trigsimp(ginv[i, j])
sp.pprint(ginv_simplified)

# Determinant
det_g = trigsimp(g.det())
print(f"\ndet(g) = {det_g}")

# Since the metric doesn't depend on phi_1, phi_2, many Christoffels vanish.
# Also, the (th1,th2) block and (ph1,ph2) block are decoupled in the derivatives.
# Let's compute only the Christoffel symbols we need.

# For R_{0202}: need Gamma^m_{02}, Gamma^m_{22}, Gamma^m_{00}
# and their derivatives w.r.t. th1 (x0) and th2 (x2).

# Christoffel symbols
print("\nComputing Christoffel symbols...")
Chr = {}
for m in range(n):
    for i in range(n):
        for j in range(i, n):
            val = sp.Rational(0)
            for k in range(n):
                val += ginv_simplified[m, k] * (diff(g[k, i], coords[j]) + diff(g[k, j], coords[i]) - diff(g[i, j], coords[k]))
            val = trigsimp(val / 2)
            if val != 0:
                Chr[(m, i, j)] = val
                if i != j:
                    Chr[(m, j, i)] = val

print(f"Non-zero Christoffel symbols: {len(Chr)}")
for (m, i, j), val in sorted(Chr.items()):
    if i <= j:
        print(f"  Gamma^{m}_{i}{j} = {val}")

# Riemann tensor R_{0202} = g_{0m} R^m_{202}
# R^m_{nrs} = dGamma^m_{ns}/dx^r - dGamma^m_{nr}/dx^s + Gamma^m_{rp}*Gamma^p_{ns} - Gamma^m_{sp}*Gamma^p_{nr}
# R_{0202} = sum_m g_{0m} * (dGamma^m_{02}/dx^0 - dGamma^m_{00}/dx^2 + sum_p [Gamma^m_{0p}*Gamma^p_{02} - Gamma^m_{2p}*Gamma^p_{00}])

print("\nComputing R_{0202}...")

def get_chr(m, i, j):
    return Chr.get((m, i, j), sp.Rational(0))

# R^m_{202} = dGamma^m_{02}/dth1 - dGamma^m_{00}/dth2 + sum_p [Gamma^m_{0p}*Gamma^p_{02} - Gamma^m_{2p}*Gamma^p_{00}]
R_up = {}
for m in range(n):
    term1 = diff(get_chr(m, 0, 2), coords[0])    # dGamma^m_{02}/dx^0
    term2 = -diff(get_chr(m, 0, 0), coords[2])    # -dGamma^m_{00}/dx^2
    term3 = sp.Rational(0)
    for p in range(n):
        term3 += get_chr(m, 0, p) * get_chr(p, 0, 2) - get_chr(m, 2, p) * get_chr(p, 0, 0)
    R_up[m] = trigsimp(term1 + term2 + term3)

R_0202 = sp.Rational(0)
for m in range(n):
    R_0202 += g[0, m] * R_up[m]
R_0202 = trigsimp(R_0202)
print(f"\nR_0202 = {R_0202}")

# Denominator for sectional curvature
denom = g[0, 0]*g[2, 2] - g[0, 2]**2
denom_simplified = trigsimp(expand(denom))
print(f"\ng_00*g_22 - g_02^2 = {denom_simplified}")

K = trigsimp(R_0202 / denom)
print(f"\nK = R_0202 / denom = {K}")

# Factor and simplify
print("\nAttempting to factor K...")
K_factored = factor(K)
print(f"K (factored) = {K_factored}")

# Check sign: K should be <= 0 when u > 0 and 0 < th1, th2 < pi
# Let's substitute some specific values to check
print("\n--- Numerical checks ---")
for t1_val, t2_val, g_val in [(sp.pi/4, sp.pi/4, sp.Rational(1,10)),
                                (sp.pi/3, sp.pi/6, sp.Rational(1,10)),
                                (sp.pi/2, sp.pi/2, sp.Rational(1,10)),
                                (sp.pi/4, sp.pi/4, sp.Rational(1,2))]:
    K_val = K_factored.subs([(th1, t1_val), (th2, t2_val), (gam, g_val)])
    print(f"  K(th1={t1_val}, th2={t2_val}, gamma={g_val}) = {sp.simplify(K_val)} ≈ {float(K_val):.6f}")

# Now let's also compute the leading-order K as gamma -> 0
# K = gamma^2 * Q + O(gamma^3)
print("\n--- Taylor expansion in gamma ---")
K_series = sp.series(R_0202 / denom, gam, 0, n=4)
print(f"K = {K_series}")

# Extract the gamma^2 coefficient
K_expanded = sp.Poly(sp.series(R_0202, gam, 0, n=4).removeO(), gam)
D_expanded = sp.Poly(sp.series(denom, gam, 0, n=4).removeO(), gam)
print(f"\nR_0202 coefficients in gamma: {K_expanded}")
print(f"denom coefficients in gamma: {D_expanded}")

# At order gamma^2: K ≈ R0202^(2) / denom^(0) where denom^(0) = g00^(0)*g22^(0) = 1*1 = 1
# since g02 = O(gamma)
# R_0202^(2) is the gamma^2 coefficient of R_0202

R_0202_expanded = sp.series(R_0202, gam, 0, n=4)
print(f"\nR_0202 expanded = {R_0202_expanded}")
