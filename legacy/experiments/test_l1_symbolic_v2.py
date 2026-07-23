#!/usr/bin/env python3
"""
CORRECTED symbolic computation of mixed sectional curvature K(d/dth1, d/dth2)
for s = cos(th1)*cos(th2) (= z1*z2, l=1 seam).

Sign convention: K(X,Y) = R_{XYXY}/denom where R_{ijkl} = g_{im} R^m_{jkl}
and R^m_{jkl} = d_k Gamma^m_{jl} - d_l Gamma^m_{jk} + Gamma^m_{kp}Gamma^p_{jl} - Gamma^m_{lp}Gamma^p_{jk}

Verified: this gives K = +1 for the round S^2.
"""

import sympy as sp
from sympy import symbols, cos, sin, trigsimp, diff, factor, expand, Rational, series

th1, th2, gam = symbols('theta_1 theta_2 gamma', real=True, positive=True)

s1, c1 = sin(th1), cos(th1)
s2, c2 = sin(th2), cos(th2)

# Seam: s = cos(th1)*cos(th2)
s = c1 * c2
u = 1 - gam * s

# Metric: g = diag(u, u*sin^2(th1), u, u*sin^2(th2)) + off-diag g_{02} = gamma*sin(th1)*sin(th2)
n = 4
g = sp.zeros(n, n)
g[0, 0] = u
g[1, 1] = u * s1**2
g[2, 2] = u
g[3, 3] = u * s2**2
g[0, 2] = gam * s1 * s2
g[2, 0] = gam * s1 * s2

coords = [th1, symbols('phi_1'), th2, symbols('phi_2')]

# Inverse metric
ginv = g.inv()
ginv_s = sp.zeros(n, n)
for i in range(n):
    for j in range(n):
        ginv_s[i, j] = trigsimp(ginv[i, j])

# Christoffel symbols Gamma^m_{ij}
print("Computing Christoffel symbols...")
Chr = {}
for m in range(n):
    for i in range(n):
        for j in range(i, n):
            val = Rational(0)
            for k in range(n):
                val += ginv_s[m, k] * (diff(g[k, i], coords[j]) + diff(g[k, j], coords[i]) - diff(g[i, j], coords[k]))
            val = trigsimp(val / 2)
            if val != 0:
                Chr[(m, i, j)] = val
                if i != j:
                    Chr[(m, j, i)] = val

def get_chr(m, i, j):
    return Chr.get((m, i, j), Rational(0))

# Compute R^m_{jkl} = d_k Gamma^m_{jl} - d_l Gamma^m_{jk} + Gamma^m_{kp}Gamma^p_{jl} - Gamma^m_{lp}Gamma^p_{jk}
def compute_R_up(m, j, k, l):
    term1 = diff(get_chr(m, j, l), coords[k])
    term2 = -diff(get_chr(m, j, k), coords[l])
    term3 = Rational(0)
    for p in range(n):
        term3 += get_chr(m, k, p) * get_chr(p, j, l) - get_chr(m, l, p) * get_chr(p, j, k)
    return term1 + term2 + term3

# R_{ijkl} = g_{im} R^m_{jkl}
def compute_R(i, j, k, l):
    val = Rational(0)
    for m in range(n):
        val += g[i, m] * compute_R_up(m, j, k, l)
    return trigsimp(val)

# Sectional curvature K(e_0, e_2) = R_{0202} / (g_{00}*g_{22} - g_{02}^2)
print("Computing R_0202...")
R_0202 = compute_R(0, 2, 0, 2)
print(f"R_0202 = {R_0202}")

# Also compute R_0220 = -R_0202 as a check
print("Computing R_0220...")
R_0220 = compute_R(0, 2, 2, 0)
print(f"R_0220 = {R_0220}")
print(f"R_0202 + R_0220 = {trigsimp(R_0202 + R_0220)} (should be 0)")

denom = g[0,0]*g[2,2] - g[0,2]**2
denom_s = trigsimp(expand(denom))

K = trigsimp(R_0202 / denom)
print(f"\nK(e_0, e_2) = R_0202 / denom = {K}")

K_factored = factor(K)
print(f"K (factored) = {K_factored}")

# Sanity: within-factor curvature K(e_0, e_1) for gamma=0 should give +1
print("\n--- Sanity check: K(e_0, e_1) at gamma=0 ---")
R_0101 = compute_R(0, 1, 0, 1)
R_0101_at0 = trigsimp(R_0101.subs(gam, 0))
denom_01 = g[0,0]*g[1,1] - g[0,1]**2
denom_01_at0 = trigsimp(denom_01.subs(gam, 0))
K_01_at0 = trigsimp(R_0101_at0 / denom_01_at0)
print(f"R_0101(gamma=0) = {R_0101_at0}")
print(f"denom_01(gamma=0) = {denom_01_at0}")
print(f"K(e_0, e_1)(gamma=0) = {K_01_at0}  (should be +1)")

# Taylor expansion in gamma
print("\n--- Taylor expansion of K(e_0, e_2) ---")
K_taylor = sp.series(K, gam, 0, n=4)
print(f"K = {K_taylor}")

# Numerical checks
print("\n--- Numerical checks ---")
for t1_val, t2_val, g_val in [(sp.pi/4, sp.pi/4, Rational(1,10)),
                                (sp.pi/3, sp.pi/6, Rational(1,10)),
                                (sp.pi/2, sp.pi/2, Rational(1,10)),
                                (sp.pi/4, sp.pi/4, Rational(1,2)),
                                (sp.pi/6, sp.pi/3, Rational(1,20))]:
    K_val = K_factored.subs([(th1, t1_val), (th2, t2_val), (gam, g_val)])
    print(f"  K(th1={float(t1_val):.3f}, th2={float(t2_val):.3f}, gamma={float(g_val):.2f}) = {float(sp.simplify(K_val)):.8f}")
