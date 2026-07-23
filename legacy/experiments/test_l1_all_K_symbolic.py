#!/usr/bin/env python3
"""
Compute ALL mixed sectional curvatures symbolically for s = z1*z2.
K(e_0,e_2) = K(dth1, dth2) = 0  [already proven]
K(e_0,e_3) = K(dth1, dph2) = ?
K(e_1,e_2) = K(dph1, dth2) = ?
K(e_1,e_3) = K(dph1, dph2) = ?

Also compute the curvature operator (mixed part) to find min over all mixed planes.
"""

import sympy as sp
from sympy import symbols, cos, sin, trigsimp, diff, factor, expand, Rational, sqrt, simplify

th1, th2, gam = symbols('theta_1 theta_2 gamma', real=True, positive=True)

s1, c1 = sin(th1), cos(th1)
s2, c2 = sin(th2), cos(th2)

s = c1 * c2
u = 1 - gam * s

n = 4
g = sp.zeros(n, n)
g[0, 0] = u
g[1, 1] = u * s1**2
g[2, 2] = u
g[3, 3] = u * s2**2
g[0, 2] = gam * s1 * s2
g[2, 0] = gam * s1 * s2

coords = [th1, symbols('phi_1'), th2, symbols('phi_2')]

ginv = g.inv()
ginv_s = sp.zeros(n, n)
for i in range(n):
    for j in range(n):
        ginv_s[i, j] = trigsimp(ginv[i, j])

# Christoffel symbols
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

def compute_R(i, j, k, l):
    """R_{ijkl} = g_{im} R^m_{jkl}"""
    val = Rational(0)
    for m in range(n):
        term1 = diff(get_chr(m, j, l), coords[k])
        term2 = -diff(get_chr(m, j, k), coords[l])
        term3 = Rational(0)
        for p in range(n):
            term3 += get_chr(m, k, p) * get_chr(p, j, l) - get_chr(m, l, p) * get_chr(p, j, k)
        val += g[i, m] * (term1 + term2 + term3)
    return trigsimp(val)

# Mixed curvatures: K(e_a, e_alpha) = R_{a,alpha,a,alpha} / (g_{aa}*g_{alpha,alpha} - g_{a,alpha}^2)
print("Computing mixed curvatures for s = cos(th1)*cos(th2)...\n")

for (a, al, name) in [(0,2,"th1,th2"), (0,3,"th1,ph2"), (1,2,"ph1,th2"), (1,3,"ph1,ph2")]:
    print(f"--- K({name}) = R_{a}{al}{a}{al} / denom ---")
    R = compute_R(a, al, a, al)
    denom = g[a,a]*g[al,al] - g[a,al]**2
    denom_s = trigsimp(expand(denom))
    
    K = trigsimp(R / denom)
    K_f = factor(K)
    print(f"  R = {R}")
    print(f"  denom = {denom_s}")
    print(f"  K = {K}")
    print(f"  K(factored) = {K_f}")
    
    # Taylor in gamma
    K_ser = sp.series(K, gam, 0, n=4)
    print(f"  K = {K_ser}")
    
    # Leading order gamma^2 coefficient
    K_poly = sp.Poly(sp.series(R, gam, 0, n=4).removeO(), gam)
    print(f"  R Taylor coeffs: {K_poly.all_coeffs()}")
    
    # Numerical checks
    for t1_val, t2_val in [(sp.pi/4, sp.pi/4), (sp.pi/3, sp.pi/6)]:
        g_val = Rational(1,10)
        K_num = K_f.subs([(th1, t1_val), (th2, t2_val), (gam, g_val)])
        print(f"  K(th1={float(t1_val):.3f}, th2={float(t2_val):.3f}, g=0.1) = {float(sp.simplify(K_num)):.8f}")
    print()
