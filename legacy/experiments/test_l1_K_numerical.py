#!/usr/bin/env python3
"""
Compute all mixed K symbolically but with minimal simplification.
Use numerical substitution to check signs.
"""

import sympy as sp
from sympy import symbols, cos, sin, diff, Rational, N as Neval

th1, th2, gam = symbols('theta_1 theta_2 gamma', real=True, positive=True)
s1, c1 = sin(th1), cos(th1)
s2, c2 = sin(th2), cos(th2)

u = 1 - gam * c1 * c2

n = 4
g = sp.zeros(n, n)
g[0, 0] = u
g[1, 1] = u * s1**2
g[2, 2] = u
g[3, 3] = u * s2**2
g[0, 2] = gam * s1 * s2
g[2, 0] = gam * s1 * s2

coords = [th1, symbols('phi_1'), th2, symbols('phi_2')]

print("Computing inverse metric...")
ginv = g.inv()

# Christoffel symbols (no simplification, just expand)
print("Computing Christoffel symbols...")
Chr = {}
for m in range(n):
    for i in range(n):
        for j in range(i, n):
            val = Rational(0)
            for k in range(n):
                val += ginv[m, k] * (diff(g[k, i], coords[j]) + diff(g[k, j], coords[i]) - diff(g[i, j], coords[k]))
            val = sp.cancel(val / 2)
            if val != 0:
                Chr[(m, i, j)] = val
                if i != j:
                    Chr[(m, j, i)] = val

def get_chr(m, i, j):
    return Chr.get((m, i, j), Rational(0))

def compute_R_numerical(i, j, k, l, subs_dict):
    """Compute R_{ijkl} at a specific point numerically."""
    val = 0
    for m in range(n):
        term1 = diff(get_chr(m, j, l), coords[k])
        term2 = -diff(get_chr(m, j, k), coords[l])
        term3 = Rational(0)
        for p in range(n):
            term3 += get_chr(m, k, p) * get_chr(p, j, l) - get_chr(m, l, p) * get_chr(p, j, k)
        Rm = term1 + term2 + term3
        val += g[i, m] * Rm
    return float(val.subs(subs_dict))

print("Evaluating curvatures numerically...\n")

# Test points
test_points = [
    {th1: sp.pi/4, th2: sp.pi/4, gam: Rational(1,10)},
    {th1: sp.pi/3, th2: sp.pi/6, gam: Rational(1,10)},
    {th1: sp.pi/2, th2: sp.pi/2, gam: Rational(1,10)},
    {th1: sp.pi/6, th2: sp.pi/3, gam: Rational(1,10)},
    {th1: sp.pi/4, th2: sp.pi/4, gam: Rational(3,10)},
    {th1: sp.pi/4, th2: sp.pi/4, gam: Rational(1,20)},
]

for pt in test_points:
    g_val = float(pt[gam])
    t1 = float(pt[th1])
    t2 = float(pt[th2])
    print(f"Point: th1={t1:.4f}, th2={t2:.4f}, gamma={g_val:.2f}")
    
    for (a, al, name) in [(0,2,"th1,th2"), (0,3,"th1,ph2"), (1,2,"ph1,th2"), (1,3,"ph1,ph2")]:
        R = compute_R_numerical(a, al, a, al, pt)
        g_aa = float(g[a,a].subs(pt))
        g_alal = float(g[al,al].subs(pt))
        g_aal = float(g[a,al].subs(pt))
        denom = g_aa * g_alal - g_aal**2
        K = R / denom if abs(denom) > 1e-15 else float('nan')
        print(f"  K({name}) = {K:+.8f}")
    print()

# Also compute: minimum mixed K over all plane orientations
import numpy as np

print("=" * 60)
print("Min/max mixed K over all planes (numerical sympy)")
print("=" * 60)

for pt in test_points[:3]:
    g_val = float(pt[gam])
    t1 = float(pt[th1])
    t2 = float(pt[th2])
    
    # Get all R_{a alpha b beta} in the mixed sector
    R_mixed = np.zeros((2,2,2,2))
    for ii, i in enumerate([0,1]):
        for jj, j in enumerate([2,3]):
            for kk, k in enumerate([0,1]):
                for ll, l in enumerate([2,3]):
                    R_mixed[ii,jj,kk,ll] = compute_R_numerical(i, j, k, l, pt)
    
    # Get metric components 
    gm = np.zeros((4,4))
    for a in range(4):
        for b in range(4):
            gm[a,b] = float(g[a,b].subs(pt))
    
    # Scan over angles
    Na, Nb = 180, 180
    angles_a = np.linspace(0, np.pi, Na, endpoint=False)
    angles_b = np.linspace(0, np.pi, Nb, endpoint=False)
    K_min = np.inf
    K_max = -np.inf
    
    for ia in range(Na):
        ca, sa = np.cos(angles_a[ia]), np.sin(angles_a[ia])
        X = np.array([ca, sa])
        for ib in range(Nb):
            cb, sb = np.cos(angles_b[ib]), np.sin(angles_b[ib])
            Y = np.array([cb, sb])
            
            num = 0
            for ii in range(2):
                for jj in range(2):
                    for kk in range(2):
                        for ll in range(2):
                            num += R_mixed[ii,jj,kk,ll] * X[ii]*Y[jj]*X[kk]*Y[ll]
            
            gXX = sum(gm[i,k]*X[ii]*X[kk] for ii,i in enumerate([0,1]) for kk,k in enumerate([0,1]))
            gYY = sum(gm[j,l]*Y[jj]*Y[ll] for jj,j in enumerate([2,3]) for ll,l in enumerate([2,3]))
            gXY = sum(gm[i,j]*X[ii]*Y[jj] for ii,i in enumerate([0,1]) for jj,j in enumerate([2,3]))
            
            denom = gXX*gYY - gXY**2
            K_val = num / denom
            K_min = min(K_min, K_val)
            K_max = max(K_max, K_val)
    
    print(f"  th1={t1:.4f}, th2={t2:.4f}, gam={g_val:.2f}: K_min={K_min:+.8f}, K_max={K_max:+.8f}")
