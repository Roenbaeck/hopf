#!/usr/bin/env python3
"""
Exact (no FD) computation of mixed curvatures for s = x1*x2 seam.
Uses SymPy differentiation for all derivatives, then evaluates numerically.

s = sin(th1)*cos(ph1)*sin(th2)*cos(ph2)

This depends on all 4 coordinates, unlike z1*z2.
"""

import sympy as sp
from sympy import symbols, cos, sin, diff, Rational, pi
import numpy as np

th1, ph1, th2, ph2, gam = symbols('theta_1 phi_1 theta_2 phi_2 gamma', real=True, positive=True)

s1, c1 = sin(th1), cos(th1)
sp1, cp1 = sin(ph1), cos(ph1)
s2, c2 = sin(th2), cos(th2)
sp2, cp2 = sin(ph2), cos(ph2)

# Seam function s = x1*x2
seam = s1*cp1 * s2*cp2

# Embedding coords and their covariant Hessian
# For l=1: Hess(s)_{within-factor} = -s * g_factor
# Within factor 1: H[0,0] = -s, H[0,1] = 0, H[1,1] = -s*sin^2(th1)
# Within factor 2: H[2,2] = -s, H[2,3] = 0, H[3,3] = -s*sin^2(th2)
# Cross terms: H[0,2] = dz1_dth1 * dx2_dth2, etc.

# Embedding derivs
x1 = s1*cp1; y1 = s1*sp1; z1 = c1
x2 = s2*cp2; y2 = s2*sp2; z2 = c2

# Gradients of embedding coords
dx1 = [diff(x1, th1), diff(x1, ph1)]  # d/dth1, d/dph1
dy1 = [diff(y1, th1), diff(y1, ph1)]
dz1 = [diff(z1, th1), diff(z1, ph1)]
dx2 = [diff(x2, th2), diff(x2, ph2)]
dy2 = [diff(y2, th2), diff(y2, ph2)]
dz2 = [diff(z2, th2), diff(z2, ph2)]

coords = [th1, ph1, th2, ph2]
n = 4

# For s = x1*x2, the matrix A has a_{00} = 1, rest = 0.
# Mixed Hessian: H_{a,alpha} = sum_ij a_ij (nabla_a f_i)(nabla_alpha g_j)
# = (nabla_a x1)(nabla_alpha x2)  [only i=0, j=0 contributes]

# Full metric
u = 1 - gam * seam
g = sp.zeros(n, n)
g[0,0] = u
g[1,1] = u * s1**2
g[2,2] = u  
g[3,3] = u * s2**2

# Cross terms from Hessian
# H_{0,2} = dx1/dth1 * dx2/dth2 = c1*cp1 * c2*cp2
g[0,2] = gam * dx1[0] * dx2[0]
g[2,0] = g[0,2]
# H_{0,3} = dx1/dth1 * dx2/dph2 = c1*cp1 * (-s2*sp2)
g[0,3] = gam * dx1[0] * dx2[1]
g[3,0] = g[0,3]
# H_{1,2} = dx1/dph1 * dx2/dth2 = (-s1*sp1) * c2*cp2
g[1,2] = gam * dx1[1] * dx2[0]
g[2,1] = g[1,2]
# H_{1,3} = dx1/dph1 * dx2/dph2 = (-s1*sp1) * (-s2*sp2) = s1*sp1*s2*sp2
g[1,3] = gam * dx1[1] * dx2[1]
g[3,1] = g[1,3]

print("Metric for s = x1*x2:")
for i in range(n):
    for j in range(i, n):
        expr = sp.trigsimp(g[i,j])
        if expr != 0:
            print(f"  g[{i},{j}] = {expr}")

# Compute inverse metric
print("\nComputing inverse metric...")
ginv = g.inv()

# Christoffel symbols (symbolic, no FD)
print("Computing Christoffel symbols...")
Chr = {}
count = 0
for m in range(n):
    for i in range(n):
        for j in range(i, n):
            val = Rational(0)
            for k in range(n):
                val += ginv[m, k] * (diff(g[k, i], coords[j]) + diff(g[k, j], coords[i]) - diff(g[i, j], coords[k]))
            val = val / 2
            # Don't simplify - just store
            if True:  # always store, simplify later at evaluation
                Chr[(m, i, j)] = val
                if i != j:
                    Chr[(m, j, i)] = val
            count += 1
            if count % 10 == 0:
                print(f"  {count}/40 Christoffel components...")

def get_chr(m, i, j):
    return Chr.get((m, i, j), Rational(0))

# Compute R_{ijkl} at specific points
def compute_R_at_point(i, j, k, l, pt):
    """Compute R_{ijkl} at a specific point. Uses SymPy diff (exact)."""
    val = 0
    for m in range(n):
        Gam_jl = get_chr(m, j, l)
        Gam_jk = get_chr(m, j, k)
        term1 = diff(Gam_jl, coords[k])
        term2 = -diff(Gam_jk, coords[l])
        term3 = Rational(0)
        for p in range(n):
            term3 += get_chr(m, k, p) * get_chr(p, j, l) - get_chr(m, l, p) * get_chr(p, j, k)
        Rm = term1 + term2 + term3
        val += g[i, m] * Rm
    return float(val.subs(pt))

# Test points  
print("\nEvaluating curvatures...")

test_points = [
    {th1: pi/4, ph1: pi/4, th2: pi/4, ph2: pi/4, gam: Rational(1,10)},
    {th1: pi/3, ph1: pi/6, th2: pi/4, ph2: pi/3, gam: Rational(1,10)},
    {th1: pi/4, ph1: 0, th2: pi/4, ph2: 0, gam: Rational(1,10)},
    {th1: pi/2, ph1: pi/4, th2: pi/2, ph2: pi/4, gam: Rational(1,10)},
]

for idx, pt in enumerate(test_points):
    t1, p1, t2, p2 = float(pt[th1]), float(pt[ph1]), float(pt[th2]), float(pt[ph2])
    gv = float(pt[gam])
    print(f"\nPoint {idx+1}: th1={t1:.4f} ph1={p1:.4f} th2={t2:.4f} ph2={p2:.4f} gam={gv}")
    
    for (a, al, name) in [(0,2,"th1,th2"), (0,3,"th1,ph2"), (1,2,"ph1,th2"), (1,3,"ph1,ph2")]:
        R = compute_R_at_point(a, al, a, al, pt)
        g_aa = float(g[a,a].subs(pt))
        g_alal = float(g[al,al].subs(pt))
        g_aal = float(g[a,al].subs(pt))
        denom = g_aa * g_alal - g_aal**2
        K = R / denom if abs(denom) > 1e-15 else float('nan')
        print(f"  K({name}) = {K:+.10f}")
    
    # Min/max over all mixed planes
    R_mixed = np.zeros((2,2,2,2))
    for ii, i in enumerate([0,1]):
        for jj, j in enumerate([2,3]):
            for kk, k in enumerate([0,1]):
                for ll, l in enumerate([2,3]):
                    R_mixed[ii,jj,kk,ll] = compute_R_at_point(i, j, k, l, pt)
    
    gm = np.zeros((4,4))
    for a in range(4):
        for b in range(4):
            gm[a,b] = float(g[a,b].subs(pt))
    
    Na, Nb = 360, 360
    K_min = np.inf
    K_max = -np.inf
    for ia in range(Na):
        ca, sa = np.cos(np.pi*ia/Na), np.sin(np.pi*ia/Na)
        X = np.array([ca, sa])
        for ib in range(Nb):
            cb, sb = np.cos(np.pi*ib/Nb), np.sin(np.pi*ib/Nb)
            Y = np.array([cb, sb])
            num = sum(R_mixed[ii,jj,kk,ll]*X[ii]*Y[jj]*X[kk]*Y[ll] 
                      for ii in range(2) for jj in range(2) for kk in range(2) for ll in range(2))
            gXX = sum(gm[i,k]*X[ii]*X[kk] for ii,i in enumerate([0,1]) for kk,k in enumerate([0,1]))
            gYY = sum(gm[j,l]*Y[jj]*Y[ll] for jj,j in enumerate([2,3]) for ll,l in enumerate([2,3]))
            gXY = sum(gm[i,j]*X[ii]*Y[jj] for ii,i in enumerate([0,1]) for jj,j in enumerate([2,3]))
            denom = gXX*gYY - gXY**2
            K_val = num/denom
            K_min = min(K_min, K_val)
            K_max = max(K_max, K_val)
    
    print(f"  Min K over all mixed: {K_min:+.10f}")
    print(f"  Max K over all mixed: {K_max:+.10f}")
