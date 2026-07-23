"""Verify torus-restricted seam has K=0 via s_kk = -s identity."""
import sympy as sp

p1, p2, s1, s2, gam = sp.symbols('phi1 phi2 sigma1 sigma2 gamma')

# Restricted seam on torus
s = s1*sp.cos(p1)*sp.cos(p2) + s2*sp.sin(p1)*sp.sin(p2)

# Check: s_11 = -s and s_22 = -s
s11 = sp.diff(s, p1, 2)
s22 = sp.diff(s, p2, 2)
print('s_11 + s =', sp.simplify(s11 + s))
print('s_22 + s =', sp.simplify(s22 + s))

# Orbit metric on torus: E = G = 1 + gamma*s_kk = 1 - gamma*s, F = gamma*s_12
E = 1 + gam*s11
G = 1 + gam*s22
F = gam*sp.diff(s, p1, p2)

print('E =', sp.simplify(E))
print('G =', sp.simplify(G))
print('F =', sp.simplify(F))

# Brioschi numerators
E2_F1 = sp.simplify(sp.diff(E, p2) - sp.diff(F, p1))
G1_F2 = sp.simplify(sp.diff(G, p1) - sp.diff(F, p2))
print('E_2 - F_1 =', E2_F1)
print('G_1 - F_2 =', G1_F2)
