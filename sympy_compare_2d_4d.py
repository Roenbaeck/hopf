"""
Compare R_0202 of the 4D metric with the Gaussian curvature of the 2D
orbit metric, both computed symbolically in SymPy.
"""
import sympy as sp

t1, t2 = sp.symbols('theta_1 theta_2')
L1 = sp.Function('L1')(t1, t2)
L2 = sp.Function('L2')(t1, t2)

# ── 2D Gaussian curvature via the standard formula for ds^2 = E du^2 + G dv^2 ──
# K = -1/(2*sqrt(EG)) * [d/du(G_u / sqrt(EG)) + d/dv(E_v / sqrt(EG))]
E, G = L1, L2
sqEG = sp.sqrt(E*G)

term1 = sp.diff(sp.diff(G, t1) / sqEG, t1)
term2 = sp.diff(sp.diff(E, t2) / sqEG, t2)
K_2D = sp.simplify(-1/(2*sqEG) * (term1 + term2))

print("Gaussian curvature of 2D metric [[L1, 0], [0, L2]]:")
print(K_2D)

# ── R_0202 from the SymPy diagonal computation ──
# From the previous script output:
L1_1 = sp.diff(L1, t1)
L1_2 = sp.diff(L1, t2)
L2_1 = sp.diff(L2, t1)
L2_2 = sp.diff(L2, t2)
L1_22 = sp.diff(L1, t2, 2)
L2_11 = sp.diff(L2, t1, 2)

R_0202 = ((L1_1*L2_1 + L1_2**2)*L2 + (L1_2*L2_2 + L2_1**2)*L1
          - 2*(L1_22 + L2_11)*L1*L2) / (4*L1*L2)
K_4D = R_0202 / (L1*L2)

diff = sp.simplify(K_2D - K_4D)
print(f"\nK_2D - K_4D = {diff}")

if diff == 0:
    print("CONFIRMED: K_02 of the 4D metric = Gaussian curvature of 2D orbit metric")
else:
    print("MISMATCH!")
    print(f"\nK_2D = {sp.simplify(K_2D)}")
    print(f"\nK_4D = {sp.simplify(K_4D)}")
