"""
Test whether L_{a alpha b beta} = 0 for ell=2 seams.

If L=0, then K_mix = O(gamma^2), and K/gamma^2 should converge as gamma -> 0.
If L != 0, then K_mix = O(gamma), and K/gamma should converge.
"""
import numpy as np
from test_Q_S2xS2_l2 import make_seam_general, build_curvature

# Define basis functions locally
f_x = lambda x,y,z: x
f_y = lambda x,y,z: y
f_z = lambda x,y,z: z
f_z2 = lambda x,y,z: 3*z**2 - 1
f_xz = lambda x,y,z: x*z
f_yz = lambda x,y,z: y*z
f_x2y2 = lambda x,y,z: x**2 - y**2
f_xy = lambda x,y,z: x*y

print("=" * 72)
print("L=0 test: K/gamma vs K/gamma^2 scaling for various seams")
print("=" * 72)
print()
print("If K=O(gamma):   K/gamma  should converge as gamma->0")
print("If K=O(gamma^2): K/gamma^2 should converge as gamma->0")
print()

gammas = [0.1, 0.05, 0.01, 0.005, 0.001]

seams = {
    "l=1 isotropic (x.y)": ([(f_x, f_x), (f_y, f_y), (f_z, f_z)], [1, 1, 1]),
    "l=2 pure Y20xY20": ([(f_z2, f_z2)], [1.0]),
    "l=2 pure Y21cxY21s": ([(f_xz, f_yz)], [1.0]),
    "l=2 pure Y22cxY22c": ([(f_x2y2, f_x2y2)], [1.0]),
    "l=1+l=2 mixed": ([(f_x, f_z2), (f_xz, f_xy)], [1.0, 0.5]),
}

for name, (pairs, coeffs) in seams.items():
    print(f"--- {name} ---")
    sf = make_seam_general(pairs, coeffs)
    header = f"  {'gamma':>8s}  {'max|K|':>12s}  {'max|K/g|':>12s}  {'max|K/g^2|':>12s}"
    print(header)
    for g in gammas:
        K, dV, _ = build_curvature(sf, g, N=8)
        sl = tuple(slice(2, -2) for _ in range(4))
        Ki = K[sl]
        maxK = np.max(np.abs(Ki))
        print(f"  {g:>8.4f}  {maxK:>12.4e}  {maxK/g:>12.4e}  {maxK/g**2:>12.4e}")
    print()
