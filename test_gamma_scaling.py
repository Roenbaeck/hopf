#!/usr/bin/env python3
"""
Check gamma-scaling of int K dV on S^2 x S^2 for l=2 seams.

If int Q dV = 0 still holds on S^2xS^2, then:
  int K dV = gamma^2 * int Q dV + O(gamma^3) = O(gamma^3)
  => int K dV / gamma^3 should be bounded

If int Q dV != 0 on S^2xS^2, then:
  int K dV / gamma^2 should be bounded
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from test_Q_S2xS2_l2 import make_seam_general, build_curvature


def main():
    f_xz = lambda x,y,z: x*z
    f_yz = lambda x,y,z: y*z
    f_x2y2 = lambda x,y,z: x**2 - y**2
    f_xy = lambda x,y,z: x*y
    f_z2 = lambda x,y,z: 3*z**2 - 1
    f_x = lambda x,y,z: x
    f_y = lambda x,y,z: y

    trim = 2
    N = 10

    seams = {
        'xz*yz': ([(f_xz, f_yz)], [1.0]),
        'z2*z2': ([(f_z2, f_z2)], [1.0]),
        'x2y2*xy': ([(f_x2y2, f_xy)], [1.0]),
        'mixed_l1l2': ([(f_x, f_xz), (f_y, f_yz), (f_z2, f_xy)], [1.0, -0.5, 0.7]),
    }

    gammas = [0.1, 0.03, 0.01, 0.003, 0.001]

    print("=" * 70)
    print("Gamma-scaling of int K dV on S^2 x S^2")
    print("=" * 70)
    print(f"{'Seam':<15} {'gamma':>8} {'int K':>12} {'int K/g^2':>12} {'int K/g^3':>12} {'Kmin':>12} {'Kmax':>12}")
    print("-" * 83)

    for name, (pairs, coeffs) in seams.items():
        sf = make_seam_general(pairs, coeffs)
        for g in gammas:
            K, dV, shape = build_curvature(sf, g, N)
            sl = tuple(slice(trim, -trim) for _ in range(4))
            Ki, dVi = K[sl], dV[sl]
            int_K = np.sum(Ki * dVi)
            kmin, kmax = np.min(Ki), np.max(Ki)
            print(f"{name:<15} {g:>8.4f} {int_K:>12.4e} {int_K/g**2:>12.4e} {int_K/g**3:>12.4e} {kmin:>12.4e} {kmax:>12.4e}")
        print()


if __name__ == '__main__':
    main()
