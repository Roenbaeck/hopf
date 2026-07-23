"""
Test K_mix at critical points using paper's analytical Hessian infrastructure.
Uses only 1 layer of FD (for Christoffel derivatives) instead of 3.
"""

import numpy as np
from test_Q_S2xS2 import make_seam_product_harmonics
from test_local_fine import compute_K_at_point

gamma = 0.3

# Isotropic seam: s = x1x2 + y1y2 + z1z2 = x.y
iso = make_seam_product_harmonics([1,0,0,0,1,0,0,0,1])

# Non-isotropic: s = 3*x1x2 + 2*y1y2 + 1*z1z2
aniso = make_seam_product_harmonics([3,0,0,0,2,0,0,0,1])

print("="*70)
print(f"K_mix at critical points, gamma={gamma}")
print("Using analytical Hessian + single FD layer (paper's method)")
print("="*70)

# Critical point x=y=e1: (pi/2, 0, pi/2, 0)
# But phi=0 is fine; theta=pi/2 avoids pole singularity
crit_e1 = (np.pi/2, 0.0, np.pi/2, 0.0)
# Critical point x=y=e2: (pi/2, pi/2, pi/2, pi/2)
crit_e2 = (np.pi/2, np.pi/2, np.pi/2, np.pi/2)
# Generic non-critical point
generic = (1.0, 0.5, 1.2, 0.8)
# Point on isotropic torus (x != y)
torus_pt = (np.pi/2, 0.5, np.pi/2, 1.2)  # on N_P, s = cos(0.5-1.2)

test_points = [
    ("crit x=y=e1", crit_e1),
    ("crit x=y=e2", crit_e2),
    ("generic", generic),
    ("torus x!=y", torus_pt),
]

# h-convergence for each case
for seam_name, seam_func, coeffs in [
    ("isotropic x.y", iso, [1,0,0,0,1,0,0,0,1]),
    ("anisotropic 3x1x2+2y1y2+z1z2", aniso, [3,0,0,0,2,0,0,0,1]),
]:
    print(f"\n{'='*70}")
    print(f"Seam: {seam_name}")
    print(f"{'='*70}")
    for pt_name, pt in test_points:
        print(f"\n  --- {pt_name}: ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}, {pt[3]:.4f}) ---")

        # Evaluate s and grad s at this point
        s_val, ds_val, _ = seam_func(
            np.array([pt[0]]), np.array([pt[1]]),
            np.array([pt[2]]), np.array([pt[3]])
        )
        grad_norm = np.sqrt(sum(d[0]**2 for d in ds_val))
        print(f"  s = {s_val[0]:.6f}, |grad s| = {grad_norm:.2e}")

        # Convergence test
        for h in [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]:
            K_min, K_max, K_coord = compute_K_at_point(seam_func, gamma, pt, h=h, scan_angles=90)
            print(f"  h={h:.0e}: Kmin={K_min:+.8e}  Kmax={K_max:+.8e}")
