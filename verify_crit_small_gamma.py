"""
Test K_mix at critical points for anisotropic seam with SMALL gamma.
For sigma = (3,2,1), need gamma < 1/(2*sigma_max) ~ 0.167 to ensure PD metric.
Use gamma = 0.05 for safety.
"""

import numpy as np
from test_Q_S2xS2 import make_seam_product_harmonics
from test_local_fine import compute_K_at_point

gamma = 0.05

aniso = make_seam_product_harmonics([3,0,0,0,2,0,0,0,1])
iso = make_seam_product_harmonics([1,0,0,0,1,0,0,0,1])

print("="*70)
print(f"Anisotropic seam 3x1x2+2y1y2+z1z2, gamma={gamma}")
print("="*70)

crit_e1 = (np.pi/2, 0.0, np.pi/2, 0.0)       # s=3
crit_e2 = (np.pi/2, np.pi/2, np.pi/2, np.pi/2)  # s=2
generic = (1.0, 0.5, 1.2, 0.8)
generic2 = (0.8, 1.5, 0.6, 2.0)

pts = [
    ("crit x=y=e1 (s=3)", crit_e1),
    ("crit x=y=e2 (s=2)", crit_e2),
    ("generic1", generic),
    ("generic2", generic2),
]

for name, pt in pts:
    s_v, ds_v, _ = aniso(
        np.array([pt[0]]), np.array([pt[1]]),
        np.array([pt[2]]), np.array([pt[3]])
    )
    grad = np.sqrt(sum(d[0]**2 for d in ds_v))
    cf = 1 - gamma*s_v[0]
    print(f"\n--- {name} ---")
    print(f"  s={s_v[0]:.4f}, |grad|={grad:.2e}, conf_factor={cf:.4f}")
    for h in [1e-3, 5e-4, 1e-4, 5e-5]:
        Kn, Kx, Kc = compute_K_at_point(aniso, gamma, pt, h=h, scan_angles=90)
        print(f"  h={h:.0e}: Kmin={Kn:+.10e}  Kmax={Kx:+.10e}")

# Compare: isotropic at critical and generic
print(f"\n{'='*70}")
print(f"Isotropic seam x.y, gamma={gamma}")
print(f"{'='*70}")
iso_pts = [
    ("crit x=y=e1 (s=1)", crit_e1),
    ("generic1", generic),
]
for name, pt in iso_pts:
    s_v, ds_v, _ = iso(
        np.array([pt[0]]), np.array([pt[1]]),
        np.array([pt[2]]), np.array([pt[3]])
    )
    grad = np.sqrt(sum(d[0]**2 for d in ds_v))
    cf = 1 - gamma*s_v[0]
    print(f"\n--- {name} ---")
    print(f"  s={s_v[0]:.4f}, |grad|={grad:.2e}, conf_factor={cf:.4f}")
    for h in [1e-3, 5e-4, 1e-4, 5e-5]:
        Kn, Kx, Kc = compute_K_at_point(iso, gamma, pt, h=h, scan_angles=90)
        print(f"  h={h:.0e}: Kmin={Kn:+.10e}  Kmax={Kx:+.10e}")
