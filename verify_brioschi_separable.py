"""
Verify: K(d_theta1, d_theta2) = 0 for separable seams of ANY ell,
using the Brioschi formula on the totally geodesic (theta1, theta2) surface.

For axisymmetric s = f(theta1) * g(theta2), the claim is that
d(lambda2)/d(theta1) - d(eta)/d(theta2) = 0 algebraically.
"""
import numpy as np
from test_local_fine import compute_K_at_point
from test_Q_S2xS2 import make_seam_product_harmonics

# Helper to make separable seams from embedding coordinates
def make_separable_seam(f1_func, f2_func, coeff=1.0):
    """
    Create s = coeff * f1(x1) * f2(x2) where f1, f2 are functions
    of embedding coordinates on S^2.
    """
    def seam_func(t1, p1, t2, p2):
        st1, ct1 = np.sin(t1), np.cos(t1)
        sp1, cp1 = np.sin(p1), np.cos(p1)
        st2, ct2 = np.sin(t2), np.cos(t2)
        sp2, cp2 = np.sin(p2), np.cos(p2)

        x1, y1, z1 = st1*cp1, st1*sp1, ct1
        x2, y2, z2 = st2*cp2, st2*sp2, ct2

        f1 = f1_func(x1, y1, z1)
        f2 = f2_func(x2, y2, z2)
        s = coeff * f1 * f2

        # Numerical derivatives
        h = 1e-7
        ds = np.zeros(4) if np.isscalar(t1) else np.zeros((4,) + np.shape(t1))
        d2s = [[None]*4 for _ in range(4)]

        coords = [t1, p1, t2, p2]
        for i in range(4):
            cp = list(coords)
            cm = list(coords)
            cp[i] = coords[i] + h
            cm[i] = coords[i] - h

            st1p, ct1p = np.sin(cp[0]), np.cos(cp[0])
            sp1p, cp1p = np.sin(cp[1]), np.cos(cp[1])
            st2p, ct2p = np.sin(cp[2]), np.cos(cp[2])
            sp2p, cp2p = np.sin(cp[3]), np.cos(cp[3])
            x1p, y1p, z1p = st1p*cp1p, st1p*sp1p, ct1p
            x2p, y2p, z2p = st2p*cp2p, st2p*sp2p, ct2p
            sp = coeff * f1_func(x1p, y1p, z1p) * f2_func(x2p, y2p, z2p)

            st1m, ct1m = np.sin(cm[0]), np.cos(cm[0])
            sp1m, cp1m = np.sin(cm[1]), np.cos(cm[1])
            st2m, ct2m = np.sin(cm[2]), np.cos(cm[2])
            sp2m, cp2m = np.sin(cm[3]), np.cos(cm[3])
            x1m, y1m, z1m = st1m*cp1m, st1m*sp1m, ct1m
            x2m, y2m, z2m = st2m*cp2m, st2m*sp2m, ct2m
            sm = coeff * f1_func(x1m, y1m, z1m) * f2_func(x2m, y2m, z2m)

            ds[i] = (sp - sm) / (2*h)

            for j in range(i, 4):
                cij = list(coords)
                cij_pp = list(coords); cij_pp[i] += h; cij_pp[j] += h
                cij_pm = list(coords); cij_pm[i] += h; cij_pm[j] -= h
                cij_mp = list(coords); cij_mp[i] -= h; cij_mp[j] += h
                cij_mm = list(coords); cij_mm[i] -= h; cij_mm[j] -= h

                def eval_s(c):
                    st1_, ct1_ = np.sin(c[0]), np.cos(c[0])
                    sp1_, cp1_ = np.sin(c[1]), np.cos(c[1])
                    st2_, ct2_ = np.sin(c[2]), np.cos(c[2])
                    sp2_, cp2_ = np.sin(c[3]), np.cos(c[3])
                    return coeff * f1_func(st1_*cp1_, st1_*sp1_, ct1_) * f2_func(st2_*cp2_, st2_*sp2_, ct2_)

                d2s[i][j] = (eval_s(cij_pp) - eval_s(cij_pm) - eval_s(cij_mp) + eval_s(cij_mm)) / (4*h*h)
                d2s[j][i] = d2s[i][j]

        return s, ds, d2s

    return seam_func


# Test points (away from poles and equator)
points = [
    (1.0, 0.5, 1.2, 0.8),
    (0.7, 1.0, 0.9, 1.5),
    (0.5, 2.0, 1.3, 0.3),
    (1.2, 0.8, 0.6, 1.8),
    (0.8, 1.2, 1.1, 0.6),
]

gamma = 0.05

# Test various separable seams
seams = {
    # l=1
    "l=1: z*z (rank-1)": (lambda x,y,z: z, lambda x,y,z: z),
    # l=2 separable (axisymmetric)
    "l=2: (3z^2-1)*(3z^2-1)": (lambda x,y,z: 3*z**2-1, lambda x,y,z: 3*z**2-1),
    "l=2: (3z^2-1)*z": (lambda x,y,z: 3*z**2-1, lambda x,y,z: z),
    # l=3 separable
    "l=3: (5z^3-3z)*(5z^3-3z)": (lambda x,y,z: 5*z**3-3*z, lambda x,y,z: 5*z**3-3*z),
    # Non-separable for comparison
    "NON-SEP l=2: xz*xz": (lambda x,y,z: x*z, lambda x,y,z: x*z),
}

print("=" * 72)
print(f"K(d_theta1, d_theta2) for separable vs non-separable seams, gamma={gamma}")
print("=" * 72)
print()

for name, (f1, f2) in seams.items():
    sf = make_separable_seam(f1, f2)
    print(f"--- {name} ---")
    for pt in points:
        K_min, K_max, K_coord = compute_K_at_point(sf, gamma, pt, h=1e-4)
        K_tt = K_coord["th1,th2"]
        K_tp = K_coord["th1,ph2"]
        K_pt = K_coord["ph1,th2"]
        K_pp = K_coord["ph1,ph2"]
        print(f"  {pt}: K(th1,th2)={K_tt:+.4e}  K(th1,ph2)={K_tp:+.4e}  K(ph1,th2)={K_pt:+.4e}  K(ph1,ph2)={K_pp:+.4e}")
    print()
