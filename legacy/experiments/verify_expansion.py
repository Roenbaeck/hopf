"""
Verify the FEEDBACK's Lemma (riemann-expansion):

  K_g(X,Y) = (gamma^2/4) * (|grad_1 s|^2 - (grad_1 s . X)^2
              + |grad_2 s|^2 - (grad_2 s . Y)^2) + O(gamma^3)

Test: compute K/gamma^2 for several gamma values and check convergence
to the predicted coefficient.

Uses the paper's analytical Hessian infrastructure for reliable FD.
"""

import numpy as np
from test_Q_S2xS2 import make_seam_product_harmonics
from test_local_fine import compute_K_at_point


def predicted_coeff(seam_func, pt, X_idx, Y_idx):
    """
    Compute the predicted O(gamma^2) coefficient from the formula:
    (1/4) * (|grad_1 s|^2 - (grad_1 s . e_{X_idx})^2
             + |grad_2 s|^2 - (grad_2 s . e_{Y_idx})^2)

    X_idx: 0 or 1 (index in TM_1 basis: e_theta1, e_phi1)
    Y_idx: 0 or 1 (index in TM_2 basis: e_theta2, e_phi2)
    """
    t1, p1, t2, p2 = pt
    s_val, ds, _ = seam_func(
        np.array([t1]), np.array([p1]),
        np.array([t2]), np.array([p2])
    )

    # ds[0] = ds/dtheta1, ds[1] = ds/dphi1, ds[2] = ds/dtheta2, ds[3] = ds/dphi2
    # In orthonormal frame on S^2: e_theta has norm 1, e_phi has norm sin(theta)
    # Gradient components in orthonormal frame:
    # (grad_1 s)_theta1 = ds[0], (grad_1 s)_phi1 = ds[1]/sin(theta1)
    st1, st2 = np.sin(t1), np.sin(t2)

    grad1 = np.array([ds[0][0], ds[1][0] / st1])  # orthonormal components
    grad2 = np.array([ds[2][0], ds[3][0] / st2])

    # X is e_{X_idx} in orthonormal frame, Y is e_{Y_idx}
    grad1_dot_X = grad1[X_idx]
    grad2_dot_Y = grad2[Y_idx]

    coeff = 0.25 * (np.dot(grad1, grad1) - grad1_dot_X**2
                    + np.dot(grad2, grad2) - grad2_dot_Y**2)
    return coeff, grad1, grad2


# Test points (avoiding coordinate singularities)
points = [
    (1.0, 0.5, 1.2, 0.8),
    (np.pi/4, np.pi/4, np.pi/4, np.pi/4),
    (0.8, 1.5, 0.6, 2.0),
    (np.pi/3, np.pi/6, np.pi/4, np.pi/3),
]

# Mixed planes: (X_idx, Y_idx) -> coordinate plane
planes = [
    (0, 0, "e_th1 ^ e_th2"),
    (0, 1, "e_th1 ^ e_ph2"),
    (1, 0, "e_ph1 ^ e_th2"),
    (1, 1, "e_ph1 ^ e_ph2"),
]

gammas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

print("="*80)
print("VERIFICATION: K/gamma^2 vs predicted O(gamma^2) coefficient")
print("="*80)

for seam_name, coeffs in [
    ("isotropic x.y", [1,0,0,0,1,0,0,0,1]),
    ("anisotropic 3x1x2+2y1y2+z1z2", [3,0,0,0,2,0,0,0,1]),
    ("single z1z2", [0,0,0,0,0,0,0,0,1]),
]:
    sf = make_seam_product_harmonics(coeffs)
    print(f"\n{'='*80}")
    print(f"Seam: {seam_name}")
    print(f"{'='*80}")

    for pt in points[:2]:  # just 2 points per seam to keep output manageable
        print(f"\n  Point: {tuple(round(x,4) for x in pt)}")

        for xi, yi, pname in planes[:2]:  # just 2 planes
            pred, g1, g2 = predicted_coeff(sf, pt, xi, yi)
            print(f"\n    Plane: {pname}")
            print(f"    Predicted coeff (from formula): {pred:.8e}")
            print(f"    grad_1 s = {g1}, grad_2 s = {g2}")
            print(f"    {'gamma':>10s}  {'K':>14s}  {'K/gamma^2':>14s}  {'predicted':>14s}  {'ratio':>10s}")

            for gamma in gammas:
                # Compute K for this specific coordinate plane
                _, _, K_coord = compute_K_at_point(sf, gamma, pt, h=5e-4, scan_angles=10)
                Kval = list(K_coord.values())[2*xi + yi]  # th1th2, th1ph2, ph1th2, ph1ph2
                Kg2 = Kval / gamma**2 if gamma > 0 else 0
                ratio = Kg2 / pred if abs(pred) > 1e-15 else float('nan')
                print(f"    {gamma:10.4f}  {Kval:+14.8e}  {Kg2:+14.8e}  {pred:+14.8e}  {ratio:10.4f}")

# Also check whether K_min corresponds to gradient-parallel direction
print(f"\n{'='*80}")
print("CHECK: Is K_min at the gradient-parallel plane?")
print("="*80)

sf = make_seam_product_harmonics([1,0,0,0,1,0,0,0,1])
pt = (1.0, 0.5, 1.2, 0.8)
gamma = 0.05

s_val, ds, _ = sf(np.array([pt[0]]), np.array([pt[1]]),
                   np.array([pt[2]]), np.array([pt[3]]))
st1, st2 = np.sin(pt[0]), np.sin(pt[2])
grad1 = np.array([ds[0][0], ds[1][0]/st1])
grad2 = np.array([ds[2][0], ds[3][0]/st2])
print(f"grad_1 s direction: angle = {np.arctan2(grad1[1], grad1[0]):.4f} rad")
print(f"grad_2 s direction: angle = {np.arctan2(grad2[1], grad2[0]):.4f} rad")

K_min, K_max, K_coord = compute_K_at_point(sf, gamma, pt, h=5e-4, scan_angles=360)
print(f"K_min = {K_min:.8e}, K_max = {K_max:.8e}")
for name, val in K_coord.items():
    print(f"  K({name}) = {val:.8e}")

# Scan to find the angle that minimizes K
print(f"\n  Fine angle scan:")
from test_local_fine import compute_K_at_point as ckp
# We need to look at the raw R_mixed tensor to find the minimum direction
