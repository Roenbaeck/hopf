"""
Test the integrated Gauss-Bonnet / integral approach.

Key insight: R_{0202} depends only on the 2x2 orbit metric
  g_2D = [[lambda1, eta], [eta, lambda2]]
and K_mix = R_{0202} / (lambda1*lambda2 - eta^2) = K_{2D}
i.e., the mixed curvature IS the Gaussian curvature of the orbit metric.

The Gauss-Bonnet theorem for the 2D orbit surface constrains
the integral of K_{2D}. This may provide the obstruction.

We also test: the full 4D volume integral of K_mix, weighted by
the orbit volume (sin(theta1) * sin(theta2) * sqrt(mu1*mu2)).
"""

import numpy as np
from numerical_verification_v2 import (
    build_metric, metric_coeffs, seam_data, compute_all_mixed_curvatures
)

def gauss_bonnet_test(s_func, params, N=80):
    """
    Compute:
    1. int K_{2D} * dA_{2D}  over [0,pi]^2  (Gauss-Bonnet integrand)
    2. int K_mix * dV_4D  over S^2 x S^2   (equivariant = int over [0,pi]^2 with weight)
    """
    th1 = np.linspace(0.01, np.pi - 0.01, N)
    th2 = np.linspace(0.01, np.pi - 0.01, N)
    dt1 = th1[1] - th1[0]
    dt2 = th2[1] - th2[0]

    integral_2D = 0.0   # int K_2D * sqrt(det g_2D) dtheta1 dtheta2
    integral_4D = 0.0   # int K_mix * sqrt(det g_4D) dtheta1 dtheta2 (integrated over phi's gives 4pi^2)
    integral_Kdet = 0.0  # int K_2D * det(g_2D) dtheta1 dtheta2

    K_min = float('inf')
    K_max = float('-inf')
    valid = True

    for t1 in th1:
        for t2 in th2:
            lam1, lam2, mu1, mu2, eta_ = metric_coeffs(t1, t2, s_func, params)
            D = lam1 * lam2 - eta_**2
            if D <= 0 or lam1 <= 0 or lam2 <= 0 or mu1 <= 0 or mu2 <= 0:
                valid = False
                continue

            K_dict = compute_all_mixed_curvatures(t1, t2, s_func, params)
            if K_dict is None:
                valid = False
                continue
            K_02 = K_dict.get((0, 2))
            if K_02 is None:
                continue

            K_min = min(K_min, K_02)
            K_max = max(K_max, K_02)

            # 2D area element: sqrt(det g_2D) = sqrt(D)
            sqrt_D = np.sqrt(D)
            integral_2D += K_02 * sqrt_D * dt1 * dt2

            # 4D volume element (after integrating out phi1, phi2 via 4pi^2):
            #   sqrt(det g_4D) = sqrt(D * mu1 * mu2) * sin(theta1) * sin(theta2)
            #   (the full det g_4D = D * mu1*sin^2*mu2*sin^2)
            sqrt_det4 = np.sqrt(D * mu1 * mu2) * np.sin(t1) * np.sin(t2)
            integral_4D += K_02 * sqrt_det4 * dt1 * dt2

            # Also K * det (not sqrt(det))
            integral_Kdet += K_02 * D * dt1 * dt2

    return {
        'valid': valid,
        'K_range': (K_min, K_max),
        'int_K_dA_2D': integral_2D,
        'int_K_dV_4D': integral_4D,  # times 4pi^2 for the full manifold
        'int_K_det': integral_Kdet,
    }


# ── Seam functions ──
seams = {
    "cos+cos": lambda t1, t2: np.cos(t1) + np.cos(t2),
    "cos*cos": lambda t1, t2: np.cos(t1) * np.cos(t2),
    "sin*sin": lambda t1, t2: np.sin(t1) * np.sin(t2),
    "cos+cos+0.3prod": lambda t1, t2: np.cos(t1) + np.cos(t2) + 0.3*np.cos(t1)*np.cos(t2),
    "cos1*sin2": lambda t1, t2: np.cos(t1) * np.sin(t2),
}

# Parameter sets to test
param_sets = [
    # (description, params)
    ("all eps=0.05",       (1, 1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)),
    ("all eps=0.1",        (1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)),
    ("all eps=0.15",       (1, 1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15)),
    ("all eps=0.2",        (1, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)),
    ("pure b12=0.1",       (1, 1, 0, 0, 0.1, 0, 0, 0)),
    ("pure g12=0.1",       (1, 1, 0, 0, 0, 0, 0, 0.1)),
    ("b12+g12=0.1",        (1, 1, 0, 0, 0.1, 0, 0, 0.1)),
    ("b12=0.15,g12=-0.1",  (1, 1, 0, 0, 0.15, 0, 0, -0.1)),
    ("asymm a",            (1.2, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)),
    ("diag+cross",         (1, 1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1)),
    ("large cross",        (1, 1, 0, 0, 0.2, 0, 0, 0.2)),
    ("neg beta, pos gamma",(1, 1, -0.05, -0.05, 0.1, 0.05, 0.05, 0.15)),
]

print("=" * 90)
print(f"{'Seam':<16} {'Params':<22} {'K range':>24} {'∫K dA_2D':>12} {'∫K dV_4D':>12} {'sign?'}")
print("=" * 90)

for sname, s_func in seams.items():
    for pdesc, params in param_sets:
        result = gauss_bonnet_test(s_func, params)
        if not result['valid']:
            status = "INVALID"
        else:
            K_lo, K_hi = result['K_range']
            if K_lo > 1e-10:
                status = "ALL POS!"
            elif K_hi < -1e-10:
                status = "all neg"
            else:
                status = "changes"

        K_lo, K_hi = result['K_range']
        int2D = result['int_K_dA_2D']
        int4D = result['int_K_dV_4D']
        print(f"{sname:<16} {pdesc:<22} [{K_lo:>10.4f},{K_hi:>10.4f}] {int2D:>12.4f} {int4D:>12.4f}  {status}")

# ── Extensive random scan ──
print(f"\n{'='*90}")
print("Random parameter scan (200 trials)")
print(f"{'='*90}")

np.random.seed(42)

int2D_positives = 0
int4D_positives = 0
total_valid = 0

for trial in range(200):
    sname = list(seams.keys())[trial % len(seams)]
    s_func = seams[sname]

    eps = np.random.uniform(0.02, 0.2)
    a1 = 1.0 + np.random.uniform(-0.1, 0.1)
    a2 = 1.0 + np.random.uniform(-0.1, 0.1)
    b1 = np.random.uniform(-eps, eps)
    b2 = np.random.uniform(-eps, eps)
    b12 = np.random.uniform(-eps, eps)
    g1 = np.random.uniform(-eps, eps)
    g2 = np.random.uniform(-eps, eps)
    g12 = np.random.uniform(-eps, eps)
    params = (a1, a2, b1, b2, b12, g1, g2, g12)

    result = gauss_bonnet_test(s_func, params, N=50)
    if not result['valid']:
        continue

    K_lo, K_hi = result['K_range']
    if K_lo >= -1e-10 and K_hi >= -1e-10:
        continue  # skip if no sign info

    total_valid += 1
    if result['int_K_dA_2D'] > 1e-8:
        int2D_positives += 1
    if result['int_K_dV_4D'] > 1e-8:
        int4D_positives += 1

print(f"\nTotal valid trials: {total_valid}")
print(f"∫K dA_2D > 0: {int2D_positives} / {total_valid}")
print(f"∫K dV_4D > 0: {int4D_positives} / {total_valid}")

if int4D_positives == 0:
    print("\n>>> ∫K dV_4D ≤ 0 in ALL cases — integral obstruction works!")
elif int2D_positives == 0:
    print("\n>>> ∫K dA_2D ≤ 0 in ALL cases — 2D Gauss-Bonnet obstruction works!")
else:
    print("\n>>> Neither integral is uniformly non-positive.")
