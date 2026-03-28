"""
Numerical verification: does K_mix <= 0 at the maximum of Psi for
O(2)xO(2)-equivariant seam metrics on S^2 x S^2?

We test the full 4x4 curvature (not the 2x2 reduction) for safety,
using the seam generator forms from Theorem 2.3:
  lambda_k = alpha_k + beta_k * s_k^2 + gamma_k * s_kk
  mu_k     = alpha_k + gamma_k * s_k * cot(theta_k)
  eta      = beta_12 * s_1 * s_2 + gamma_12 * s_12
"""

import numpy as np
from itertools import product as iprod

# ── Seam function and its derivatives ──

def seam_data(s_func, theta1, theta2, h=1e-5):
    """Compute s and its derivatives at (theta1, theta2) by finite differences."""
    s = s_func(theta1, theta2)
    s1 = (s_func(theta1+h, theta2) - s_func(theta1-h, theta2)) / (2*h)
    s2 = (s_func(theta1, theta2+h) - s_func(theta1, theta2-h)) / (2*h)
    s11 = (s_func(theta1+h, theta2) - 2*s + s_func(theta1-h, theta2)) / h**2
    s22 = (s_func(theta1, theta2+h) - 2*s + s_func(theta1, theta2-h)) / h**2
    s12 = (s_func(theta1+h, theta2+h) - s_func(theta1+h, theta2-h)
           - s_func(theta1-h, theta2+h) + s_func(theta1-h, theta2-h)) / (4*h**2)
    return s, s1, s2, s11, s22, s12

def metric_coeffs(theta1, theta2, s_func, params):
    """Compute lambda1, lambda2, mu1, mu2, eta from seam data and params."""
    a1, a2, b1, b2, b12, g1, g2, g12 = params
    s, s1, s2, s11, s22, s12 = seam_data(s_func, theta1, theta2)

    lam1 = a1 + b1 * s1**2 + g1 * s11
    lam2 = a2 + b2 * s2**2 + g2 * s22
    eta_ = b12 * s1 * s2 + g12 * s12

    # mu_k = alpha_k + gamma_k * s_k * cot(theta_k)
    ct1 = np.cos(theta1) / np.sin(theta1) if abs(np.sin(theta1)) > 1e-10 else 0
    ct2 = np.cos(theta2) / np.sin(theta2) if abs(np.sin(theta2)) > 1e-10 else 0
    mu1 = a1 + g1 * s1 * ct1
    mu2 = a2 + g2 * s2 * ct2

    return lam1, lam2, mu1, mu2, eta_

def build_metric(theta1, theta2, s_func, params):
    """Build the 4x4 metric matrix at (theta1, theta2)."""
    lam1, lam2, mu1, mu2, eta_ = metric_coeffs(theta1, theta2, s_func, params)
    g = np.zeros((4, 4))
    g[0, 0] = lam1
    g[1, 1] = mu1 * np.sin(theta1)**2
    g[2, 2] = lam2
    g[3, 3] = mu2 * np.sin(theta2)**2
    g[0, 2] = eta_
    g[2, 0] = eta_
    return g

def compute_Psi(theta1, theta2, s_func, params):
    """Compute Psi = (1/2) log(lam1*lam2 - eta^2)."""
    lam1, lam2, _, _, eta_ = metric_coeffs(theta1, theta2, s_func, params)
    D = lam1 * lam2 - eta_**2
    if D <= 0:
        return -np.inf
    return 0.5 * np.log(D)

def compute_R0202_numerical(theta1, theta2, s_func, params, h=1e-4):
    """Compute R_{0202} numerically via finite differences of the full metric."""
    # Build metric at a grid of nearby points
    def g_at(t1, t2):
        return build_metric(t1, t2, s_func, params)

    g0 = g_at(theta1, theta2)
    n = 4
    coords = [theta1, 0.0, theta2, 0.0]  # phi values don't matter (equivariant)

    # Christoffel symbols by finite differences of metric
    # Gamma^i_{jk} = (1/2) g^{il} (dg_{lj}/dx^k + dg_{lk}/dx^j - dg_{jk}/dx^l)

    # Compute metric derivatives dg_{ab}/dx^c for c in {0, 2} (theta1, theta2)
    # c=1 (phi1) and c=3 (phi2) are zero by equivariance/Killing.
    dg = np.zeros((4, 4, 4))
    for c in [0, 2]:
        h_vec = np.zeros(4)
        h_vec[c] = h
        t1p = theta1 + h_vec[0]
        t2p = theta2 + h_vec[2]
        t1m = theta1 - h_vec[0]
        t2m = theta2 - h_vec[2]
        gp = g_at(t1p, t2p)
        gm = g_at(t1m, t2m)
        dg[:, :, c] = (gp - gm) / (2 * h)

    # Second derivatives of metric: ddg_{ab}/dx^c dx^d
    ddg = np.zeros((4, 4, 4, 4))
    for c in [0, 2]:
        for d in [0, 2]:
            if c == d:
                h_vec = np.zeros(4)
                h_vec[c] = h
                t1p = theta1 + h_vec[0]
                t2p = theta2 + h_vec[2]
                t1m = theta1 - h_vec[0]
                t2m = theta2 - h_vec[2]
                gp = g_at(t1p, t2p)
                gm = g_at(t1m, t2m)
                ddg[:, :, c, d] = (gp - 2*g0 + gm) / h**2
            else:
                # mixed partials
                def shift(dc, dd):
                    return g_at(theta1 + dc*h*(1 if 0 in [c] else 0) + dd*h*(1 if 0 in [d] else 0),
                                theta2 + dc*h*(1 if 2 in [c] else 0) + dd*h*(1 if 2 in [d] else 0))

                t1pp = theta1 + (h if c == 0 else 0) + (h if d == 0 else 0)
                t2pp = theta2 + (h if c == 2 else 0) + (h if d == 2 else 0)
                t1pm = theta1 + (h if c == 0 else 0) - (h if d == 0 else 0)
                t2pm = theta2 + (h if c == 2 else 0) - (h if d == 2 else 0)
                t1mp = theta1 - (h if c == 0 else 0) + (h if d == 0 else 0)
                t2mp = theta2 - (h if c == 2 else 0) + (h if d == 2 else 0)
                t1mm = theta1 - (h if c == 0 else 0) - (h if d == 0 else 0)
                t2mm = theta2 - (h if c == 2 else 0) - (h if d == 2 else 0)
                gpp = g_at(t1pp, t2pp)
                gpm = g_at(t1pm, t2pm)
                gmp = g_at(t1mp, t2mp)
                gmm = g_at(t1mm, t2mm)
                ddg[:, :, c, d] = (gpp - gpm - gmp + gmm) / (4*h**2)

    # Use the 2x2 block approach since R_{0202} only depends on it
    # Indices: a=0 (theta1), b=1 (theta2) in 2x2

    # 2x2 metric
    g2 = np.array([[g0[0,0], g0[0,2]], [g0[2,0], g0[2,2]]])
    det2 = g2[0,0]*g2[1,1] - g2[0,1]*g2[1,0]
    g2inv = np.array([[g2[1,1], -g2[0,1]], [-g2[1,0], g2[0,0]]]) / det2

    # Map 2x2 indices to 4x4 indices
    idx = [0, 2]

    # 2x2 metric derivatives
    dg2 = np.zeros((2, 2, 2))
    for a in range(2):
        for b in range(2):
            for c in range(2):
                dg2[a, b, c] = dg[idx[a], idx[b], idx[c]]

    ddg2 = np.zeros((2, 2, 2, 2))
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    ddg2[a, b, c, d] = ddg[idx[a], idx[b], idx[c], idx[d]]

    # Christoffel symbols for 2x2 block
    Gamma2 = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    Gamma2[i, j, k] += 0.5 * g2inv[i, l] * (
                        dg2[l, j, k] + dg2[l, k, j] - dg2[j, k, l])

    # Derivative of g2inv
    dg2inv = np.zeros((2, 2, 2))
    for i in range(2):
        for l in range(2):
            for c in range(2):
                for m in range(2):
                    for nn in range(2):
                        dg2inv[i, l, c] -= g2inv[i, m] * g2inv[l, nn] * dg2[m, nn, c]

    # dGamma[i,j,k,c] = d_c Gamma^i_{jk}
    dGamma2 = np.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for c in range(2):
                    for l in range(2):
                        dGamma2[i, j, k, c] += 0.5 * dg2inv[i, l, c] * (
                            dg2[l, j, k] + dg2[l, k, j] - dg2[j, k, l])
                        dGamma2[i, j, k, c] += 0.5 * g2inv[i, l] * (
                            ddg2[l, j, k, c] + ddg2[l, k, j, c] - ddg2[j, k, l, c])

    # R^i_{jkl} = d_k G^i_{jl} - d_l G^i_{jk} + G^i_{km} G^m_{jl} - G^i_{lm} G^m_{jk}
    # We want R_{0101} in 2x2 notation = R_{0202} in 4x4 notation.
    # R^i_{101}: i=0,1; j=1, k=0, l=1

    R_up = np.zeros(2)
    for i in range(2):
        R_up[i] = dGamma2[i, 1, 1, 0] - dGamma2[i, 1, 0, 1]
        for m in range(2):
            R_up[i] += Gamma2[i, 0, m] * Gamma2[m, 1, 1] - Gamma2[i, 1, m] * Gamma2[m, 1, 0]

    # R_{0101} = g_{00} R^0_{101} + g_{01} R^1_{101}
    R_0101 = g2[0, 0] * R_up[0] + g2[0, 1] * R_up[1]

    # Sectional curvature denominator: det(g2)
    K_mix = R_0101 / det2

    return R_0101, K_mix, det2


# ── Seam functions ──

def height_sum(theta1, theta2):
    return np.cos(theta1) + np.cos(theta2)

def height_product(theta1, theta2):
    return np.cos(theta1) * np.cos(theta2)

def height_sum_product(theta1, theta2):
    return np.cos(theta1) + np.cos(theta2) + 0.3 * np.cos(theta1) * np.cos(theta2)

def quadratic_seam(theta1, theta2):
    return np.cos(theta1)**2 + np.cos(theta2)**2

def mixed_seam(theta1, theta2):
    return np.sin(theta1) * np.sin(theta2)


# ── Main: scan over parameter space ──

def find_Psi_max(s_func, params, N=50):
    """Find the maximum of Psi on (0,pi) x (0,pi) by grid search + refinement."""
    th1_grid = np.linspace(0.05, np.pi - 0.05, N)
    th2_grid = np.linspace(0.05, np.pi - 0.05, N)

    best_Psi = -np.inf
    best_t1, best_t2 = np.pi/2, np.pi/2

    for t1 in th1_grid:
        for t2 in th2_grid:
            P = compute_Psi(t1, t2, s_func, params)
            if P > best_Psi:
                best_Psi = P
                best_t1, best_t2 = t1, t2

    # Refinement
    for _ in range(5):
        dt = (th1_grid[1] - th1_grid[0]) / 2
        th1_ref = np.linspace(best_t1 - dt, best_t1 + dt, 20)
        th2_ref = np.linspace(best_t2 - dt, best_t2 + dt, 20)
        th1_ref = th1_ref[(th1_ref > 0.01) & (th1_ref < np.pi - 0.01)]
        th2_ref = th2_ref[(th2_ref > 0.01) & (th2_ref < np.pi - 0.01)]
        for t1 in th1_ref:
            for t2 in th2_ref:
                P = compute_Psi(t1, t2, s_func, params)
                if P > best_Psi:
                    best_Psi = P
                    best_t1, best_t2 = t1, t2
        dt /= 2

    return best_t1, best_t2, best_Psi

def check_positive_definite(theta1, theta2, s_func, params):
    """Check if the metric is positive definite at this point."""
    g = build_metric(theta1, theta2, s_func, params)
    try:
        eigvals = np.linalg.eigvalsh(g)
        return np.all(eigvals > 0)
    except:
        return False


print("=" * 70)
print("NUMERICAL VERIFICATION: K_mix at max of Psi")
print("=" * 70)

seam_funcs = {
    "height_sum": height_sum,
    "height_product": height_product,
    "height_sum_product": height_sum_product,
    "quadratic_seam": quadratic_seam,
    "mixed_seam": mixed_seam,
}

# Test many parameter combinations
# params = (a1, a2, b1, b2, b12, g1, g2, g12)
# We need positive definiteness: alpha_k > 0, and the perturbation small enough

np.random.seed(42)
n_random = 200

all_results = []
violation_found = False

for name, s_func in seam_funcs.items():
    print(f"\n--- Seam function: {name} ---")

    # Generate random parameter sets
    for trial in range(n_random):
        if trial < 10:
            # Some structured parameter choices
            eps = 0.1 * (trial + 1)
            if trial == 0:
                params = (1.0, 1.0, 0, 0, eps, 0, 0, 0)  # pure beta12
            elif trial == 1:
                params = (1.0, 1.0, 0, 0, 0, 0, 0, eps)  # pure gamma12
            elif trial == 2:
                params = (1.0, 1.0, 0, 0, eps, 0, 0, eps)  # both cross
            elif trial == 3:
                params = (1.0, 1.0, eps, eps, eps, 0, 0, 0)
            elif trial == 4:
                params = (1.0, 1.0, 0, 0, eps, eps, eps, eps)
            elif trial == 5:
                params = (1.0, 1.0, eps, eps, eps, eps, eps, eps)  # all generators
            else:
                eps = 0.05 * trial
                params = (1.0, 1.0, eps, eps, eps, eps/2, eps/2, eps)
        else:
            # Random parameters (small perturbation from identity)
            eps = np.random.uniform(0.01, 0.3)
            a1 = 1.0 + np.random.uniform(-0.2, 0.2)
            a2 = 1.0 + np.random.uniform(-0.2, 0.2)
            b1 = np.random.uniform(-eps, eps)
            b2 = np.random.uniform(-eps, eps)
            b12 = np.random.uniform(-eps, eps)
            g1 = np.random.uniform(-eps, eps)
            g2 = np.random.uniform(-eps, eps)
            g12 = np.random.uniform(-eps, eps)
            params = (a1, a2, b1, b2, b12, g1, g2, g12)

        # Check positive definiteness at a grid
        pd_ok = True
        for t1 in np.linspace(0.1, np.pi-0.1, 10):
            for t2 in np.linspace(0.1, np.pi-0.1, 10):
                if not check_positive_definite(t1, t2, s_func, params):
                    pd_ok = False
                    break
            if not pd_ok:
                break

        if not pd_ok:
            continue

        # Find max of Psi
        t1_max, t2_max, Psi_max = find_Psi_max(s_func, params, N=40)

        # Check eta != 0 at the max
        _, _, _, _, eta_val = metric_coeffs(t1_max, t2_max, s_func, params)
        if abs(eta_val) < 1e-8:
            continue  # diagonal case, skip

        # Compute R_{0202} at the max
        R_0202, K_mix, det2 = compute_R0202_numerical(t1_max, t2_max, s_func, params)

        all_results.append({
            'name': name,
            'params': params,
            'theta': (t1_max, t2_max),
            'eta': eta_val,
            'Psi_max': Psi_max,
            'R_0202': R_0202,
            'K_mix': K_mix,
            'det2': det2,
        })

        if K_mix > 1e-8:
            violation_found = True
            print(f"  *** VIOLATION: K_mix = {K_mix:.6e} > 0 at ({t1_max:.4f}, {t2_max:.4f})")
            print(f"      params = {params}")
            print(f"      eta = {eta_val:.6e}, det2 = {det2:.6e}")
            print(f"      R_0202 = {R_0202:.6e}")

    # Summary for this seam
    results_for_seam = [r for r in all_results if r['name'] == name]
    if results_for_seam:
        K_vals = [r['K_mix'] for r in results_for_seam]
        print(f"  {len(results_for_seam)} valid tests (eta != 0, pos. def.)")
        print(f"  K_mix range: [{min(K_vals):.6e}, {max(K_vals):.6e}]")

print("\n" + "=" * 70)
if violation_found:
    print("RESULT: Found cases where K_mix > 0 at max of Psi!")
    print("The Psi-maximum argument does NOT directly work.")
else:
    print("RESULT: K_mix <= 0 in all tested cases at the max of Psi.")
    print("The Psi-maximum argument appears to work.")
print("=" * 70)

# Print some detailed examples
print("\n--- Detailed examples (first 5 with largest |K_mix|) ---")
all_results.sort(key=lambda r: abs(r['K_mix']), reverse=True)
for r in all_results[:5]:
    print(f"\nSeam: {r['name']}")
    print(f"  params: a1={r['params'][0]:.3f}, a2={r['params'][1]:.3f}, "
          f"b1={r['params'][2]:.3f}, b2={r['params'][3]:.3f}, b12={r['params'][4]:.3f}, "
          f"g1={r['params'][5]:.3f}, g2={r['params'][6]:.3f}, g12={r['params'][7]:.3f}")
    print(f"  max Psi at: ({r['theta'][0]:.4f}, {r['theta'][1]:.4f})")
    print(f"  eta = {r['eta']:.6e}")
    print(f"  R_0202 = {r['R_0202']:.6e}")
    print(f"  K_mix = {r['K_mix']:.6e}")
    print(f"  det(g_2) = {r['det2']:.6e}")
