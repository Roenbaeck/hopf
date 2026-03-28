"""
Refined numerical test: restrict to interior of orbit space and
ensure eta is genuinely non-zero. Also test per-plane: the
obstruction only requires that SOME mixed plane has K <= 0,
so check all four mixed planes.
"""

import numpy as np

# ── Seam function and derivatives ──

def seam_data(s_func, theta1, theta2, h=1e-5):
    s = s_func(theta1, theta2)
    s1 = (s_func(theta1+h, theta2) - s_func(theta1-h, theta2)) / (2*h)
    s2 = (s_func(theta1, theta2+h) - s_func(theta1, theta2-h)) / (2*h)
    s11 = (s_func(theta1+h, theta2) - 2*s + s_func(theta1-h, theta2)) / h**2
    s22 = (s_func(theta1, theta2+h) - 2*s + s_func(theta1, theta2-h)) / h**2
    s12 = (s_func(theta1+h, theta2+h) - s_func(theta1+h, theta2-h)
           - s_func(theta1-h, theta2+h) + s_func(theta1-h, theta2-h)) / (4*h**2)
    return s, s1, s2, s11, s22, s12


def metric_coeffs(theta1, theta2, s_func, params):
    a1, a2, b1, b2, b12, g1, g2, g12 = params
    s, s1, s2, s11, s22, s12 = seam_data(s_func, theta1, theta2)
    lam1 = a1 + b1 * s1**2 + g1 * s11
    lam2 = a2 + b2 * s2**2 + g2 * s22
    eta_ = b12 * s1 * s2 + g12 * s12
    ct1 = np.cos(theta1) / np.sin(theta1)
    ct2 = np.cos(theta2) / np.sin(theta2)
    mu1 = a1 + g1 * s1 * ct1
    mu2 = a2 + g2 * s2 * ct2
    return lam1, lam2, mu1, mu2, eta_


def build_metric(theta1, theta2, s_func, params):
    lam1, lam2, mu1, mu2, eta_ = metric_coeffs(theta1, theta2, s_func, params)
    g = np.zeros((4,4))
    g[0,0] = lam1
    g[1,1] = mu1 * np.sin(theta1)**2
    g[2,2] = lam2
    g[3,3] = mu2 * np.sin(theta2)**2
    g[0,2] = eta_
    g[2,0] = eta_
    return g


def compute_all_mixed_curvatures(theta1, theta2, s_func, params, h=5e-4):
    """Compute all four mixed sectional curvatures using full 4x4 Riemann tensor."""
    def g_at(t1, t2):
        return build_metric(t1, t2, s_func, params)

    g0 = g_at(theta1, theta2)
    n = 4

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(g0)
    if np.min(eigvals) <= 0:
        return None

    ginv0 = np.linalg.inv(g0)

    # Metric derivatives wrt theta1 (index 0) and theta2 (index 2) only
    dg = np.zeros((n,n,n))
    for c in [0, 2]:
        eps = np.zeros(4)
        eps[c] = h
        gp = g_at(theta1 + eps[0], theta2 + eps[2])
        gm = g_at(theta1 - eps[0], theta2 - eps[2])
        dg[:,:,c] = (gp - gm) / (2*h)

    # Second derivatives
    ddg = np.zeros((n,n,n,n))
    for c in [0, 2]:
        for d in [0, 2]:
            if c == d:
                eps = np.zeros(4); eps[c] = h
                gp = g_at(theta1+eps[0], theta2+eps[2])
                gm = g_at(theta1-eps[0], theta2-eps[2])
                ddg[:,:,c,d] = (gp - 2*g0 + gm) / h**2
            else:
                t1pp = theta1 + (h if c==0 else 0) + (h if d==0 else 0)
                t2pp = theta2 + (h if c==2 else 0) + (h if d==2 else 0)
                t1pm = theta1 + (h if c==0 else 0) - (h if d==0 else 0)
                t2pm = theta2 + (h if c==2 else 0) - (h if d==2 else 0)
                t1mp = theta1 - (h if c==0 else 0) + (h if d==0 else 0)
                t2mp = theta2 - (h if c==2 else 0) + (h if d==2 else 0)
                t1mm = theta1 - (h if c==0 else 0) - (h if d==0 else 0)
                t2mm = theta2 - (h if c==2 else 0) - (h if d==2 else 0)
                ddg[:,:,c,d] = (g_at(t1pp,t2pp) - g_at(t1pm,t2pm)
                                - g_at(t1mp,t2mp) + g_at(t1mm,t2mm)) / (4*h**2)

    # Christoffel symbols
    Gamma = np.zeros((n,n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    Gamma[i,j,k] += 0.5 * ginv0[i,l] * (dg[l,j,k] + dg[l,k,j] - dg[j,k,l])

    # dg^{il}/dx^c = -g^{im} g^{ln} dg_{mn}/dx^c
    dginv = np.zeros((n,n,n))
    for i in range(n):
        for l in range(n):
            for c in range(n):
                for m in range(n):
                    for nn in range(n):
                        dginv[i,l,c] -= ginv0[i,m] * ginv0[l,nn] * dg[m,nn,c]

    # dGamma[i,j,k,c] = d_c Gamma^i_{jk}
    dGamma = np.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for c in range(n):
                    for l in range(n):
                        dGamma[i,j,k,c] += 0.5 * dginv[i,l,c] * (dg[l,j,k] + dg[l,k,j] - dg[j,k,l])
                        dGamma[i,j,k,c] += 0.5 * ginv0[i,l] * (ddg[l,j,k,c] + ddg[l,k,j,c] - ddg[j,k,l,c])

    # Riemann tensor R^i_{jkl} = d_k G^i_{jl} - d_l G^i_{jk} + G^i_{km}G^m_{jl} - G^i_{lm}G^m_{jk}
    def R_up(i, j, k, l):
        v = dGamma[i,j,l,k] - dGamma[i,j,k,l]
        for m in range(n):
            v += Gamma[i,k,m]*Gamma[m,j,l] - Gamma[i,l,m]*Gamma[m,j,k]
        return v

    # R_{ijkl} = g_{im} R^m_{jkl}
    def R_down(i, j, k, l):
        v = 0
        for m in range(n):
            v += g0[i,m] * R_up(m, j, k, l)
        return v

    # Four mixed planes: (0,2), (0,3), (1,2), (1,3)
    # Sectional curvature K(e_a, e_b) = R_{abab} / (g_{aa}g_{bb} - g_{ab}^2)
    results = {}
    for (a, b) in [(0,2), (0,3), (1,2), (1,3)]:
        R_abab = R_down(a, b, a, b)
        denom = g0[a,a]*g0[b,b] - g0[a,b]**2
        if abs(denom) < 1e-15:
            results[(a,b)] = None
        else:
            results[(a,b)] = R_abab / denom

    return results


def compute_Psi(theta1, theta2, s_func, params):
    lam1, lam2, _, _, eta_ = metric_coeffs(theta1, theta2, s_func, params)
    D = lam1 * lam2 - eta_**2
    if D <= 0:
        return -np.inf
    return 0.5 * np.log(D)


def find_Psi_max(s_func, params, N=60, margin=0.3):
    """Find max of Psi in the interior [margin, pi-margin]^2."""
    th1_grid = np.linspace(margin, np.pi - margin, N)
    th2_grid = np.linspace(margin, np.pi - margin, N)

    best_Psi = -np.inf
    best_t1, best_t2 = np.pi/2, np.pi/2

    for t1 in th1_grid:
        for t2 in th2_grid:
            P = compute_Psi(t1, t2, s_func, params)
            if P > best_Psi:
                best_Psi = P
                best_t1, best_t2 = t1, t2

    # Refine
    for _ in range(8):
        dt = (th1_grid[1] - th1_grid[0]) / 2
        th1_ref = np.linspace(max(margin, best_t1-dt), min(np.pi-margin, best_t1+dt), 30)
        th2_ref = np.linspace(max(margin, best_t2-dt), min(np.pi-margin, best_t2+dt), 30)
        for t1 in th1_ref:
            for t2 in th2_ref:
                P = compute_Psi(t1, t2, s_func, params)
                if P > best_Psi:
                    best_Psi = P
                    best_t1, best_t2 = t1, t2
        dt /= 3

    return best_t1, best_t2, best_Psi


# ── Seam functions ──
seams = {
    "cos+cos":          lambda t1, t2: np.cos(t1) + np.cos(t2),
    "cos*cos":          lambda t1, t2: np.cos(t1) * np.cos(t2),
    "cos+cos+0.3*prod": lambda t1, t2: np.cos(t1) + np.cos(t2) + 0.3*np.cos(t1)*np.cos(t2),
    "sin*sin":          lambda t1, t2: np.sin(t1) * np.sin(t2),
    "cos2+cos2":        lambda t1, t2: np.cos(t1)**2 + np.cos(t2)**2,
    "cos1*sin2":        lambda t1, t2: np.cos(t1) * np.sin(t2),
}

np.random.seed(123)

if __name__ == "__main__":
    _run_main()

def _run_main():
    print("=" * 70)
    print("REFINED VERIFICATION: K_mix planes at max of Psi (interior only)")
    print("=" * 70)

violation_count = 0
test_count = 0

for sname, s_func in seams.items():
    print(f"\n{'='*50}")
    print(f"Seam: {sname}")
    print(f"{'='*50}")

    for trial in range(150):
        # Generate parameters
        if trial < 5:
            eps = 0.05 * (trial + 1)
            params = (1.0, 1.0, eps, eps, eps, eps, eps, eps)
        else:
            eps = np.random.uniform(0.02, 0.25)
            a1 = 1.0 + np.random.uniform(-0.15, 0.15)
            a2 = 1.0 + np.random.uniform(-0.15, 0.15)
            b1 = np.random.uniform(-eps, eps)
            b2 = np.random.uniform(-eps, eps)
            b12 = np.random.uniform(-eps, eps)
            g1 = np.random.uniform(-eps, eps)
            g2 = np.random.uniform(-eps, eps)
            g12 = np.random.uniform(-eps, eps)
            params = (a1, a2, b1, b2, b12, g1, g2, g12)

        # Check positive definiteness on grid in the interior
        pd_ok = True
        for t1 in np.linspace(0.3, np.pi-0.3, 8):
            for t2 in np.linspace(0.3, np.pi-0.3, 8):
                try:
                    g = build_metric(t1, t2, s_func, params)
                    if np.min(np.linalg.eigvalsh(g)) <= 0:
                        pd_ok = False
                        break
                except:
                    pd_ok = False
                    break
            if not pd_ok:
                break
        if not pd_ok:
            continue

        # Find max of Psi in interior
        t1_max, t2_max, Psi_max = find_Psi_max(s_func, params, margin=0.3)

        # Check eta
        _, _, _, _, eta_val = metric_coeffs(t1_max, t2_max, s_func, params)
        if abs(eta_val) < 1e-4:
            continue  # skip near-diagonal

        test_count += 1

        # Compute all mixed curvatures
        K_all = compute_all_mixed_curvatures(t1_max, t2_max, s_func, params)
        if K_all is None:
            continue

        K_02 = K_all.get((0,2))  # theta1-theta2
        K_03 = K_all.get((0,3))  # theta1-phi2
        K_12 = K_all.get((1,2))  # phi1-theta2
        K_13 = K_all.get((1,3))  # phi1-phi2

        all_positive = True
        min_K = float('inf')
        for key, val in K_all.items():
            if val is not None:
                min_K = min(min_K, val)
                if val <= 1e-10:
                    all_positive = False

        if K_02 is not None and K_02 > 1e-8:
            violation_count += 1
            if violation_count <= 10:
                print(f"  K(θ₁,θ₂)={K_02:.4e} > 0 at ({t1_max:.3f},{t2_max:.3f}), η={eta_val:.4e}")
                print(f"    K(θ₁,φ₂)={K_03:.4e}, K(φ₁,θ₂)={K_12:.4e}, K(φ₁,φ₂)={K_13:.4e}")
                print(f"    params: a=({params[0]:.3f},{params[1]:.3f}) b=({params[2]:.3f},{params[3]:.3f},{params[4]:.3f}) g=({params[5]:.3f},{params[6]:.3f},{params[7]:.3f})")
                # Check: is some OTHER mixed plane negative?
                neg_planes = [k for k,v in K_all.items() if v is not None and v < -1e-10]
                if neg_planes:
                    print(f"    -> Other negative planes: {neg_planes}")
                else:
                    print(f"    -> ALL mixed planes positive! (min = {min_K:.4e})")

print(f"\n{'='*70}")
print(f"Summary: {test_count} valid tests, {violation_count} with K(θ₁,θ₂) > 0 at max(Ψ)")
print(f"{'='*70}")

# ── Additional test: check if K_mix can be positive EVERYWHERE ──
print(f"\n{'='*70}")
print("GLOBAL SCAN: does K_mix change sign over the manifold?")
print("Testing a few seam metrics with eta != 0")
print(f"{'='*70}")

test_cases = [
    ("cos*cos", lambda t1,t2: np.cos(t1)*np.cos(t2), (1.0, 1.0, 0, 0, 0, 0.1, 0.1, 0.1)),
    ("cos*cos", lambda t1,t2: np.cos(t1)*np.cos(t2), (1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)),
    ("sin*sin", lambda t1,t2: np.sin(t1)*np.sin(t2), (1.0, 1.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1)),
    ("cos+cos", lambda t1,t2: np.cos(t1)+np.cos(t2), (1.0, 1.0, 0, 0, 0.15, 0, 0, 0.15)),
]

for sname, s_func, params in test_cases:
    print(f"\nSeam: {sname}, params: {params}")
    K_min = float('inf')
    K_max = float('-inf')
    K_min_pos = None
    K_max_pos = None

    for t1 in np.linspace(0.3, np.pi-0.3, 30):
        for t2 in np.linspace(0.3, np.pi-0.3, 30):
            K = compute_all_mixed_curvatures(t1, t2, s_func, params)
            if K is None:
                continue
            K02 = K.get((0,2))
            if K02 is not None:
                if K02 < K_min:
                    K_min = K02
                    K_min_pos = (t1,t2)
                if K02 > K_max:
                    K_max = K02
                    K_max_pos = (t1,t2)

    if K_min == float('inf'):
        print("  No valid points")
    else:
        print(f"  K(θ₁,θ₂) range: [{K_min:.4e}, {K_max:.4e}]")
        if K_min <= 0 and K_max > 0:
            print(f"  -> Curvature CHANGES SIGN (obstruction confirmed)")
        elif K_min > 0:
            print(f"  -> All positive in interior (check boundary!)")
        else:
            print(f"  -> All non-positive")
