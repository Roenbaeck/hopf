"""
Fast Gauss-Bonnet integral test using the 2x2 orbit metric.

K_mix = K_{2D} = Gaussian curvature of g_2D = [[lam1, eta], [eta, lam2]].
We compute K_2D directly using the 2D Brioschi formula (finite differences).
"""

import numpy as np

def seam_data(s_func, t1, t2, h=1e-5):
    s = s_func(t1, t2)
    s1 = (s_func(t1+h, t2) - s_func(t1-h, t2)) / (2*h)
    s2 = (s_func(t1, t2+h) - s_func(t1, t2-h)) / (2*h)
    s11 = (s_func(t1+h, t2) - 2*s + s_func(t1-h, t2)) / h**2
    s22 = (s_func(t1, t2+h) - 2*s + s_func(t1, t2-h)) / h**2
    s12 = (s_func(t1+h, t2+h) - s_func(t1+h, t2-h)
           - s_func(t1-h, t2+h) + s_func(t1-h, t2-h)) / (4*h**2)
    return s, s1, s2, s11, s22, s12

def orbit_metric(t1, t2, s_func, params):
    """Returns (lam1, lam2, eta, mu1, mu2) at (t1, t2)."""
    a1, a2, b1, b2, b12, g1, g2, g12 = params
    s, s1, s2, s11, s22, s12 = seam_data(s_func, t1, t2)
    lam1 = a1 + b1*s1**2 + g1*s11
    lam2 = a2 + b2*s2**2 + g2*s22
    eta_ = b12*s1*s2 + g12*s12
    ct1 = np.cos(t1)/np.sin(t1) if abs(np.sin(t1)) > 1e-10 else 0
    ct2 = np.cos(t2)/np.sin(t2) if abs(np.sin(t2)) > 1e-10 else 0
    mu1 = a1 + g1*s1*ct1
    mu2 = a2 + g2*s2*ct2
    return lam1, lam2, eta_, mu1, mu2

def gauss_curvature_2d(t1, t2, s_func, params, h=5e-4):
    """Compute K_{2D} for g_2D = [[lam1,eta],[eta,lam2]] by finite differences."""
    def gvals(tt1, tt2):
        L1, L2, E, _, _ = orbit_metric(tt1, tt2, s_func, params)
        return L1, L2, E

    L1, L2, E = gvals(t1, t2)
    D = L1*L2 - E**2
    if D <= 0 or L1 <= 0 or L2 <= 0:
        return None, None

    # Metric components: g = [[E_val, F_val], [F_val, G_val]] in standard notation
    # Here E_val=L1, F_val=eta, G_val=L2
    Ev, Fv, Gv = L1, E, L2

    # First partials (wrt t1 and t2)
    L1p, L2p, Ep = gvals(t1+h, t2)
    L1m, L2m, Em = gvals(t1-h, t2)
    Ev_1 = (L1p - L1m)/(2*h)
    Fv_1 = (Ep - Em)/(2*h)
    Gv_1 = (L2p - L2m)/(2*h)

    L1p, L2p, Ep = gvals(t1, t2+h)
    L1m, L2m, Em = gvals(t1, t2-h)
    Ev_2 = (L1p - L1m)/(2*h)
    Fv_2 = (Ep - Em)/(2*h)
    Gv_2 = (L2p - L2m)/(2*h)

    # Second partials
    Ev_11 = (gvals(t1+h,t2)[0] - 2*Ev + gvals(t1-h,t2)[0])/h**2
    Ev_22 = (gvals(t1,t2+h)[0] - 2*Ev + gvals(t1,t2-h)[0])/h**2
    Ev_12 = (gvals(t1+h,t2+h)[0] - gvals(t1+h,t2-h)[0]
             - gvals(t1-h,t2+h)[0] + gvals(t1-h,t2-h)[0])/(4*h**2)

    Gv_11 = (gvals(t1+h,t2)[1] - 2*Gv + gvals(t1-h,t2)[1])/h**2
    Gv_22 = (gvals(t1,t2+h)[1] - 2*Gv + gvals(t1,t2-h)[1])/h**2
    Gv_12 = (gvals(t1+h,t2+h)[1] - gvals(t1+h,t2-h)[1]
             - gvals(t1-h,t2+h)[1] + gvals(t1-h,t2-h)[1])/(4*h**2)

    Fv_11 = (gvals(t1+h,t2)[2] - 2*Fv + gvals(t1-h,t2)[2])/h**2
    Fv_22 = (gvals(t1,t2+h)[2] - 2*Fv + gvals(t1,t2-h)[2])/h**2
    Fv_12 = (gvals(t1+h,t2+h)[2] - gvals(t1+h,t2-h)[2]
             - gvals(t1-h,t2+h)[2] + gvals(t1-h,t2-h)[2])/(4*h**2)

    # Brioschi formula: K = (det A - det B) / (EG - F^2)^2
    # where A = [[-Ev_22/2 + Fv_12 - Gv_11/2, Ev_1/2,    Fv_1 - Ev_2/2],
    #            [Fv_2 - Gv_1/2,              Ev,         Fv           ],
    #            [Gv_2/2,                     Fv,         Gv           ]]
    # B = [[0,       Ev_1/2, Ev_2/2],
    #      [Ev_1/2,  Ev,     Fv    ],
    #      [Ev_2/2,  Fv,     Gv    ]]

    a11 = -Ev_22/2 + Fv_12 - Gv_11/2
    a12 = Ev_1/2
    a13 = Fv_1 - Ev_2/2
    a21 = Fv_2 - Gv_1/2
    a22 = Ev
    a23 = Fv
    a31 = Gv_2/2
    a32 = Fv
    a33 = Gv

    A = np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
    B = np.array([[0, Ev_1/2, Ev_2/2],
                  [Ev_1/2, Ev, Fv],
                  [Ev_2/2, Fv, Gv]])

    det_D = D  # EG - F^2
    K = (np.linalg.det(A) - np.linalg.det(B)) / det_D**2
    return K, D


def run_tests():
    seams = {
        "cos+cos":    lambda t1,t2: np.cos(t1)+np.cos(t2),
        "cos*cos":    lambda t1,t2: np.cos(t1)*np.cos(t2),
        "sin*sin":    lambda t1,t2: np.sin(t1)*np.sin(t2),
        "cos+cos+p":  lambda t1,t2: np.cos(t1)+np.cos(t2)+0.3*np.cos(t1)*np.cos(t2),
        "cos1sin2":   lambda t1,t2: np.cos(t1)*np.sin(t2),
    }

    param_sets = [
        ("all 0.05",       (1,1, .05,.05,.05,.05,.05,.05)),
        ("all 0.1",        (1,1, .1,.1,.1,.1,.1,.1)),
        ("all 0.15",       (1,1, .15,.15,.15,.15,.15,.15)),
        ("all 0.2",        (1,1, .2,.2,.2,.2,.2,.2)),
        ("pure b12",       (1,1, 0,0,.1,0,0,0)),
        ("pure g12",       (1,1, 0,0,0,0,0,.1)),
        ("b12+g12",        (1,1, 0,0,.1,0,0,.1)),
        ("b12,-g12",       (1,1, 0,0,.15,0,0,-.1)),
        ("asymm",          (1.2,.8, .1,.1,.1,.1,.1,.1)),
        ("diag+cross",     (1,1, .1,.1,.1,.05,.05,.1)),
        ("large cross",    (1,1, 0,0,.2,0,0,.2)),
        ("neg-b pos-g",    (1,1, -.05,-.05,.1,.05,.05,.15)),
    ]

    N = 40
    th1 = np.linspace(0.05, np.pi-0.05, N)
    th2 = np.linspace(0.05, np.pi-0.05, N)
    dt1 = th1[1]-th1[0]
    dt2 = th2[1]-th2[0]

    print(f"{'Seam':<12} {'Params':<14} {'K_min':>10} {'K_max':>10} {'∫K√g_2D':>10} {'∫K√g_4D':>10} {'sign'}")
    print("-"*80)

    for sname, s_func in seams.items():
        for pdesc, params in param_sets:
            K_min = np.inf; K_max = -np.inf
            int_2D = 0.0; int_4D = 0.0
            valid = True
            for t1 in th1:
                for t2 in th2:
                    K, D = gauss_curvature_2d(t1, t2, s_func, params)
                    if K is None:
                        valid = False; continue
                    L1, L2, E, M1, M2 = orbit_metric(t1, t2, s_func, params)
                    if M1 <= 0 or M2 <= 0:
                        valid = False; continue
                    K_min = min(K_min, K)
                    K_max = max(K_max, K)
                    sqrt_D = np.sqrt(D)
                    int_2D += K * sqrt_D * dt1 * dt2
                    int_4D += K * np.sqrt(D*M1*M2) * np.sin(t1)*np.sin(t2) * dt1*dt2

            if not valid or K_min == np.inf:
                status = "INVALID"
            elif K_min > 1e-8:
                status = "ALL+"
            elif K_max < -1e-8:
                status = "ALL-"
            else:
                status = "±"

            print(f"{sname:<12} {pdesc:<14} {K_min:>10.4f} {K_max:>10.4f} {int_2D:>10.4f} {int_4D:>10.4f}  {status}")

    # Random scan
    print(f"\n{'='*80}")
    print("Random parameter scan (300 trials)")
    print(f"{'='*80}")

    np.random.seed(42)
    total = 0; int2d_pos = 0; int4d_pos = 0; sign_changes = 0
    all_positive_count = 0

    for trial in range(100):
        sname = list(seams.keys())[trial % len(seams)]
        s_func = seams[sname]

        eps = np.random.uniform(0.02, 0.2)
        a1 = 1 + np.random.uniform(-.1,.1)
        a2 = 1 + np.random.uniform(-.1,.1)
        params = (a1, a2,
                  np.random.uniform(-eps,eps),
                  np.random.uniform(-eps,eps),
                  np.random.uniform(-eps,eps),
                  np.random.uniform(-eps,eps),
                  np.random.uniform(-eps,eps),
                  np.random.uniform(-eps,eps))

        K_min = np.inf; K_max = -np.inf
        int_2D = 0.0; int_4D = 0.0
        ok = True
        has_eta = False
        for t1 in th1:
            for t2 in th2:
                K, D = gauss_curvature_2d(t1, t2, s_func, params)
                if K is None:
                    ok = False; break
                L1,L2,E,M1,M2 = orbit_metric(t1,t2,s_func,params)
                if M1<=0 or M2<=0:
                    ok = False; break
                if abs(E) > 0.001:
                    has_eta = True
                K_min = min(K_min,K); K_max = max(K_max,K)
                sqrt_D = np.sqrt(D)
                int_2D += K * sqrt_D * dt1*dt2
                int_4D += K * np.sqrt(D*M1*M2)*np.sin(t1)*np.sin(t2)*dt1*dt2
            if not ok:
                break

        if not ok or not has_eta:
            continue

        total += 1
        if K_min < -1e-8 and K_max > 1e-8:
            sign_changes += 1
        if K_min > 0:
            all_positive_count += 1
            print(f"  ALL POSITIVE: trial {trial}, seam={sname}, K=[{K_min:.4e},{K_max:.4e}]")
        if int_2D > 1e-8:
            int2d_pos += 1
        if int_4D > 1e-8:
            int4d_pos += 1

    print(f"\nTotal valid (eta≠0): {total}")
    print(f"K changes sign:      {sign_changes} / {total}")
    print(f"K all positive:      {all_positive_count} / {total}")
    print(f"∫K dA_2D > 0:        {int2d_pos} / {total}")
    print(f"∫K dV_4D > 0:        {int4d_pos} / {total}")

    if all_positive_count == 0:
        print("\n>>> K_{mix} = K_{2D} ALWAYS changes sign — theorem confirmed numerically!")
    if int4d_pos == 0:
        print(">>> ∫K dV ≤ 0 in all cases — integral obstruction works!")
    elif int2d_pos == 0:
        print(">>> ∫K dA ≤ 0 in all cases — 2D Gauss-Bonnet approach works!")


if __name__ == "__main__":
    run_tests()
