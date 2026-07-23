"""
1. Verify diagonal case (Prop 4.4): does K_mix <= 0 at max of Phi = log(lam1*lam2)
   when eta = 0?

2. Test the Chern-Gauss-Bonnet approach: compute the Pfaffian integrand and check
   whether positive mixed curvature forces the integral to exceed chi = 4.
"""
import numpy as np

N = 60
margin = 0.05
th1_1d = np.linspace(margin, np.pi - margin, N)
th2_1d = np.linspace(margin, np.pi - margin, N)
dt = th1_1d[1] - th1_1d[0]
TH1, TH2 = np.meshgrid(th1_1d, th2_1d, indexing='ij')

def seam_derivs_grid(s_func, h=1e-5):
    s = s_func(TH1, TH2)
    s1 = (s_func(TH1+h, TH2) - s_func(TH1-h, TH2)) / (2*h)
    s2 = (s_func(TH1, TH2+h) - s_func(TH1, TH2-h)) / (2*h)
    s11 = (s_func(TH1+h, TH2) - 2*s + s_func(TH1-h, TH2)) / h**2
    s22 = (s_func(TH1, TH2+h) - 2*s + s_func(TH1, TH2-h)) / h**2
    s12 = (s_func(TH1+h, TH2+h) - s_func(TH1+h, TH2-h)
           - s_func(TH1-h, TH2+h) + s_func(TH1-h, TH2-h)) / (4*h**2)
    return s, s1, s2, s11, s22, s12

def metric_grid(s_func, params):
    a1, a2, b1, b2, b12, g1, g2, g12 = params
    s, s1, s2, s11, s22, s12 = seam_derivs_grid(s_func)
    L1 = a1 + b1*s1**2 + g1*s11
    L2 = a2 + b2*s2**2 + g2*s22
    E  = b12*s1*s2 + g12*s12
    ct1 = np.cos(TH1) / np.sin(TH1)
    ct2 = np.cos(TH2) / np.sin(TH2)
    M1 = a1 + g1*s1*ct1
    M2 = a2 + g2*s2*ct2
    return L1, L2, E, M1, M2

def gauss_K_grid(L1, L2, E):
    """Gaussian curvature of 2D metric [[L1,E],[E,L2]] via Brioschi."""
    h = dt
    pL1 = np.pad(L1, 1, mode='edge')
    pL2 = np.pad(L2, 1, mode='edge')
    pE  = np.pad(E,  1, mode='edge')

    L1_1 = (pL1[2:,1:-1] - pL1[:-2,1:-1]) / (2*h)
    L1_2 = (pL1[1:-1,2:] - pL1[1:-1,:-2]) / (2*h)
    L2_1 = (pL2[2:,1:-1] - pL2[:-2,1:-1]) / (2*h)
    L2_2 = (pL2[1:-1,2:] - pL2[1:-1,:-2]) / (2*h)
    E_1  = (pE[2:,1:-1]  - pE[:-2,1:-1])  / (2*h)
    E_2  = (pE[1:-1,2:]  - pE[1:-1,:-2])  / (2*h)

    L1_22 = (pL1[1:-1,2:] - 2*L1 + pL1[1:-1,:-2]) / h**2
    L2_11 = (pL2[2:,1:-1] - 2*L2 + pL2[:-2,1:-1]) / h**2
    E_12  = (pE[2:,2:] - pE[2:,:-2] - pE[:-2,2:] + pE[:-2,:-2]) / (4*h**2)

    D = L1*L2 - E**2

    a11 = -L1_22/2 + E_12 - L2_11/2
    a12 = L1_1/2
    a13 = E_1 - L1_2/2
    a21 = E_2 - L2_1/2
    a22 = L1; a23 = E
    a31 = L2_2/2
    a32 = E; a33 = L2

    detA = a11*(a22*a33-a23*a32) - a12*(a21*a33-a23*a31) + a13*(a21*a32-a22*a31)
    # Correct Brioschi B: [[0, E_v/2, G_u/2], [E_v/2, E, F], [G_u/2, F, G]]
    # E_v = L1_2, G_u = L2_1
    detB = -L1_2/2*(L1_2/2*L2 - E*L2_1/2) + L2_1/2*(L1_2/2*E - L1*L2_1/2)

    K = (detA - detB) / D**2
    return K, D


# ──────────────────────────────────
# TEST 1: Diagonal case (eta=0)
# ──────────────────────────────────
print("=" * 70)
print("TEST 1: Diagonal case verification (Prop 4.4)")
print("Does K_mix <= 0 at max of Phi = log(lam1*lam2) when eta=0?")
print("=" * 70)

seams = {
    "cos+cos": lambda t1,t2: np.cos(t1)+np.cos(t2),
    "cos*cos": lambda t1,t2: np.cos(t1)*np.cos(t2),
    "sin*sin": lambda t1,t2: np.sin(t1)*np.sin(t2),
}

# Diagonal params: b12=0, g12=0 => eta=0
diag_params = [
    ("a+b",   (1,1,.1,.1,0,0,0,0)),
    ("a+g",   (1,1,0,0,0,.1,.1,0)),
    ("a+b+g", (1,1,.1,.1,0,.1,.1,0)),
    ("a+b+g2",(1,1,.15,.15,0,.1,.1,0)),
    ("asymm", (1.2,.8,.1,.1,0,.05,.05,0)),
]

violations_diag = 0
tests_diag = 0

for sname, s_func in seams.items():
    for pdesc, params in diag_params:
        L1, L2, E, M1, M2 = metric_grid(s_func, params)
        # Verify eta = 0
        assert np.max(np.abs(E)) < 1e-10, "eta not zero!"

        if np.any(L1 <= 0) or np.any(L2 <= 0) or np.any(M1 <= 0) or np.any(M2 <= 0):
            continue

        Phi = np.log(L1 * L2)
        K, D = gauss_K_grid(L1, L2, E)

        # Find max of Phi
        idx = np.unravel_index(np.argmax(Phi), Phi.shape)
        K_at_max = K[idx]
        Phi_max = Phi[idx]

        tests_diag += 1
        status = "OK (K<=0)" if K_at_max <= 1e-8 else "VIOLATION!"
        if K_at_max > 1e-8:
            violations_diag += 1

        print(f"  {sname:<10} {pdesc:<8}: K={K_at_max:>10.6f} at max(Phi)={Phi_max:.4f}  {status}")

print(f"\nDiagonal case: {violations_diag}/{tests_diag} violations")
if violations_diag == 0:
    print(">>> Prop 4.4 confirmed: K_mix <= 0 at max(Phi) in all diagonal cases.")


# ──────────────────────────────────
# TEST 2: Diagonal case with more random params
# ──────────────────────────────────
print(f"\n{'='*70}")
print("Diagonal case: 200 random parameter sets")
print(f"{'='*70}")

np.random.seed(7)
diag_violations = 0
diag_total = 0

for trial in range(200):
    sname = list(seams.keys())[trial % len(seams)]
    s_func = seams[sname]
    eps = np.random.uniform(0.02, 0.25)
    a1 = 1 + np.random.uniform(-.15,.15)
    a2 = 1 + np.random.uniform(-.15,.15)
    b1 = np.random.uniform(-eps, eps)
    b2 = np.random.uniform(-eps, eps)
    g1 = np.random.uniform(-eps, eps)
    g2 = np.random.uniform(-eps, eps)
    params = (a1, a2, b1, b2, 0, g1, g2, 0)  # eta = 0

    L1, L2, E, M1, M2 = metric_grid(s_func, params)
    if np.any(L1 <= 0) or np.any(L2 <= 0) or np.any(M1 <= 0) or np.any(M2 <= 0):
        continue
    D = L1*L2 - E**2
    if np.any(D <= 0):
        continue

    Phi = np.log(L1 * L2)
    K, _ = gauss_K_grid(L1, L2, E)

    idx = np.unravel_index(np.argmax(Phi), Phi.shape)
    K_at_max = K[idx]

    diag_total += 1
    if K_at_max > 1e-6:
        diag_violations += 1
        print(f"  VIOLATION: trial {trial}, {sname}, K={K_at_max:.6e}")
        print(f"    params: {params}")

print(f"\n{diag_violations}/{diag_total} violations in diagonal case")


# ──────────────────────────────────
# TEST 3: Full case — try auxiliary functions
# ──────────────────────────────────
print(f"\n{'='*70}")
print("TEST 3: Full case — test various auxiliary functions")
print("Does K_mix <= 0 at max of F for different choices of F?")
print(f"{'='*70}")

seams_full = {
    "cos+cos": lambda t1,t2: np.cos(t1)+np.cos(t2),
    "cos*cos": lambda t1,t2: np.cos(t1)*np.cos(t2),
    "sin*sin": lambda t1,t2: np.sin(t1)*np.sin(t2),
}

full_params = [
    ("all .1",  (1,1,.1,.1,.1,.1,.1,.1)),
    ("all .15", (1,1,.15,.15,.15,.15,.15,.15)),
    ("all .2",  (1,1,.2,.2,.2,.2,.2,.2)),
    ("b12 only",(1,1,0,0,.15,0,0,0)),
    ("g12 only",(1,1,0,0,0,0,0,.15)),
    ("mixed",   (1,1,.1,.1,.15,.05,.05,.1)),
]

aux_funcs = {
    "log(L1*L2)":       lambda L1,L2,E,M1,M2: np.log(L1*L2),
    "log(L1*L2-E^2)":   lambda L1,L2,E,M1,M2: np.log(L1*L2 - E**2),
    "log(L1*L2*M1*M2)": lambda L1,L2,E,M1,M2: np.log(L1*L2*M1*M2),
    "log(det4)":         lambda L1,L2,E,M1,M2: np.log((L1*L2-E**2)*M1*M2*np.sin(TH1)**2*np.sin(TH2)**2 + 1e-30),
    "L1+L2":            lambda L1,L2,E,M1,M2: L1+L2,
    "L1*L2":            lambda L1,L2,E,M1,M2: L1*L2,
    "L1*L2-E^2":        lambda L1,L2,E,M1,M2: L1*L2-E**2,
    "tr(g2D)":          lambda L1,L2,E,M1,M2: L1+L2,
    "log(M1*M2)":       lambda L1,L2,E,M1,M2: np.log(M1*M2),
}

for sname, s_func in seams_full.items():
    for pdesc, params in full_params:
        L1, L2, E, M1, M2 = metric_grid(s_func, params)
        if np.any(L1 <= 0) or np.any(L2 <= 0) or np.any(M1 <= 0) or np.any(M2 <= 0):
            continue
        D = L1*L2 - E**2
        if np.any(D <= 0):
            continue

        K, _ = gauss_K_grid(L1, L2, E)

        print(f"\n  {sname} / {pdesc}:")
        for fname, F_func in aux_funcs.items():
            try:
                F = F_func(L1, L2, E, M1, M2)
                if np.any(~np.isfinite(F)):
                    continue
                idx = np.unravel_index(np.argmax(F[2:-2,2:-2]) + 2*F.shape[1]+2,
                                       F.shape) if F.shape[0] > 4 else np.unravel_index(np.argmax(F), F.shape)
                # Simpler: just max over entire grid
                idx = np.unravel_index(np.argmax(F), F.shape)
                K_at_max = K[idx]
                status = "≤0" if K_at_max <= 1e-8 else f">0 ({K_at_max:.4f})"
                print(f"    max({fname}): K={K_at_max:>10.4f}  {status}")
            except:
                pass

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
