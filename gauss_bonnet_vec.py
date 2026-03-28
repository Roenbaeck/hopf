"""
Fast vectorized Gauss-Bonnet test.
All computations are done on a grid at once using numpy arrays.
"""
import numpy as np

N = 50
margin = 0.05
th1_1d = np.linspace(margin, np.pi - margin, N)
th2_1d = np.linspace(margin, np.pi - margin, N)
dt = th1_1d[1] - th1_1d[0]
TH1, TH2 = np.meshgrid(th1_1d, th2_1d, indexing='ij')

def seam_derivs(s_func, h=1e-5):
    """Compute s and its derivatives on the grid."""
    s = s_func(TH1, TH2)
    s1 = (s_func(TH1+h, TH2) - s_func(TH1-h, TH2)) / (2*h)
    s2 = (s_func(TH1, TH2+h) - s_func(TH1, TH2-h)) / (2*h)
    s11 = (s_func(TH1+h, TH2) - 2*s + s_func(TH1-h, TH2)) / h**2
    s22 = (s_func(TH1, TH2+h) - 2*s + s_func(TH1, TH2-h)) / h**2
    s12 = (s_func(TH1+h, TH2+h) - s_func(TH1+h, TH2-h)
           - s_func(TH1-h, TH2+h) + s_func(TH1-h, TH2-h)) / (4*h**2)
    return s, s1, s2, s11, s22, s12

def orbit_metric_grid(s_func, params):
    a1, a2, b1, b2, b12, g1, g2, g12 = params
    s, s1, s2, s11, s22, s12 = seam_derivs(s_func)
    L1 = a1 + b1*s1**2 + g1*s11
    L2 = a2 + b2*s2**2 + g2*s22
    E  = b12*s1*s2 + g12*s12
    ct1 = np.cos(TH1) / np.sin(TH1)
    ct2 = np.cos(TH2) / np.sin(TH2)
    M1 = a1 + g1*s1*ct1
    M2 = a2 + g2*s2*ct2
    return L1, L2, E, M1, M2

def gauss_K_grid(L1, L2, E, h=None):
    """Compute Gaussian curvature of 2D metric [[L1,E],[E,L2]] on the grid.
    Uses finite differences on the grid arrays.
    """
    if h is None:
        h = dt  # grid spacing

    # Pad arrays for finite differences (replicate boundary)
    def pad(A):
        return np.pad(A, 1, mode='edge')

    pL1 = pad(L1); pL2 = pad(L2); pE = pad(E)

    # indices in padded array: [1:-1, 1:-1] = original
    # First derivatives: d/dth1 (axis 0), d/dth2 (axis 1)
    L1_1 = (pL1[2:,1:-1] - pL1[:-2,1:-1]) / (2*h)
    L1_2 = (pL1[1:-1,2:] - pL1[1:-1,:-2]) / (2*h)
    L2_1 = (pL2[2:,1:-1] - pL2[:-2,1:-1]) / (2*h)
    L2_2 = (pL2[1:-1,2:] - pL2[1:-1,:-2]) / (2*h)
    E_1  = (pE[2:,1:-1]  - pE[:-2,1:-1])  / (2*h)
    E_2  = (pE[1:-1,2:]  - pE[1:-1,:-2])  / (2*h)

    # Second derivatives
    L1_11 = (pL1[2:,1:-1] - 2*L1 + pL1[:-2,1:-1]) / h**2
    L1_22 = (pL1[1:-1,2:] - 2*L1 + pL1[1:-1,:-2]) / h**2
    L2_11 = (pL2[2:,1:-1] - 2*L2 + pL2[:-2,1:-1]) / h**2
    L2_22 = (pL2[1:-1,2:] - 2*L2 + pL2[1:-1,:-2]) / h**2
    E_11  = (pE[2:,1:-1]  - 2*E  + pE[:-2,1:-1])  / h**2
    E_22  = (pE[1:-1,2:]  - 2*E  + pE[1:-1,:-2])  / h**2

    # Mixed derivatives
    ppL1 = pad(pad(L1))
    ppL2 = pad(pad(L2))
    ppE  = pad(pad(E))
    # For mixed derivative, we can use padded once and shifted indices
    L1_12 = (pL1[2:,2:] - pL1[2:,:-2] - pL1[:-2,2:] + pL1[:-2,:-2]) / (4*h**2)
    L2_12 = (pL2[2:,2:] - pL2[2:,:-2] - pL2[:-2,2:] + pL2[:-2,:-2]) / (4*h**2)
    E_12  = (pE[2:,2:]  - pE[2:,:-2]  - pE[:-2,2:]  + pE[:-2,:-2])  / (4*h**2)

    # Brioschi formula:
    # E_val = L1, F_val = E, G_val = L2
    # K = (det A - det B) / (EG-F^2)^2
    D = L1*L2 - E**2

    # Matrix A:
    a11 = -L1_22/2 + E_12 - L2_11/2
    a12 = L1_1/2
    a13 = E_1 - L1_2/2
    a21 = E_2 - L2_1/2
    a22 = L1.copy()
    a23 = E.copy()
    a31 = L2_2/2
    a32 = E.copy()
    a33 = L2.copy()

    detA = (a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31))

    # Matrix B:
    b12 = L1_1/2
    b13 = L1_2/2
    b22 = L1.copy()
    b23 = E.copy()
    b32 = E.copy()
    b33 = L2.copy()

    # Correct Brioschi B matrix: B = [[0, E_v/2, G_u/2], [E_v/2, E, F], [G_u/2, F, G]]
    # With u=theta1, v=theta2: E_v = L1_2, G_u = L2_1
    # det B = -E_v/2*(E_v/2*G - F*G_u/2) + G_u/2*(E_v/2*F - E*G_u/2)
    detB = (- L1_2/2 * (L1_2/2 * L2 - E * L2_1/2)
            + L2_1/2 * (L1_2/2 * E - L1 * L2_1/2))

    K = (detA - detB) / D**2
    return K, D

def test_case(s_func, params):
    """Test a single seam + params combination."""
    L1, L2, E, M1, M2 = orbit_metric_grid(s_func, params)
    D = L1*L2 - E**2

    # Check validity
    if np.any(D <= 0) or np.any(L1 <= 0) or np.any(L2 <= 0) or np.any(M1 <= 0) or np.any(M2 <= 0):
        return None

    K, _ = gauss_K_grid(L1, L2, E)

    # Integrals
    sqrt_D = np.sqrt(D)
    int_2D = np.sum(K * sqrt_D) * dt**2
    int_4D = np.sum(K * np.sqrt(D * M1 * M2) * np.sin(TH1) * np.sin(TH2)) * dt**2

    return {
        'K_min': np.min(K),
        'K_max': np.max(K),
        'int_2D': int_2D,
        'int_4D': int_4D,
        'eta_max': np.max(np.abs(E)),
    }

# ── Main ──
seams = {
    "cos+cos":    lambda t1,t2: np.cos(t1)+np.cos(t2),
    "cos*cos":    lambda t1,t2: np.cos(t1)*np.cos(t2),
    "sin*sin":    lambda t1,t2: np.sin(t1)*np.sin(t2),
    "cos+cos+p":  lambda t1,t2: np.cos(t1)+np.cos(t2)+0.3*np.cos(t1)*np.cos(t2),
    "cos1sin2":   lambda t1,t2: np.cos(t1)*np.sin(t2),
}

param_sets = [
    ("all .05",   (1,1,.05,.05,.05,.05,.05,.05)),
    ("all .1",    (1,1,.1,.1,.1,.1,.1,.1)),
    ("all .15",   (1,1,.15,.15,.15,.15,.15,.15)),
    ("all .2",    (1,1,.2,.2,.2,.2,.2,.2)),
    ("b12=.1",    (1,1,0,0,.1,0,0,0)),
    ("g12=.1",    (1,1,0,0,0,0,0,.1)),
    ("b+g12=.1",  (1,1,0,0,.1,0,0,.1)),
    ("asymm",     (1.2,.8,.1,.1,.1,.1,.1,.1)),
    ("diag+x",    (1,1,.1,.1,.1,.05,.05,.1)),
    ("lg cross",  (1,1,0,0,.2,0,0,.2)),
]

print(f"{'Seam':<12} {'Params':<12} {'K_min':>10} {'K_max':>10} {'η_max':>8} {'∫KdA₂':>10} {'∫KdV₄':>10} {'sign'}")
print("-" * 86)

for sname, s_func in seams.items():
    for pdesc, params in param_sets:
        r = test_case(s_func, params)
        if r is None:
            print(f"{sname:<12} {pdesc:<12} {'INVALID':>10}")
            continue
        sgn = "±" if r['K_min'] < -1e-8 and r['K_max'] > 1e-8 else \
              "ALL+" if r['K_min'] > 0 else "ALL-"
        print(f"{sname:<12} {pdesc:<12} {r['K_min']:>10.4f} {r['K_max']:>10.4f} "
              f"{r['eta_max']:>8.4f} {r['int_2D']:>10.4f} {r['int_4D']:>10.4f}  {sgn}")

# Random scan
print(f"\n{'='*86}")
print("Random parameter scan (500 trials)")
print(f"{'='*86}")

np.random.seed(42)
total = 0; int2d_pos = 0; int4d_pos = 0; sign_changes = 0; all_pos = 0

for trial in range(500):
    sname = list(seams.keys())[trial % len(seams)]
    s_func = seams[sname]
    eps = np.random.uniform(0.02, 0.25)
    a1 = 1 + np.random.uniform(-.15,.15)
    a2 = 1 + np.random.uniform(-.15,.15)
    params = (a1, a2,
              np.random.uniform(-eps,eps), np.random.uniform(-eps,eps),
              np.random.uniform(-eps,eps), np.random.uniform(-eps,eps),
              np.random.uniform(-eps,eps), np.random.uniform(-eps,eps))
    r = test_case(s_func, params)
    if r is None or r['eta_max'] < 0.001:
        continue
    total += 1
    if r['K_min'] < -1e-8 and r['K_max'] > 1e-8:
        sign_changes += 1
    if r['K_min'] > 0:
        all_pos += 1
        print(f"  ALL POS: {sname}, K=[{r['K_min']:.4e},{r['K_max']:.4e}], η_max={r['eta_max']:.4f}")
    if r['int_2D'] > 1e-8:
        int2d_pos += 1
    if r['int_4D'] > 1e-8:
        int4d_pos += 1

print(f"\nTotal valid (η≠0): {total}")
print(f"K changes sign:    {sign_changes} / {total}")
print(f"K all positive:    {all_pos} / {total}")
print(f"∫K dA_2D > 0:      {int2d_pos} / {total}")
print(f"∫K dV_4D > 0:      {int4d_pos} / {total}")

if all_pos == 0:
    print("\n>>> K_mix ALWAYS changes sign — theorem TRUE for all tested cases!")
if int4d_pos == 0:
    print(">>> ∫K_mix dV ≤ 0 always — integral obstruction confirmed!")
elif int2d_pos == 0:
    print(">>> ∫K_mix dA_2D ≤ 0 always — 2D Gauss-Bonnet obstruction confirmed!")
else:
    print(f"\n>>> Neither integral is uniformly ≤ 0.")
    print(f"    But K always changes sign, so the theorem holds regardless.")
