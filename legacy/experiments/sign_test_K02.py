"""
Test whether K = Gaussian curvature of the orbit 2D metric has to be <= 0
somewhere, by trying many seam functions and parameter sets.

For the DIAGONAL case (eta=0), the 2D metric is [[L1, 0],[0, L2]].
The Gaussian curvature is given by the standard Liouville formula.

Key question: Does K change sign on (0,pi)^2 for ALL diagonal seam metrics?
If so, the theorem is true (just with a different proof).
"""
import numpy as np

N = 200  # high res
margin = 0.02
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
    return L1, L2, E


def gauss_K_brioschi(L1, L2, E_coeff):
    """Gaussian curvature of 2D metric [[L1,E],[E,L2]] via CORRECTED Brioschi."""
    h = dt
    pL1 = np.pad(L1, 1, mode='edge')
    pL2 = np.pad(L2, 1, mode='edge')
    pE  = np.pad(E_coeff, 1, mode='edge')

    L1_1 = (pL1[2:,1:-1] - pL1[:-2,1:-1]) / (2*h)
    L1_2 = (pL1[1:-1,2:] - pL1[1:-1,:-2]) / (2*h)
    L2_1 = (pL2[2:,1:-1] - pL2[:-2,1:-1]) / (2*h)
    L2_2 = (pL2[1:-1,2:] - pL2[1:-1,:-2]) / (2*h)
    E_1  = (pE[2:,1:-1]  - pE[:-2,1:-1])  / (2*h)
    E_2  = (pE[1:-1,2:]  - pE[1:-1,:-2])  / (2*h)

    L1_22 = (pL1[1:-1,2:] - 2*L1 + pL1[1:-1,:-2]) / h**2
    L2_11 = (pL2[2:,1:-1] - 2*L2 + pL2[:-2,1:-1]) / h**2
    E_12  = (pE[2:,2:] - pE[2:,:-2] - pE[:-2,2:] + pE[:-2,:-2]) / (4*h**2)

    D = L1*L2 - E_coeff**2

    a11 = -L1_22/2 + E_12 - L2_11/2
    a12 = L1_1/2
    a13 = E_1 - L1_2/2
    a21 = E_2 - L2_1/2
    a22 = L1; a23 = E_coeff
    a31 = L2_2/2
    a32 = E_coeff; a33 = L2

    detA = a11*(a22*a33-a23*a32) - a12*(a21*a33-a23*a31) + a13*(a21*a32-a22*a31)

    # Corrected B matrix: [[0, E_v/2, G_u/2], [E_v/2, E, F], [G_u/2, F, G]]
    detB = -L1_2/2*(L1_2/2*L2 - E_coeff*L2_1/2) + L2_1/2*(L1_2/2*E_coeff - L1*L2_1/2)

    K = (detA - detB) / D**2
    return K


def check_sign_change(s_func, params, label=""):
    """Check if K of the 2D orbit metric changes sign."""
    L1, L2, E = metric_grid(s_func, params)
    D = L1*L2 - E**2

    # Check positivity
    if np.min(L1) <= 0 or np.min(L2) <= 0 or np.min(D) <= 0:
        return None

    K = gauss_K_brioschi(L1, L2, E)

    Kmin = np.min(K)
    Kmax = np.max(K)
    has_neg = Kmin < -1e-10
    has_pos = Kmax > 1e-10

    return Kmin, Kmax, has_neg, has_pos


# ── Run tests ──
print("Does the Gaussian curvature of the orbit metric ALWAYS change sign?")
print("=" * 70)

seams = {
    "cos+cos": lambda t1, t2: np.cos(t1) + np.cos(t2),
    "cos*cos": lambda t1, t2: np.cos(t1) * np.cos(t2),
    "sin*sin": lambda t1, t2: np.sin(t1) * np.sin(t2),
    "cos(2t1)+cos(2t2)": lambda t1, t2: np.cos(2*t1) + np.cos(2*t2),
    "sin*cos":  lambda t1, t2: np.sin(t1) * np.cos(t2),
}

# Diagonal cases (b12=g12=0)
print("\n--- DIAGONAL CASES (eta=0) ---")
np.random.seed(42)

n_all_pos = 0
n_valid = 0
n_total = 0

for sname, sfunc in seams.items():
    for trial in range(100):
        n_total += 1
        a1 = np.random.uniform(0.7, 1.3)
        a2 = np.random.uniform(0.7, 1.3)
        b1 = np.random.uniform(-0.3, 0.3)
        b2 = np.random.uniform(-0.3, 0.3)
        g1 = np.random.uniform(-0.2, 0.2)
        g2 = np.random.uniform(-0.2, 0.2)
        params = (a1, a2, b1, b2, 0, g1, g2, 0)
        result = check_sign_change(sfunc, params)
        if result is None:
            continue
        n_valid += 1
        Kmin, Kmax, has_neg, has_pos = result
        if has_pos and not has_neg:
            n_all_pos += 1
            print(f"  ALL POSITIVE! {sname} trial {trial}: Kmin={Kmin:.6f} Kmax={Kmax:.6f}")
            print(f"    params: {params}")

print(f"\nDiagonal: {n_all_pos}/{n_valid} all-positive out of {n_valid} valid (total {n_total})")

# Full cases (eta != 0)
print("\n--- FULL CASES (eta != 0) ---")
np.random.seed(42)

n_all_pos = 0
n_valid = 0
n_total = 0

for sname, sfunc in seams.items():
    for trial in range(100):
        n_total += 1
        a1 = np.random.uniform(0.7, 1.3)
        a2 = np.random.uniform(0.7, 1.3)
        b1 = np.random.uniform(-0.2, 0.2)
        b2 = np.random.uniform(-0.2, 0.2)
        b12 = np.random.uniform(-0.15, 0.15)
        g1 = np.random.uniform(-0.15, 0.15)
        g2 = np.random.uniform(-0.15, 0.15)
        g12 = np.random.uniform(-0.15, 0.15)
        params = (a1, a2, b1, b2, b12, g1, g2, g12)
        result = check_sign_change(sfunc, params)
        if result is None:
            continue
        n_valid += 1
        Kmin, Kmax, has_neg, has_pos = result
        if has_pos and not has_neg:
            n_all_pos += 1
            print(f"  ALL POSITIVE! {sname} trial {trial}: Kmin={Kmin:.6f} Kmax={Kmax:.6f}")
            print(f"    params: {params}")

print(f"\nFull: {n_all_pos}/{n_valid} all-positive out of {n_valid} valid (total {n_total})")

# Also try the product metric (all coefficients = 0 except a1=a2=1)
print("\n--- PRODUCT METRIC (a1=a2=1, rest=0) ---")
for sname, sfunc in seams.items():
    params = (1.0, 1.0, 0, 0, 0, 0, 0, 0)
    result = check_sign_change(sfunc, params)
    if result is None:
        print(f"  {sname}: not pos-def")
    else:
        Kmin, Kmax, has_neg, has_pos = result
        print(f"  {sname}: Kmin={Kmin:.6e}  Kmax={Kmax:.6e}  sign={'+' if has_pos else ''}{'-' if has_neg else ''}")
