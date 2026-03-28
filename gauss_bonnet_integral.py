"""
Compute the total Gaussian curvature integral:
  I = int_{[0,pi]^2} K * sqrt(det(g_2D)) d(theta_1) d(theta_2)
for various seam metrics, to check if Gauss-Bonnet constrains the sign of K.
"""
import numpy as np

N = 200
margin = 0.01
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
    detB = -L1_2/2*(L1_2/2*L2 - E_coeff*L2_1/2) + L2_1/2*(L1_2/2*E_coeff - L1*L2_1/2)

    K = (detA - detB) / D**2
    return K, D


np.random.seed(42)

print("Gauss-Bonnet integral: I = ∫K√(det g_2D) dθ₁dθ₂")
print("=" * 70)

seams = {
    "cos+cos": lambda t1, t2: np.cos(t1) + np.cos(t2),
    "cos*cos": lambda t1, t2: np.cos(t1) * np.cos(t2),
    "sin*sin": lambda t1, t2: np.sin(t1) * np.sin(t2),
}

# Diagonal cases
print("\n--- DIAGONAL CASES ---")
for sname, sfunc in seams.items():
    for trial in range(20):
        a1 = np.random.uniform(0.8, 1.2)
        a2 = np.random.uniform(0.8, 1.2)
        b1 = np.random.uniform(-0.15, 0.15)
        b2 = np.random.uniform(-0.15, 0.15)
        g1 = np.random.uniform(-0.1, 0.1)
        g2 = np.random.uniform(-0.1, 0.1)
        params = (a1, a2, b1, b2, 0, g1, g2, 0)
        L1, L2, E = metric_grid(sfunc, params)
        D = L1*L2 - E**2
        if np.min(L1) <= 0 or np.min(L2) <= 0 or np.min(D) <= 0:
            continue
        K, _ = gauss_K_brioschi(L1, L2, E)
        sqrtD = np.sqrt(D)
        I = np.sum(K * sqrtD) * dt**2
        print(f"  {sname} #{trial:2d}: I={I: .6f}  Kmin={np.min(K):.4f}  Kmax={np.max(K):.4f}")

# Full cases
print("\n--- FULL CASES ---")
for sname, sfunc in seams.items():
    for trial in range(20):
        a1 = np.random.uniform(0.8, 1.2)
        a2 = np.random.uniform(0.8, 1.2)
        b1 = np.random.uniform(-0.1, 0.1)
        b2 = np.random.uniform(-0.1, 0.1)
        b12 = np.random.uniform(-0.1, 0.1)
        g1  = np.random.uniform(-0.1, 0.1)
        g2  = np.random.uniform(-0.1, 0.1)
        g12 = np.random.uniform(-0.1, 0.1)
        params = (a1, a2, b1, b2, b12, g1, g2, g12)
        L1, L2, E = metric_grid(sfunc, params)
        D = L1*L2 - E**2
        if np.min(L1) <= 0 or np.min(L2) <= 0 or np.min(D) <= 0:
            continue
        K, _ = gauss_K_brioschi(L1, L2, E)
        sqrtD = np.sqrt(D)
        I = np.sum(K * sqrtD) * dt**2
        print(f"  {sname} #{trial:2d}: I={I: .6f}  Kmin={np.min(K):.4f}  Kmax={np.max(K):.4f}")
