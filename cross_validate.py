"""
Cross-validate Brioschi K_2D vs full 4x4 Riemann tensor K_02.
Self-contained — no imports from other project scripts.
"""
import numpy as np


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


def compute_K02_riemann(theta1, theta2, s_func, params, h=5e-4):
    """Compute K_{02} sectional curvature using full 4x4 Riemann tensor."""
    def g_at(t1, t2):
        return build_metric(t1, t2, s_func, params)

    g0 = g_at(theta1, theta2)
    n = 4
    eigvals = np.linalg.eigvalsh(g0)
    if np.min(eigvals) <= 0:
        return None

    ginv0 = np.linalg.inv(g0)

    dg = np.zeros((n,n,n))
    for c in [0, 2]:
        eps = np.zeros(4); eps[c] = h
        gp = g_at(theta1 + eps[0], theta2 + eps[2])
        gm = g_at(theta1 - eps[0], theta2 - eps[2])
        dg[:,:,c] = (gp - gm) / (2*h)

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

    Gamma = np.zeros((n,n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    Gamma[i,j,k] += 0.5 * ginv0[i,l] * (dg[l,j,k] + dg[l,k,j] - dg[j,k,l])

    dginv = np.zeros((n,n,n))
    for i in range(n):
        for l in range(n):
            for c in range(n):
                for m in range(n):
                    for nn_ in range(n):
                        dginv[i,l,c] -= ginv0[i,m] * ginv0[l,nn_] * dg[m,nn_,c]

    dGamma = np.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for c in range(n):
                    for l in range(n):
                        dGamma[i,j,k,c] += 0.5 * dginv[i,l,c] * (dg[l,j,k] + dg[l,k,j] - dg[j,k,l])
                        dGamma[i,j,k,c] += 0.5 * ginv0[i,l] * (ddg[l,j,k,c] + ddg[l,k,j,c] - ddg[j,k,l,c])

    def R_up(i, j, k, l):
        v = dGamma[i,j,l,k] - dGamma[i,j,k,l]
        for m in range(n):
            v += Gamma[i,k,m]*Gamma[m,j,l] - Gamma[i,l,m]*Gamma[m,j,k]
        return v

    def R_down(i, j, k, l):
        v = 0
        for m in range(n):
            v += g0[i,m] * R_up(m, j, k, l)
        return v

    R_0202 = R_down(0, 2, 0, 2)
    denom = g0[0,0]*g0[2,2] - g0[0,2]**2
    if abs(denom) < 1e-15:
        return None
    return R_0202 / denom

def seam_derivs_at(s_func, t1, t2, h=1e-5):
    s = s_func(t1, t2)
    s1 = (s_func(t1+h, t2) - s_func(t1-h, t2)) / (2*h)
    s2 = (s_func(t1, t2+h) - s_func(t1, t2-h)) / (2*h)
    s11 = (s_func(t1+h, t2) - 2*s + s_func(t1-h, t2)) / h**2
    s22 = (s_func(t1, t2+h) - 2*s + s_func(t1, t2-h)) / h**2
    s12 = (s_func(t1+h, t2+h) - s_func(t1+h, t2-h)
           - s_func(t1-h, t2+h) + s_func(t1-h, t2-h)) / (4*h**2)
    return s, s1, s2, s11, s22, s12

def brioschi_K_at(s_func, params, t1, t2, h=5e-4):
    """Compute Gaussian curvature of 2D metric [[L1, E],[E, L2]]
    at a single point (t1,t2) using Brioschi formula with finite diffs."""
    a1, a2, b1, b2, b12, g1, g2, g12 = params

    def L1_at(u, v):
        _, s1, _, s11, _, _ = seam_derivs_at(s_func, u, v)
        return a1 + b1*s1**2 + g1*s11

    def L2_at(u, v):
        _, _, s2, _, s22, _ = seam_derivs_at(s_func, u, v)
        return a2 + b2*s2**2 + g2*s22

    def E_at(u, v):
        _, s1, s2, _, _, s12 = seam_derivs_at(s_func, u, v)
        return b12*s1*s2 + g12*s12

    L1  = L1_at(t1, t2)
    L2  = L2_at(t1, t2)
    EE  = E_at(t1, t2)

    # First derivatives
    L1_1 = (L1_at(t1+h, t2) - L1_at(t1-h, t2)) / (2*h)
    L1_2 = (L1_at(t1, t2+h) - L1_at(t1, t2-h)) / (2*h)
    L2_1 = (L2_at(t1+h, t2) - L2_at(t1-h, t2)) / (2*h)
    L2_2 = (L2_at(t1, t2+h) - L2_at(t1, t2-h)) / (2*h)
    E_1  = (E_at(t1+h, t2) - E_at(t1-h, t2)) / (2*h)
    E_2  = (E_at(t1, t2+h) - E_at(t1, t2-h)) / (2*h)

    # Second derivatives
    L1_22 = (L1_at(t1, t2+h) - 2*L1 + L1_at(t1, t2-h)) / h**2
    L2_11 = (L2_at(t1+h, t2) - 2*L2 + L2_at(t1-h, t2)) / h**2
    E_12  = (E_at(t1+h, t2+h) - E_at(t1+h, t2-h)
             - E_at(t1-h, t2+h) + E_at(t1-h, t2-h)) / (4*h**2)

    D = L1*L2 - EE**2

    # Brioschi A matrix
    a11 = -L1_22/2 + E_12 - L2_11/2
    a12 = L1_1/2
    a13 = E_1 - L1_2/2
    a21 = E_2 - L2_1/2
    a22 = L1; a23 = EE
    a31 = L2_2/2
    a32 = EE; a33 = L2

    detA = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)

    # Brioschi B matrix: [[0, E_v/2, G_u/2], [E_v/2, E, F], [G_u/2, F, G]]
    # u = theta1, v = theta2; E_coeff = L1, F = EE, G = L2
    # E_v = L1_2, G_u = L2_1
    detB = -L1_2/2*(L1_2/2*L2 - EE*L2_1/2) + L2_1/2*(L1_2/2*EE - L1*L2_1/2)

    K = (detA - detB) / D**2
    return K


# Test cases
print("Cross-validating Brioschi K_2D vs 4x4 Riemann R_{0202}")
print("=" * 70)

seams = {
    "cos+cos": lambda t1, t2: np.cos(t1) + np.cos(t2),
    "cos*cos": lambda t1, t2: np.cos(t1) * np.cos(t2),
    "sin*sin": lambda t1, t2: np.sin(t1) * np.sin(t2),
}

# Diagonal case: b12=g12=0
test_cases = [
    ("a+b",   (1.0, 1.0, 0.1, 0.1, 0, 0, 0, 0)),
    ("a+g",   (1.0, 1.0, 0, 0, 0, 0.1, 0.1, 0)),
    ("a+b+g", (1.0, 1.0, 0.1, 0.1, 0, 0.05, 0.05, 0)),
]

# Also test with eta != 0
test_cases_eta = [
    ("b12",  (1.0, 1.0, 0.1, 0.1, 0.05, 0, 0, 0)),
    ("g12",  (1.0, 1.0, 0, 0, 0, 0.1, 0.1, 0.05)),
    ("both", (1.0, 1.0, 0.1, 0.1, 0.05, 0.05, 0.05, 0.03)),
]

# Test points
test_points = [(1.0, 1.5), (np.pi/2, np.pi/2)]

for sname, sfunc in seams.items():
    for label, params in test_cases + test_cases_eta:
        print(f"\n{sname} / {label}:")
        for t1, t2 in test_points:
            K_brioschi = brioschi_K_at(sfunc, params, t1, t2)
            K_riemann = compute_K02_riemann(t1, t2, sfunc, params)
            if K_riemann is None:
                print(f"  ({t1:.2f},{t2:.2f}): metric not pos-def")
                continue
            K02 = K_riemann
            if K02 is None:
                print(f"  ({t1:.2f},{t2:.2f}): denom zero")
                continue
            diff = abs(K_brioschi - K02)
            rel = diff / max(abs(K02), 1e-12)
            status = "OK" if rel < 0.01 else "MISMATCH!"
            print(f"  ({t1:.2f},{t2:.2f}): K_2D={K_brioschi: .6f}  K_02={K02: .6f}  "
                  f"diff={diff:.2e} rel={rel:.2e}  {status}")
