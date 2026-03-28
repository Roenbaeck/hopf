"""
Check which formula for K_mix is correct in the diagonal case (eta=0):
  Formula A: K = -1/2 * [d2^2(log L1) + d1^2(log L2)]        (eq 4.8)
  Formula B: K = [-L2/2*d2^2(log L1) - L1/2*d1^2(log L2)] / (L1*L2)  (eq 4.6 with eta=0)
vs the direct 4x4 Riemann tensor computation.
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


def build_metric(theta1, theta2, s_func, params):
    a1, a2, b1, b2, b12, g1, g2, g12 = params
    s, s1, s2, s11, s22, s12 = seam_data(s_func, theta1, theta2)
    lam1 = a1 + b1 * s1**2 + g1 * s11
    lam2 = a2 + b2 * s2**2 + g2 * s22
    eta_ = b12 * s1 * s2 + g12 * s12
    ct1 = np.cos(theta1) / np.sin(theta1)
    ct2 = np.cos(theta2) / np.sin(theta2)
    mu1 = a1 + g1 * s1 * ct1
    mu2 = a2 + g2 * s2 * ct2
    g = np.zeros((4,4))
    g[0,0] = lam1; g[1,1] = mu1 * np.sin(theta1)**2
    g[2,2] = lam2; g[3,3] = mu2 * np.sin(theta2)**2
    g[0,2] = eta_; g[2,0] = eta_
    return g


def compute_K02_riemann(theta1, theta2, s_func, params, h=5e-4):
    """Full 4x4 Riemann R_{0202}/denom."""
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
        dg[:,:,c] = (g_at(theta1 + eps[0], theta2 + eps[2]) - g_at(theta1 - eps[0], theta2 - eps[2])) / (2*h)
    ddg = np.zeros((n,n,n,n))
    for c in [0, 2]:
        for d in [0, 2]:
            if c == d:
                eps = np.zeros(4); eps[c] = h
                ddg[:,:,c,d] = (g_at(theta1+eps[0], theta2+eps[2]) - 2*g0 + g_at(theta1-eps[0], theta2-eps[2])) / h**2
            else:
                t1pp = theta1 + (h if c==0 else 0) + (h if d==0 else 0)
                t2pp = theta2 + (h if c==2 else 0) + (h if d==2 else 0)
                t1pm = theta1 + (h if c==0 else 0) - (h if d==0 else 0)
                t2pm = theta2 + (h if c==2 else 0) - (h if d==2 else 0)
                t1mp = theta1 - (h if c==0 else 0) + (h if d==0 else 0)
                t2mp = theta2 - (h if c==2 else 0) + (h if d==2 else 0)
                t1mm = theta1 - (h if c==0 else 0) - (h if d==0 else 0)
                t2mm = theta2 - (h if c==2 else 0) - (h if d==2 else 0)
                ddg[:,:,c,d] = (g_at(t1pp,t2pp) - g_at(t1pm,t2pm) - g_at(t1mp,t2mp) + g_at(t1mm,t2mm)) / (4*h**2)
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
    return R_0202 / denom


def formula_A(s_func, params, t1, t2, h=5e-4):
    """K = -1/2[d2^2(logL1) + d1^2(logL2)]  -- eq 4.8"""
    a1, a2, b1, b2, _, g1, g2, _ = params
    def L1_at(u,v):
        _,s1,_,s11,_,_ = seam_data(s_func, u, v)
        return a1 + b1*s1**2 + g1*s11
    def L2_at(u,v):
        _,_,s2,_,s22,_ = seam_data(s_func, u, v)
        return a2 + b2*s2**2 + g2*s22
    logL1 = np.log(L1_at(t1, t2))
    logL1_22 = (np.log(L1_at(t1, t2+h)) - 2*logL1 + np.log(L1_at(t1, t2-h))) / h**2
    logL2 = np.log(L2_at(t1, t2))
    logL2_11 = (np.log(L2_at(t1+h, t2)) - 2*logL2 + np.log(L2_at(t1-h, t2))) / h**2
    return -0.5*(logL1_22 + logL2_11)


def formula_B(s_func, params, t1, t2, h=5e-4):
    """K = [-L2/2*d2^2(logL1) - L1/2*d1^2(logL2)] / (L1*L2)  -- eq 4.6 with eta=0"""
    a1, a2, b1, b2, _, g1, g2, _ = params
    def L1_at(u,v):
        _,s1,_,s11,_,_ = seam_data(s_func, u, v)
        return a1 + b1*s1**2 + g1*s11
    def L2_at(u,v):
        _,_,s2,_,s22,_ = seam_data(s_func, u, v)
        return a2 + b2*s2**2 + g2*s22
    L1 = L1_at(t1, t2)
    L2 = L2_at(t1, t2)
    logL1 = np.log(L1)
    logL1_22 = (np.log(L1_at(t1, t2+h)) - 2*logL1 + np.log(L1_at(t1, t2-h))) / h**2
    logL2 = np.log(L2)
    logL2_11 = (np.log(L2_at(t1+h, t2)) - 2*logL2 + np.log(L2_at(t1-h, t2))) / h**2
    return (-L2/2*logL1_22 - L1/2*logL2_11) / (L1*L2)


# Test
s_func = lambda t1, t2: np.cos(t1) * np.cos(t2)
test_cases = [
    ("a+b",   (1.0, 1.0, 0.1, 0.1, 0, 0, 0, 0)),
    ("a+g",   (1.0, 1.0, 0, 0, 0, 0.1, 0.1, 0)),
    ("a+b+g", (1.0, 1.0, 0.1, 0.1, 0, 0.05, 0.05, 0)),
    ("large b", (1.0, 0.8, 0.3, 0.2, 0, 0, 0, 0)),
    ("large g", (1.0, 0.9, 0, 0, 0, 0.2, 0.15, 0)),
]

print("Comparing formula A (eq 4.8), formula B (eq 4.6), and direct Riemann")
print("=" * 80)

for label, params in test_cases:
    print(f"\ncos*cos / {label}:")
    for t1, t2 in [(1.0, 1.5), (0.8, 2.0), (np.pi/2, np.pi/2)]:
        KA = formula_A(s_func, params, t1, t2)
        KB = formula_B(s_func, params, t1, t2)
        KR = compute_K02_riemann(t1, t2, s_func, params)
        if KR is None:
            print(f"  ({t1:.2f},{t2:.2f}): not pos-def")
            continue
        match_A = "YES" if abs(KA-KR) < 0.001*max(abs(KR),1e-8) else "NO"
        match_B = "YES" if abs(KB-KR) < 0.001*max(abs(KR),1e-8) else "NO"
        print(f"  ({t1:.2f},{t2:.2f}): K_A={KA: .6f}  K_B={KB: .6f}  K_Riemann={KR: .6f}  "
              f"A={match_A}  B={match_B}")
