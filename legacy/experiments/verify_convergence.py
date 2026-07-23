"""
Convergence test: K(e_ph1, e_ph2) at x=y=e1 for isotropic seam.
This plane is tangent to the totally geodesic torus N_P (P = span(e1,e2)).
The paper proves K = 0 there. Check if FD converges to 0 or to a nonzero value.

Also test along the torus at a point where (x,y) are NOT equal, to
distinguish critical-point effects from the general torus result.
"""

import numpy as np

def embed(theta, phi):
    return np.array([np.sin(theta)*np.cos(phi),
                     np.sin(theta)*np.sin(phi),
                     np.cos(theta)])

def seam_val(sigma, c):
    th1, ph1, th2, ph2 = c
    x = embed(th1, ph1)
    y = embed(th2, ph2)
    return sigma[0]*x[0]*y[0] + sigma[1]*x[1]*y[1] + sigma[2]*x[2]*y[2]

def metric_at(sigma, gamma, c, h=1e-6):
    th1, ph1, th2, ph2 = c
    s0 = seam_val(sigma, c)
    hbg = np.diag([1.0, np.sin(th1)**2, 1.0, np.sin(th2)**2])
    ds = np.zeros(4)
    for i in range(4):
        cp = c.copy(); cm = c.copy()
        cp[i] += h; cm[i] -= h
        ds[i] = (seam_val(sigma, cp) - seam_val(sigma, cm)) / (2*h)
    d2s = np.zeros((4, 4))
    for i in range(4):
        for j in range(i, 4):
            if i == j:
                cp = c.copy(); cm = c.copy()
                cp[i] += h; cm[i] -= h
                d2s[i,i] = (seam_val(sigma, cp) - 2*s0 + seam_val(sigma, cm)) / h**2
            else:
                pp = c.copy(); pm = c.copy(); mp = c.copy(); mm = c.copy()
                pp[i] += h; pp[j] += h; pm[i] += h; pm[j] -= h
                mp[i] -= h; mp[j] += h; mm[i] -= h; mm[j] -= h
                d2s[i,j] = (seam_val(sigma, pp) - seam_val(sigma, pm)
                            - seam_val(sigma, mp) + seam_val(sigma, mm)) / (4*h**2)
                d2s[j,i] = d2s[i,j]
    G = np.zeros((4, 4, 4))
    st1 = np.sin(th1); ct1 = np.cos(th1)
    st2 = np.sin(th2); ct2 = np.cos(th2)
    if abs(st1) > 1e-10:
        G[0,1,1] = -st1*ct1; G[1,0,1] = G[1,1,0] = ct1/st1
    if abs(st2) > 1e-10:
        G[2,3,3] = -st2*ct2; G[3,2,3] = G[3,3,2] = ct2/st2
    cov = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            cov[i,j] = d2s[i,j] - sum(G[k,i,j]*ds[k] for k in range(4))
    return (1-gamma*s0)*hbg + gamma*cov

def K_mix(sigma, gamma, c, X, V, h_outer):
    h_inner = h_outer * 0.1
    g0 = metric_at(sigma, gamma, c, h=h_inner)
    ginv = np.linalg.inv(g0)

    def get_gamma(cc):
        gg = metric_at(sigma, gamma, cc, h=h_inner)
        gi = np.linalg.inv(gg)
        ddg = np.zeros((4,4,4))
        for k in range(4):
            cp = cc.copy(); cm = cc.copy()
            cp[k] += h_outer; cm[k] -= h_outer
            ddg[k] = (metric_at(sigma, gamma, cp, h=h_inner)
                      - metric_at(sigma, gamma, cm, h=h_inner)) / (2*h_outer)
        GG = np.zeros((4,4,4))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    GG[i,j,k] = 0.5*sum(gi[i,l]*(ddg[j,l,k]+ddg[k,l,j]-ddg[l,j,k]) for l in range(4))
        return GG

    Gam0 = get_gamma(c)
    dGam = np.zeros((4,4,4,4))
    for m in range(4):
        cp = c.copy(); cm = c.copy()
        cp[m] += h_outer; cm[m] -= h_outer
        dGam[m] = (get_gamma(cp) - get_gamma(cm)) / (2*h_outer)

    R = np.zeros((4,4,4,4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    R[i,j,k,l] = dGam[k,i,j,l] - dGam[l,i,j,k]
                    for m_ in range(4):
                        R[i,j,k,l] += Gam0[i,k,m_]*Gam0[m_,j,l] - Gam0[i,l,m_]*Gam0[m_,j,k]

    Rd = np.einsum('im,mjkl->ijkl', g0, R)
    num = np.einsum('ijkl,i,j,k,l', Rd, X, V, V, X)
    gXX = np.einsum('ij,i,j', g0, X, X)
    gVV = np.einsum('ij,i,j', g0, V, V)
    gXV = np.einsum('ij,i,j', g0, X, V)
    denom = gXX * gVV - gXV**2
    return num / denom

sigma = np.array([1.0, 1.0, 1.0])
gamma = 0.3

# Test 1: At x=y=e1 (critical point, s=1) — torus plane e_ph1 ^ e_ph2
print("Test 1: x=y=e1 (s=1), plane tangent to totally geodesic torus")
print("Convergence of K as h -> 0:")
c1 = np.array([np.pi/2, 0.0, np.pi/2, 0.0])
X_ph = np.array([0, 1, 0, 0.])
V_ph = np.array([0, 0, 0, 1.])
for h in [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]:
    K = K_mix(sigma, gamma, c1, X_ph, V_ph, h)
    print(f"  h = {h:.0e}: K = {K:+.10e}")

# Test 2: Torus point with x != y (NOT a critical point)
# On N_P with P=span(e1,e2): x = (cos(a1), sin(a1), 0), y = (cos(a2), sin(a2), 0)
# In spherical coords: theta=pi/2 for both, phi1=a1, phi2=a2
print("\nTest 2: x=(cos0.5,sin0.5,0), y=(cos1.2,sin1.2,0) on torus N_P")
print("s = cos(phi1-phi2) = cos(-0.7)")
c2 = np.array([np.pi/2, 0.5, np.pi/2, 1.2])
s2 = seam_val(sigma, c2)
print(f"s = {s2:.6f}, cos(-0.7) = {np.cos(-0.7):.6f}")
print("Convergence of K(tangent to torus = e_ph1 ^ e_ph2):")
for h in [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]:
    K = K_mix(sigma, gamma, c2, X_ph, V_ph, h)
    print(f"  h = {h:.0e}: K = {K:+.10e}")

# Test 3: Non-torus plane at same point (should be > 0)
print("\nTest 3: Same torus point, NON-torus plane e_th1 ^ e_th2:")
X_th = np.array([1, 0, 0, 0.])
V_th = np.array([0, 0, 1, 0.])
for h in [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]:
    K = K_mix(sigma, gamma, c2, X_th, V_th, h)
    print(f"  h = {h:.0e}: K = {K:+.10e}")

# Test 4: Non-isotropic sigma at critical point
print("\nTest 4: sigma=(3,2,1), x=y=e1, e_th1 ^ e_th2 plane:")
sigma2 = np.array([3.0, 2.0, 1.0])
for h in [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]:
    K = K_mix(sigma2, gamma, c1, X_th, V_th, h)
    print(f"  h = {h:.0e}: K = {K:+.10e}")
