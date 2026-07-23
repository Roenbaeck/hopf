"""
Clean test: K_mix at critical points of s for non-isotropic seams.

FEEDBACK claims: at critical points of s, ALL mixed curvatures are zero.
We test this directly using the well-tested FD approach from the paper's scripts.

Method: compute K_mix for various mixed planes at critical points where grad(s) = 0.
"""

import numpy as np
from itertools import product as iprod

def embed(theta, phi):
    return np.array([np.sin(theta)*np.cos(phi),
                     np.sin(theta)*np.sin(phi),
                     np.cos(theta)])

def seam_val(sigma, theta1, phi1, theta2, phi2):
    x = embed(theta1, phi1)
    y = embed(theta2, phi2)
    return sigma[0]*x[0]*y[0] + sigma[1]*x[1]*y[1] + sigma[2]*x[2]*y[2]

def metric_at(sigma, gamma, c, h=1e-6):
    """Full 4x4 metric g_IJ at point c = (th1, ph1, th2, ph2)."""
    th1, ph1, th2, ph2 = c
    s0 = seam_val(sigma, *c)

    # Background metric
    hbg = np.diag([1.0, np.sin(th1)**2, 1.0, np.sin(th2)**2])

    # Covariant Hessian via FD with background Christoffel correction
    ds = np.zeros(4)
    for i in range(4):
        cp = c.copy(); cm = c.copy()
        cp[i] += h; cm[i] -= h
        ds[i] = (seam_val(sigma, *cp) - seam_val(sigma, *cm)) / (2*h)

    d2s = np.zeros((4, 4))
    for i in range(4):
        for j in range(i, 4):
            if i == j:
                cp = c.copy(); cm = c.copy()
                cp[i] += h; cm[i] -= h
                d2s[i, i] = (seam_val(sigma, *cp) - 2*s0 + seam_val(sigma, *cm)) / h**2
            else:
                pp = c.copy(); pm = c.copy(); mp = c.copy(); mm = c.copy()
                pp[i] += h; pp[j] += h
                pm[i] += h; pm[j] -= h
                mp[i] -= h; mp[j] += h
                mm[i] -= h; mm[j] -= h
                d2s[i, j] = (seam_val(sigma, *pp) - seam_val(sigma, *pm)
                             - seam_val(sigma, *mp) + seam_val(sigma, *mm)) / (4*h**2)
                d2s[j, i] = d2s[i, j]

    # Background Christoffel symbols
    G = np.zeros((4, 4, 4))
    G[0, 1, 1] = -np.sin(th1)*np.cos(th1)
    G[1, 0, 1] = G[1, 1, 0] = np.cos(th1)/np.sin(th1) if np.sin(th1) > 1e-10 else 0
    G[2, 3, 3] = -np.sin(th2)*np.cos(th2)
    G[3, 2, 3] = G[3, 3, 2] = np.cos(th2)/np.sin(th2) if np.sin(th2) > 1e-10 else 0

    cov = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            cov[i, j] = d2s[i, j] - sum(G[k, i, j]*ds[k] for k in range(4))

    return (1 - gamma*s0) * hbg + gamma * cov


def sectional_curvature(sigma, gamma, c, X, V, h=1e-4):
    """
    Compute K(X,V) via FD of Christoffel symbols of FD of metric.
    Uses R_{XVVX} / denom convention.
    """
    g0 = metric_at(sigma, gamma, c, h=h*0.1)
    ginv = np.linalg.inv(g0)

    # dg/dx^k
    dg = np.zeros((4, 4, 4))
    for k in range(4):
        cp = c.copy(); cm = c.copy()
        cp[k] += h; cm[k] -= h
        dg[k] = (metric_at(sigma, gamma, cp, h=h*0.1)
                 - metric_at(sigma, gamma, cm, h=h*0.1)) / (2*h)

    # Gamma^i_{jk}
    Gam = np.zeros((4, 4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                Gam[i, j, k] = 0.5 * sum(
                    ginv[i, l]*(dg[j, l, k] + dg[k, l, j] - dg[l, j, k])
                    for l in range(4))

    # dGamma/dx^m via FD
    def gamma_at(cc):
        gg = metric_at(sigma, gamma, cc, h=h*0.1)
        gi = np.linalg.inv(gg)
        ddg = np.zeros((4, 4, 4))
        for kk in range(4):
            cp2 = cc.copy(); cm2 = cc.copy()
            cp2[kk] += h; cm2[kk] -= h
            ddg[kk] = (metric_at(sigma, gamma, cp2, h=h*0.1)
                       - metric_at(sigma, gamma, cm2, h=h*0.1)) / (2*h)
        GG = np.zeros((4, 4, 4))
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    GG[ii, jj, kk] = 0.5*sum(
                        gi[ii, ll]*(ddg[jj, ll, kk]+ddg[kk, ll, jj]-ddg[ll, jj, kk])
                        for ll in range(4))
        return GG

    dGam = np.zeros((4, 4, 4, 4))
    for m in range(4):
        cp = c.copy(); cm = c.copy()
        cp[m] += h; cm[m] -= h
        dGam[m] = (gamma_at(cp) - gamma_at(cm)) / (2*h)

    # R^i_{jkl}
    R = np.zeros((4, 4, 4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    R[i, j, k, l] = dGam[k, i, j, l] - dGam[l, i, j, k]
                    for m in range(4):
                        R[i, j, k, l] += Gam[i, k, m]*Gam[m, j, l] - Gam[i, l, m]*Gam[m, j, k]

    # R_{ijkl} = g_{im} R^m_{jkl}
    Rd = np.einsum('im,mjkl->ijkl', g0, R)

    # K = R(X,V,V,X) / (|X|^2|V|^2 - <X,V>^2)
    num = np.einsum('ijkl,i,j,k,l', Rd, X, V, V, X)
    gXX = np.einsum('ij,i,j', g0, X, X)
    gVV = np.einsum('ij,i,j', g0, V, V)
    gXV = np.einsum('ij,i,j', g0, X, V)
    denom = gXX * gVV - gXV**2
    if abs(denom) < 1e-20:
        return float('nan')
    return num / denom


# ========================================================
# Test at critical points for sigma = (3, 2, 1)
# ========================================================
gamma = 0.3
sigma = np.array([3.0, 2.0, 1.0])

print("="*70)
print("SANITY CHECK: gamma=0 round product S^2 x S^2")
print("All mixed K should be 0")
print("="*70)
c_test = np.array([1.0, 0.5, 1.2, 0.8])
X = np.array([1.0, 0.0, 0.0, 0.0])
V = np.array([0.0, 0.0, 1.0, 0.0])
K_san = sectional_curvature(np.array([1,1,1.]), 0.0, c_test, X, V)
print(f"  K(e_th1, e_th2) at generic point, gamma=0: {K_san:+.6e}")
X2 = np.array([0.0, 1.0, 0.0, 0.0])
V2 = np.array([0.0, 0.0, 0.0, 1.0])
K_san2 = sectional_curvature(np.array([1,1,1.]), 0.0, c_test, X2, V2)
print(f"  K(e_ph1, e_ph2) at generic point, gamma=0: {K_san2:+.6e}")

print()
print("="*70)
print(f"K_mix at critical points of s, sigma={sigma}, gamma={gamma}")
print("FEEDBACK claims: ALL K_mix = 0 at critical points")
print("="*70)

# Critical point: x = y = e1 => theta1=theta2=pi/2, phi1=phi2=0
# s = sigma[0] = 3.0
crit_e1 = np.array([np.pi/2, 0.0, np.pi/2, 0.0])
# Critical point: x = y = e2 => theta1=theta2=pi/2, phi1=phi2=pi/2
# s = sigma[1] = 2.0
crit_e2 = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2])

test_cases = [
    ("x=y=e1 (s=3)", crit_e1),
    ("x=y=e2 (s=2)", crit_e2),
]

# Mixed plane basis vectors
mixed_planes = [
    ("e_th1 ^ e_th2", np.array([1,0,0,0.]), np.array([0,0,1,0.])),
    ("e_th1 ^ e_ph2", np.array([1,0,0,0.]), np.array([0,0,0,1.])),
    ("e_ph1 ^ e_th2", np.array([0,1,0,0.]), np.array([0,0,1,0.])),
    ("e_ph1 ^ e_ph2", np.array([0,1,0,0.]), np.array([0,0,0,1.])),
]

for name, c in test_cases:
    print(f"\n--- Critical point: {name} ---")
    s_val_here = seam_val(sigma, *c)
    print(f"  s = {s_val_here:.6f}, conf factor = {1-gamma*s_val_here:.4f}")

    # Check grad s ≈ 0
    h_fd = 1e-6
    ds = np.zeros(4)
    for i in range(4):
        cp = c.copy(); cm = c.copy()
        cp[i] += h_fd; cm[i] -= h_fd
        ds[i] = (seam_val(sigma,*cp) - seam_val(sigma,*cm)) / (2*h_fd)
    print(f"  |grad s| = {np.linalg.norm(ds):.2e}")

    for label, X, V in mixed_planes:
        K = sectional_curvature(sigma, gamma, c, X, V, h=5e-5)
        marker = "" if abs(K) < 0.01 else " *** NONZERO"
        print(f"  K({label}) = {K:+.8e}{marker}")

# Also test the isotropic case at its critical point for reference
print()
print("="*70)
print("CONTROL: Isotropic sigma=(1,1,1), x=y=e1 (s=1), gamma=0.3")
print("="*70)
sigma_iso = np.array([1.0, 1.0, 1.0])
c_iso = np.array([np.pi/2, 0.0, np.pi/2, 0.0])
s_iso = seam_val(sigma_iso, *c_iso)
print(f"  s = {s_iso:.6f}, conf factor = {1-gamma*s_iso:.4f}")
for label, X, V in mixed_planes:
    K = sectional_curvature(sigma_iso, gamma, c_iso, X, V, h=5e-5)
    marker = "" if abs(K) < 0.01 else " *** NONZERO"
    print(f"  K({label}) = {K:+.8e}{marker}")
