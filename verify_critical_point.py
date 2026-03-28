"""
Test FEEDBACK claim: at critical points of s, K_mix = 0 for ALL mixed planes.

For sigma = (3, 2, 1), critical points are x = y = e_i.
At x = y = e_3, s = sigma_3 = 1.
The tangent coupling matrix is diag(sigma_1 - sigma_3, sigma_2 - sigma_3) = diag(2, 1),
which is NONZERO.  If the lemma were correct, ALL mixed curvatures would vanish here.

We check by computing K_mix for several mixed planes at exact critical points.
"""

import numpy as np

def seam(sigma, x, y):
    return sigma[0]*x[0]*y[0] + sigma[1]*x[1]*y[1] + sigma[2]*x[2]*y[2]

def make_metric(sigma, gamma, theta1, phi1, theta2, phi2):
    """Compute the 4x4 metric g_{IJ} at a point on S^2 x S^2 via finite differences."""
    h = 1e-5
    coords = np.array([theta1, phi1, theta2, phi2])

    def point(th1, ph1, th2, ph2):
        x = np.array([np.sin(th1)*np.cos(ph1), np.sin(th1)*np.sin(ph1), np.cos(th1)])
        y = np.array([np.sin(th2)*np.cos(ph2), np.sin(th2)*np.sin(ph2), np.cos(th2)])
        return x, y

    def s_val(c):
        x, y = point(*c)
        return seam(sigma, x, y)

    def h_metric(c):
        """Background product metric in (theta1, phi1, theta2, phi2) coords."""
        th1, ph1, th2, ph2 = c
        g = np.zeros((4,4))
        g[0,0] = 1.0
        g[1,1] = np.sin(th1)**2
        g[2,2] = 1.0
        g[3,3] = np.sin(th2)**2
        return g

    # Full metric via FD of s
    # g_IJ = (1 - gamma*s) h_IJ + gamma * nabla^2_IJ s
    # where nabla^2 s is the covariant Hessian of s on S^2 x S^2

    # Compute s and its first/second derivatives numerically
    s0 = s_val(coords)

    # First derivatives
    ds = np.zeros(4)
    for i in range(4):
        cp = coords.copy(); cm = coords.copy()
        cp[i] += h; cm[i] -= h
        ds[i] = (s_val(cp) - s_val(cm)) / (2*h)

    # Second derivatives (coordinate)
    d2s = np.zeros((4,4))
    for i in range(4):
        for j in range(i, 4):
            if i == j:
                cp = coords.copy(); cm = coords.copy()
                cp[i] += h; cm[i] -= h
                d2s[i,i] = (s_val(cp) - 2*s0 + s_val(cm)) / h**2
            else:
                cpp = coords.copy(); cpm = coords.copy()
                cmp = coords.copy(); cmm = coords.copy()
                cpp[i] += h; cpp[j] += h
                cpm[i] += h; cpm[j] -= h
                cmp[i] -= h; cmp[j] += h
                cmm[i] -= h; cmm[j] -= h
                d2s[i,j] = (s_val(cpp) - s_val(cpm) - s_val(cmp) + s_val(cmm)) / (4*h**2)
                d2s[j,i] = d2s[i,j]

    # Background Christoffel symbols for S^2 x S^2
    # S^2 in (theta, phi): Gamma^theta_{phi,phi} = -sin(theta)cos(theta)
    #                       Gamma^phi_{theta,phi} = cos(theta)/sin(theta)
    th1, ph1, th2, ph2 = coords
    Gamma_bg = np.zeros((4,4,4))
    # First S^2
    Gamma_bg[0,1,1] = -np.sin(th1)*np.cos(th1)
    Gamma_bg[1,0,1] = np.cos(th1)/np.sin(th1)
    Gamma_bg[1,1,0] = np.cos(th1)/np.sin(th1)
    # Second S^2
    Gamma_bg[2,3,3] = -np.sin(th2)*np.cos(th2)
    Gamma_bg[3,2,3] = np.cos(th2)/np.sin(th2)
    Gamma_bg[3,3,2] = np.cos(th2)/np.sin(th2)

    # Covariant Hessian = d2s - Gamma^k ds_k
    cov_hess = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            cov_hess[i,j] = d2s[i,j]
            for k in range(4):
                cov_hess[i,j] -= Gamma_bg[k,i,j] * ds[k]

    h_bg = h_metric(coords)
    g = (1 - gamma*s0) * h_bg + gamma * cov_hess

    return g, s0, ds


def riemann_fd(sigma, gamma, coords, h=1e-4):
    """Compute full 4x4x4x4 Riemann tensor via FD of Christoffel symbols."""

    def point(c):
        th1, ph1, th2, ph2 = c
        x = np.array([np.sin(th1)*np.cos(ph1), np.sin(th1)*np.sin(ph1), np.cos(th1)])
        y = np.array([np.sin(th2)*np.cos(ph2), np.sin(th2)*np.sin(ph2), np.cos(th2)])
        return x, y

    def s_val(c):
        x, y = point(c)
        return seam(sigma, x, y)

    def h_metric(c):
        th1, ph1, th2, ph2 = c
        g = np.zeros((4,4))
        g[0,0] = 1.0
        g[1,1] = np.sin(th1)**2
        g[2,2] = 1.0
        g[3,3] = np.sin(th2)**2
        return g

    def full_metric(c):
        th1, ph1, th2, ph2 = c
        s0 = s_val(c)
        ds = np.zeros(4)
        for i in range(4):
            cp = c.copy(); cm = c.copy()
            cp[i] += h/10; cm[i] -= h/10
            ds[i] = (s_val(cp) - s_val(cm)) / (2*h/10)
        d2s = np.zeros((4,4))
        for i in range(4):
            for j in range(i, 4):
                if i == j:
                    cp = c.copy(); cm = c.copy()
                    cp[i] += h/10; cm[i] -= h/10
                    d2s[i,i] = (s_val(cp) - 2*s0 + s_val(cm)) / (h/10)**2
                else:
                    cpp = c.copy(); cpm = c.copy(); cmp = c.copy(); cmm = c.copy()
                    cpp[i] += h/10; cpp[j] += h/10
                    cpm[i] += h/10; cpm[j] -= h/10
                    cmp[i] -= h/10; cmp[j] += h/10
                    cmm[i] -= h/10; cmm[j] -= h/10
                    d2s[i,j] = (s_val(cpp) - s_val(cpm) - s_val(cmp) + s_val(cmm)) / (4*(h/10)**2)
                    d2s[j,i] = d2s[i,j]

        Gamma_bg = np.zeros((4,4,4))
        Gamma_bg[0,1,1] = -np.sin(th1)*np.cos(th1)
        Gamma_bg[1,0,1] = np.cos(th1)/np.sin(th1)
        Gamma_bg[1,1,0] = np.cos(th1)/np.sin(th1)
        Gamma_bg[2,3,3] = -np.sin(th2)*np.cos(th2)
        Gamma_bg[3,2,3] = np.cos(th2)/np.sin(th2)
        Gamma_bg[3,3,2] = np.cos(th2)/np.sin(th2)

        cov_hess = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                cov_hess[i,j] = d2s[i,j]
                for k in range(4):
                    cov_hess[i,j] -= Gamma_bg[k,i,j] * ds[k]

        hbg = h_metric(c)
        return (1 - gamma*s0) * hbg + gamma * cov_hess

    # Christoffel symbols via FD of metric
    g0 = full_metric(coords)
    ginv = np.linalg.inv(g0)

    dg = np.zeros((4,4,4))  # dg[k,i,j] = dg_{ij}/dx^k
    for k in range(4):
        cp = coords.copy(); cm = coords.copy()
        cp[k] += h; cm[k] -= h
        dg[k] = (full_metric(cp) - full_metric(cm)) / (2*h)

    Gamma = np.zeros((4,4,4))  # Gamma^i_{jk}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    Gamma[i,j,k] += 0.5 * ginv[i,l] * (dg[j,l,k] + dg[k,l,j] - dg[l,j,k])

    # Riemann via FD of Christoffel
    dGamma = np.zeros((4,4,4,4))  # dGamma[m, i,j,k] = d Gamma^i_{jk} / dx^m
    for m in range(4):
        cp = coords.copy(); cm = coords.copy()
        cp[m] += h; cm[m] -= h

        def get_gamma(c):
            gm = full_metric(c)
            gminv = np.linalg.inv(gm)
            dgm = np.zeros((4,4,4))
            for kk in range(4):
                cpp2 = c.copy(); cmm2 = c.copy()
                cpp2[kk] += h; cmm2[kk] -= h
                dgm[kk] = (full_metric(cpp2) - full_metric(cmm2)) / (2*h)
            Gm = np.zeros((4,4,4))
            for ii in range(4):
                for jj in range(4):
                    for kk in range(4):
                        for ll in range(4):
                            Gm[ii,jj,kk] += 0.5 * gminv[ii,ll] * (dgm[jj,ll,kk] + dgm[kk,ll,jj] - dgm[ll,jj,kk])
            return Gm

        dGamma[m] = (get_gamma(cp) - get_gamma(cm)) / (2*h)

    R = np.zeros((4,4,4,4))  # R^i_{jkl}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    R[i,j,k,l] = dGamma[k,i,j,l] - dGamma[l,i,j,k]
                    for m in range(4):
                        R[i,j,k,l] += Gamma[i,k,m]*Gamma[m,j,l] - Gamma[i,l,m]*Gamma[m,j,k]

    # Lower first index
    Rdown = np.zeros((4,4,4,4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    for m in range(4):
                        Rdown[m,j,k,l] += g0[m,i] * R[i,j,k,l]

    return Rdown, g0


def sectional_curvature(Rdown, g, X, V):
    """K(X,V) = R(X,V,X,V) / (|X|^2|V|^2 - <X,V>^2)"""
    num = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    num += Rdown[i,j,k,l] * X[i] * V[j] * X[k] * V[l]
    gXX = sum(g[i,j]*X[i]*X[j] for i in range(4) for j in range(4))
    gVV = sum(g[i,j]*V[i]*V[j] for i in range(4) for j in range(4))
    gXV = sum(g[i,j]*X[i]*V[j] for i in range(4) for j in range(4))
    denom = gXX * gVV - gXV**2
    if abs(denom) < 1e-20:
        return float('nan')
    return num / denom


print("="*70)
print("Test: does K_mix = 0 for ALL mixed planes at critical points of s?")
print("="*70)

gamma = 0.3

# Critical points for sigma = (3, 2, 1):
# x = y = e_i  =>  (theta, phi) = (pi/2, 0), (pi/2, pi/2), (epsilon, 0)
# We use small epsilon instead of 0 to avoid coordinate singularity

critical_points = [
    # x=y=e1: theta=pi/2, phi=0
    ("e1", np.array([np.pi/2, 0.0, np.pi/2, 0.0])),
    # x=y=e2: theta=pi/2, phi=pi/2
    ("e2", np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2])),
    # x=y=e3: theta=epsilon, phi=0 (near north pole)
    ("e3", np.array([0.05, 0.0, 0.05, 0.0])),
]

sigma = np.array([3.0, 2.0, 1.0])

for name, coords in critical_points:
    print(f"\n--- Critical point x=y={name}, sigma={sigma} ---")

    # Check gradient is near zero
    g_met, s0, ds = make_metric(sigma, gamma, *coords)
    print(f"  s = {s0:.6f}, |grad s| = {np.linalg.norm(ds):.2e}")

    # Compute Riemann tensor
    Rdown, g_full = riemann_fd(sigma, gamma, coords)

    # Test several mixed planes
    # Mixed plane: X has components in (theta1, phi1), V has components in (theta2, phi2)
    mixed_dirs = [
        ("e_th1 ^ e_th2", np.array([1,0,0,0], dtype=float), np.array([0,0,1,0], dtype=float)),
        ("e_th1 ^ e_ph2", np.array([1,0,0,0], dtype=float), np.array([0,0,0,1], dtype=float)),
        ("e_ph1 ^ e_th2", np.array([0,1,0,0], dtype=float), np.array([0,0,1,0], dtype=float)),
        ("e_ph1 ^ e_ph2", np.array([0,1,0,0], dtype=float), np.array([0,0,0,1], dtype=float)),
        ("mix1",
         np.array([1, 0.5, 0, 0], dtype=float),
         np.array([0, 0, 0.7, 1], dtype=float)),
        ("mix2",
         np.array([0.3, 1, 0, 0], dtype=float),
         np.array([0, 0, 1, 0.4], dtype=float)),
    ]

    any_nonzero = False
    for label, X, V in mixed_dirs:
        K = sectional_curvature(Rdown, g_full, X, V)
        marker = "" if abs(K) < 1e-4 else " *** NONZERO!"
        if abs(K) >= 1e-4:
            any_nonzero = True
        print(f"  K({label}) = {K:+.8e}{marker}")

    if any_nonzero:
        print(f"  ==> NOT all K_mix = 0 at this critical point!")
    else:
        print(f"  ==> All K_mix ≈ 0 at this critical point")

# Also test isotropic case for comparison
print(f"\n{'='*70}")
print("Isotropic control: sigma = (1,1,1), x=y=e1")
print("(Here s=±1 makes metric a product => ALL K_mix should be 0)")
print("="*70)

sigma_iso = np.array([1.0, 1.0, 1.0])
coords_iso = np.array([np.pi/2, 0.0, np.pi/2, 0.0])
g_met, s0, ds = make_metric(sigma_iso, gamma, *coords_iso)
print(f"  s = {s0:.6f}, |grad s| = {np.linalg.norm(ds):.2e}")
Rdown, g_full = riemann_fd(sigma_iso, gamma, coords_iso)
for label, X, V in mixed_dirs:
    K = sectional_curvature(Rdown, g_full, X, V)
    print(f"  K({label}) = {K:+.8e}")
