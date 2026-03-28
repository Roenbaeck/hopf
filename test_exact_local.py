#!/usr/bin/env python3
"""
Ultra-precise single-point curvature computation for x1x2 seam.
Uses local fine FD around a specific point (h ~ 1e-6) to get
essentially exact curvature values. No global grid needed.
"""

import numpy as np

def make_seam_x1x2():
    """s = x1*x2 = sin(th1)*cos(ph1)*sin(th2)*cos(ph2)"""
    def seam_full(t1, p1, t2, p2):
        s = np.sin(t1)*np.cos(p1)*np.sin(t2)*np.cos(p2)
        return s
    return seam_full

def make_seam_z1z2():
    """s = z1*z2 = cos(th1)*cos(th2)"""
    def seam_full(t1, p1, t2, p2):
        return np.cos(t1)*np.cos(t2)
    return seam_full

def make_l1_seam(coeffs):
    """General l=1 seam: s = sum c_ij f_i(unit1) g_j(unit2)"""
    c = np.array(coeffs).reshape(3,3)
    def seam_full(t1, p1, t2, p2):
        f = [np.sin(t1)*np.cos(p1), np.sin(t1)*np.sin(p1), np.cos(t1)]
        g = [np.sin(t2)*np.cos(p2), np.sin(t2)*np.sin(p2), np.cos(t2)]
        s = sum(c[i,j]*f[i]*g[j] for i in range(3) for j in range(3))
        return s
    return seam_full

def compute_metric(seam_func, gamma, t1, p1, t2, p2):
    """Compute the 4x4 metric g = h + gamma*Hess(s) at a single point.
    Uses ultra-fine FD for the covariant Hessian."""
    h = 1e-6  # very small step for FD
    coords = np.array([t1, p1, t2, p2])
    st1, ct1 = np.sin(t1), np.cos(t1)
    st2, ct2 = np.sin(t2), np.cos(t2)
    
    s0 = seam_func(t1, p1, t2, p2)
    
    # First derivatives (central FD)
    ds = np.zeros(4)
    for i in range(4):
        c_plus = coords.copy(); c_plus[i] += h
        c_minus = coords.copy(); c_minus[i] -= h
        ds[i] = (seam_func(*c_plus) - seam_func(*c_minus)) / (2*h)
    
    # Second derivatives (central FD)
    d2s = np.zeros((4,4))
    for i in range(4):
        for j in range(i, 4):
            if i == j:
                c_plus = coords.copy(); c_plus[i] += h
                c_minus = coords.copy(); c_minus[i] -= h
                d2s[i,i] = (seam_func(*c_plus) - 2*s0 + seam_func(*c_minus)) / h**2
            else:
                c_pp = coords.copy(); c_pp[i] += h; c_pp[j] += h
                c_pm = coords.copy(); c_pm[i] += h; c_pm[j] -= h
                c_mp = coords.copy(); c_mp[i] -= h; c_mp[j] += h
                c_mm = coords.copy(); c_mm[i] -= h; c_mm[j] -= h
                d2s[i,j] = (seam_func(*c_pp) - seam_func(*c_pm) - seam_func(*c_mp) + seam_func(*c_mm)) / (4*h**2)
                d2s[j,i] = d2s[i,j]
    
    # Covariant Hessian
    H = np.zeros((4,4))
    H[0,0] = d2s[0,0]
    H[0,1] = d2s[0,1] - (ct1/st1)*ds[1]
    H[0,2] = d2s[0,2]
    H[0,3] = d2s[0,3]
    H[1,0] = H[0,1]
    H[1,1] = d2s[1,1] + st1*ct1*ds[0]
    H[1,2] = d2s[1,2]
    H[1,3] = d2s[1,3]
    H[2,0] = H[0,2]; H[2,1] = H[1,2]
    H[2,2] = d2s[2,2]
    H[2,3] = d2s[2,3] - (ct2/st2)*ds[3]
    H[3,0] = H[0,3]; H[3,1] = H[1,3]; H[3,2] = H[2,3]
    H[3,3] = d2s[3,3] + st2*ct2*ds[2]
    
    # Full metric
    G = gamma * H
    G[0,0] += 1.0; G[1,1] += st1**2; G[2,2] += 1.0; G[3,3] += st2**2
    
    return G

def compute_curvature_at_point(seam_func, gamma, point):
    """Compute all R_{ijkl} at a single point using ultra-fine local FD."""
    t1, p1, t2, p2 = point
    h_fd = 1e-5  # step for differentiating metric/Christoffels
    
    # Compute metric at the central point and at +/- h in each direction
    coords = np.array(point)
    
    def metric_at(pt):
        return compute_metric(seam_func, gamma, *pt)
    
    G0 = metric_at(coords)
    Ginv0 = np.linalg.inv(G0)
    
    # dG/dx^k by FD
    dG = np.zeros((4,4,4))
    for k in range(4):
        cp = coords.copy(); cp[k] += h_fd
        cm = coords.copy(); cm[k] -= h_fd
        Gp = metric_at(cp)
        Gm = metric_at(cm)
        dG[:,:,k] = (Gp - Gm) / (2*h_fd)
    
    # Christoffel symbols at center
    Chr0 = np.zeros((4,4,4))
    for m in range(4):
        for i in range(4):
            for j in range(4):
                val = 0
                for k in range(4):
                    val += Ginv0[m,k] * (dG[k,i,j] + dG[k,j,i] - dG[i,j,k])
                Chr0[m,i,j] = 0.5 * val
    
    # dChr/dx^l by FD
    def chr_at(pt):
        """Compute Christoffel symbols at a nearby point."""
        Gl = metric_at(pt)
        Ginvl = np.linalg.inv(Gl)
        # Need dG at that point too - use FD
        dGl = np.zeros((4,4,4))
        for k in range(4):
            cp = pt.copy(); cp[k] += h_fd
            cm = pt.copy(); cm[k] -= h_fd
            dGl[:,:,k] = (metric_at(cp) - metric_at(cm)) / (2*h_fd)
        Chrl = np.zeros((4,4,4))
        for m in range(4):
            for i in range(4):
                for j in range(4):
                    val = 0
                    for k in range(4):
                        val += Ginvl[m,k] * (dGl[k,i,j] + dGl[k,j,i] - dGl[i,j,k])
                    Chrl[m,i,j] = 0.5 * val
        return Chrl
    
    dChr = np.zeros((4,4,4,4))
    for l in range(4):
        cp = coords.copy(); cp[l] += h_fd
        cm = coords.copy(); cm[l] -= h_fd
        dChr[:,:,:,l] = (chr_at(cp) - chr_at(cm)) / (2*h_fd)
    
    # R_{ijkl} = g_{im} R^m_{jkl}
    # R^m_{jkl} = dChr^m_{jl,k} - dChr^m_{jk,l} + Chr^m_{kp}Chr^p_{jl} - Chr^m_{lp}Chr^p_{jk}
    def compute_R(i, j, k, l):
        val = 0
        for m in range(4):
            Rm = dChr[m,j,l,k] - dChr[m,j,k,l]
            for p in range(4):
                Rm += Chr0[m,k,p]*Chr0[p,j,l] - Chr0[m,l,p]*Chr0[p,j,k]
            val += G0[i,m] * Rm
        return val
    
    return compute_R, G0

# Test
gamma = 0.1

test_points = [
    (np.pi/4, np.pi/4, np.pi/4, np.pi/4),
    (np.pi/3, np.pi/6, np.pi/4, np.pi/3),
    (np.pi/2, np.pi/4, np.pi/2, np.pi/4),
    (np.pi/6, np.pi/3, np.pi/3, np.pi/6),
    (1.0, 0.5, 1.2, 0.8),
    (0.8, 1.5, 0.6, 2.0),
]

for seam_name, seam_func_maker in [("z1z2", make_seam_z1z2), ("x1x2", make_seam_x1x2)]:
    print(f"\n{'='*60}")
    print(f"Seam: {seam_name}, gamma={gamma}")
    print(f"{'='*60}")
    
    sf = seam_func_maker()
    for pt in test_points:
        R_func, G0 = compute_curvature_at_point(sf, gamma, pt)
        
        # All mixed R_{aAlphaBBeta}
        R_mixed = np.zeros((2,2,2,2))
        for ii, i in enumerate([0,1]):
            for jj, j in enumerate([2,3]):
                for kk, k in enumerate([0,1]):
                    for ll, l in enumerate([2,3]):
                        R_mixed[ii,jj,kk,ll] = R_func(i, j, k, l)
        
        gm = G0
        
        # Scan over all mixed planes
        K_min = np.inf
        K_max = -np.inf
        for ia in range(360):
            ca, sa = np.cos(np.pi*ia/360), np.sin(np.pi*ia/360)
            X = np.array([ca, sa])
            for ib in range(360):
                cb, sb = np.cos(np.pi*ib/360), np.sin(np.pi*ib/360)
                Y = np.array([cb, sb])
                num = sum(R_mixed[ii,jj,kk,ll]*X[ii]*Y[jj]*X[kk]*Y[ll]
                          for ii in range(2) for jj in range(2) for kk in range(2) for ll in range(2))
                gXX = sum(gm[i,k]*X[ii]*X[kk] for ii,i in enumerate([0,1]) for kk,k in enumerate([0,1]))
                gYY = sum(gm[j,l]*Y[jj]*Y[ll] for jj,j in enumerate([2,3]) for ll,l in enumerate([2,3]))
                gXY = sum(gm[i,j]*X[ii]*Y[jj] for ii,i in enumerate([0,1]) for jj,j in enumerate([2,3]))
                denom = gXX*gYY - gXY**2
                K_val = num/denom
                K_min = min(K_min, K_val)
                K_max = max(K_max, K_val)
        
        print(f"  pt=({pt[0]:.2f},{pt[1]:.2f},{pt[2]:.2f},{pt[3]:.2f}): K_min={K_min:+.8e}, K_max={K_max:+.8e}")

# Random l=1 seams
print(f"\n{'='*60}")
print("Random l=1 seams (exact local FD)")
print(f"{'='*60}")
rng = np.random.default_rng(42)
random_point = (np.pi/4, np.pi/4, np.pi/4, np.pi/4)
for trial in range(5):
    c = rng.normal(size=9)
    sf = make_l1_seam(c)
    R_func, G0 = compute_curvature_at_point(sf, gamma, random_point)
    
    R_mixed = np.zeros((2,2,2,2))
    for ii, i in enumerate([0,1]):
        for jj, j in enumerate([2,3]):
            for kk, k in enumerate([0,1]):
                for ll, l in enumerate([2,3]):
                    R_mixed[ii,jj,kk,ll] = R_func(i, j, k, l)
    
    gm = G0
    K_min = np.inf; K_max = -np.inf
    for ia in range(360):
        ca, sa = np.cos(np.pi*ia/360), np.sin(np.pi*ia/360)
        X = np.array([ca, sa])
        for ib in range(360):
            cb, sb = np.cos(np.pi*ib/360), np.sin(np.pi*ib/360)
            Y = np.array([cb, sb])
            num = sum(R_mixed[ii,jj,kk,ll]*X[ii]*Y[jj]*X[kk]*Y[ll]
                      for ii in range(2) for jj in range(2) for kk in range(2) for ll in range(2))
            gXX = sum(gm[i,k]*X[ii]*X[kk] for ii,i in enumerate([0,1]) for kk,k in enumerate([0,1]))
            gYY = sum(gm[j,l]*Y[jj]*Y[ll] for jj,j in enumerate([2,3]) for ll,l in enumerate([2,3]))
            gXY = sum(gm[i,j]*X[ii]*Y[jj] for ii,i in enumerate([0,1]) for jj,j in enumerate([2,3]))
            denom = gXX*gYY - gXY**2
            K_val = num/denom
            K_min = min(K_min, K_val)
            K_max = max(K_max, K_val)
    print(f"  Trial {trial+1}: K_min={K_min:+.8e}, K_max={K_max:+.8e}")
