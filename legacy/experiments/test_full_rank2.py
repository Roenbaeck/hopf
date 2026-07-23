#!/usr/bin/env python3
"""
Fast test: does the full-rank l=1 seam s = n1.n2 have K > 0 everywhere?
Focus on specific points including the "problematic" ones near antipodal points.
"""

import numpy as np
from test_Q_S2xS2 import make_seam_product_harmonics

def compute_K_at_point(seam_func, gamma, center, h=1e-4, scan_angles=90):
    """Compute min/max mixed K at a specific point."""
    t10, p10, t20, p20 = center
    N = 7
    th1 = t10 + h * np.arange(-(N//2), N//2+1)
    ph1 = p10 + h * np.arange(-(N//2), N//2+1)
    th2 = t20 + h * np.arange(-(N//2), N//2+1)
    ph2 = p20 + h * np.arange(-(N//2), N//2+1)
    T1, P1, T2, P2 = np.meshgrid(th1, ph1, th2, ph2, indexing='ij')
    shape = T1.shape
    st1, ct1 = np.sin(T1), np.cos(T1)
    st2, ct2 = np.sin(T2), np.cos(T2)
    S, dS, d2S = seam_func(T1, P1, T2, P2)
    H = np.zeros((4, 4) + shape)
    H[0,0] = d2S[0][0]; H[0,1] = d2S[0][1] - (ct1/st1)*dS[1]
    H[1,1] = d2S[1][1] + st1*ct1*dS[0]
    H[0,2] = d2S[0][2]; H[0,3] = d2S[0][3]; H[1,2] = d2S[1][2]; H[1,3] = d2S[1][3]
    H[2,2] = d2S[2][2]; H[2,3] = d2S[2][3] - (ct2/st2)*dS[3]
    H[3,3] = d2S[3][3] + st2*ct2*dS[2]
    for i in range(4):
        for j in range(i+1, 4):
            H[j,i] = H[i,j]
    G = gamma * H.copy()
    G[0,0] += 1; G[1,1] += st1**2; G[2,2] += 1; G[3,3] += st2**2
    def fd(arr, axis):
        return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2*h)
    dG = np.zeros((4,4,4)+shape)
    for k in range(4):
        for i in range(4):
            for j in range(i,4):
                dG[i,j,k] = fd(G[i,j], k); dG[j,i,k] = dG[i,j,k]
    Gf = G.reshape(4,4,-1).transpose(2,0,1)
    Gi = np.linalg.inv(Gf).transpose(1,2,0).reshape(4,4,*shape)
    Chr = np.zeros((4,4,4)+shape)
    for m in range(4):
        for i in range(4):
            for j in range(i,4):
                v = np.zeros(shape)
                for k in range(4):
                    v += Gi[m,k]*(dG[k,i,j]+dG[k,j,i]-dG[i,j,k])
                Chr[m,i,j] = 0.5*v
                if j>i: Chr[m,j,i] = Chr[m,i,j]
    dChr = np.zeros((4,4,4,4)+shape)
    for l in range(4):
        for m in range(4):
            for i in range(4):
                for j in range(i,4):
                    dChr[m,i,j,l] = fd(Chr[m,i,j],l)
                    if j>i: dChr[m,j,i,l] = dChr[m,i,j,l]
    c = N//2
    G0=G[:,:,c,c,c,c]; Chr0=Chr[:,:,:,c,c,c,c]; dChr0=dChr[:,:,:,:,c,c,c,c]
    def compute_R(i,j,k,l):
        val = 0
        for m in range(4):
            Rm = dChr0[m,j,l,k]-dChr0[m,j,k,l]
            for p in range(4):
                Rm += Chr0[m,k,p]*Chr0[p,j,l]-Chr0[m,l,p]*Chr0[p,j,k]
            val += G0[i,m]*Rm
        return val
    R_mixed = np.zeros((2,2,2,2))
    for ii,i in enumerate([0,1]):
        for jj,j in enumerate([2,3]):
            for kk,k in enumerate([0,1]):
                for ll,l in enumerate([2,3]):
                    R_mixed[ii,jj,kk,ll] = compute_R(i,j,k,l)
    K_min=np.inf; K_max=-np.inf
    for ia in range(scan_angles):
        ca,sa = np.cos(np.pi*ia/scan_angles), np.sin(np.pi*ia/scan_angles)
        X = np.array([ca, sa])
        for ib in range(scan_angles):
            cb,sb = np.cos(np.pi*ib/scan_angles), np.sin(np.pi*ib/scan_angles)
            Y = np.array([cb, sb])
            num = sum(R_mixed[ii,jj,kk,ll]*X[ii]*Y[jj]*X[kk]*Y[ll]
                      for ii in range(2) for jj in range(2) for kk in range(2) for ll in range(2))
            gXX = sum(G0[i,k]*X[ii]*X[kk] for ii,i in enumerate([0,1]) for kk,k in enumerate([0,1]))
            gYY = sum(G0[j,l]*Y[jj]*Y[ll] for jj,j in enumerate([2,3]) for ll,l in enumerate([2,3]))
            gXY = sum(G0[i,j]*X[ii]*Y[jj] for ii,i in enumerate([0,1]) for jj,j in enumerate([2,3]))
            denom = gXX*gYY - gXY**2
            K_val = num/denom
            K_min=min(K_min,K_val); K_max=max(K_max,K_val)
    return K_min, K_max

gamma = 0.1

# Key points chosen strategically + random
test_points = [
    (np.pi/4, np.pi/4, np.pi/4, np.pi/4),
    (np.pi/2, np.pi/2, np.pi/2, np.pi/2),
    (0.3, 0.5, 0.3, 0.5),  # near antipodal in n1.n2
    (0.3, 0.5, np.pi-0.3, np.pi+0.5),  # n1 ≈ -n2  
    (np.pi/3, 1.0, np.pi/3, 1.0),  # n1 = n2 (s = 1)
    (np.pi/3, 1.0, 2*np.pi/3, np.pi+1.0),  # n1 = -n2 (s = -1)
    (np.pi/4, 0, 3*np.pi/4, np.pi),  # n1 ≈ -n2
    (1.0, 2.0, 1.5, 3.0),  # generic
    (0.5, 1.0, 2.0, 4.0),  # generic
    (1.2, 0.3, 0.8, 2.5),  # generic
]

# Add random points
np.random.seed(42)
for _ in range(10):
    t1 = np.random.uniform(0.2, np.pi-0.2)
    p1 = np.random.uniform(0, 2*np.pi)
    t2 = np.random.uniform(0.2, np.pi-0.2)
    p2 = np.random.uniform(0, 2*np.pi)
    test_points.append((t1, p1, t2, p2))

# Compute n1.n2 value at each point
def n1_dot_n2(t1, p1, t2, p2):
    return (np.sin(t1)*np.cos(p1)*np.sin(t2)*np.cos(p2) + 
            np.sin(t1)*np.sin(p1)*np.sin(t2)*np.sin(p2) + 
            np.cos(t1)*np.cos(t2))

seams = {
    'n1.n2 (full rank)': make_seam_product_harmonics([1,0,0, 0,1,0, 0,0,1]),
    'z1z2 (rank 1)': make_seam_product_harmonics([0,0,0, 0,0,0, 0,0,1]),
    'x1x2+y1y2 (rank 2)': make_seam_product_harmonics([1,0,0, 0,1,0, 0,0,0]),
}

for name, sf in seams.items():
    print(f"\n{'='*70}")
    print(f"Seam: {name}, gamma={gamma}")
    print(f"{'='*70}")
    
    all_Kmin = []
    all_Kmax = []
    
    for pt in test_points:
        t1, p1, t2, p2 = pt
        s_val = n1_dot_n2(t1, p1, t2, p2) if 'n1.n2' in name else None
        Kmin, Kmax = compute_K_at_point(sf, gamma, pt)
        all_Kmin.append(Kmin)
        all_Kmax.append(Kmax)
        
        s_str = f", s={s_val:.4f}" if s_val is not None else ""
        if Kmin < 1e-6:
            print(f"  ({t1:.2f},{p1:.2f},{t2:.2f},{p2:.2f}{s_str}): "
                  f"Kmin={Kmin:+.4e}, Kmax={Kmax:+.4e}")
    
    print(f"\n  Summary:")
    print(f"    min(Kmin) = {min(all_Kmin):+.6e}")
    print(f"    max(Kmax) = {max(all_Kmax):+.6e}")
    print(f"    ALL K >= 0 (tol 1e-8)? {min(all_Kmin) >= -1e-8}")
    print(f"    ALL K > 0 (tol 1e-6)?  {min(all_Kmin) > 1e-6}")

# Special focus: n1.n2 seam at points where s = n1.n2 ≈ 0
print(f"\n{'='*70}")
print(f"FOCUS: n1.n2 seam at points with s ≈ 0 (equator of coupling)")
print(f"{'='*70}")
sf = make_seam_product_harmonics([1,0,0, 0,1,0, 0,0,1])
# n1.n2 = 0 when n1 ⊥ n2
# e.g., (pi/2, 0, pi/2, pi/2) → n1=(1,0,0), n2=(0,1,0): n1.n2=0
perp_points = [
    (np.pi/2, 0, np.pi/2, np.pi/2),  # (1,0,0).(0,1,0) = 0
    (np.pi/2, 0, 0.3, 0.5),  # n1=(1,0,0), roughly generic n2
    (np.pi/4, 0, np.pi/4, np.pi/2),  # need to check
    (np.pi/3, np.pi/6, 2*np.pi/3, np.pi/6+np.pi/2),  # roughly perpendicular
]

for pt in perp_points:
    t1, p1, t2, p2 = pt
    s_val = n1_dot_n2(t1, p1, t2, p2)
    Kmin, Kmax = compute_K_at_point(sf, gamma, pt)
    print(f"  ({t1:.2f},{p1:.2f},{t2:.2f},{p2:.2f}), s={s_val:.6f}: "
          f"Kmin={Kmin:+.4e}, Kmax={Kmax:+.4e}")

# h-convergence for n1.n2 at a generic point
print(f"\n{'='*70}")
print(f"h-convergence for n1.n2 at generic point (1.0, 2.0, 1.5, 3.0)")
print(f"{'='*70}")
sf = make_seam_product_harmonics([1,0,0, 0,1,0, 0,0,1])
pt = (1.0, 2.0, 1.5, 3.0)
for h in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]:
    Kmin, Kmax = compute_K_at_point(sf, gamma, pt, h=h, scan_angles=90)
    print(f"  h={h:.0e}: Kmin={Kmin:+.6e}, Kmax={Kmax:+.6e}")

# And at n1 ⊥ n2 point
print(f"\nh-convergence for n1.n2 at n1⊥n2 point (pi/2, 0, pi/2, pi/2)")
pt = (np.pi/2, 0, np.pi/2, np.pi/2)
for h in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]:
    Kmin, Kmax = compute_K_at_point(sf, gamma, pt, h=h, scan_angles=90)
    print(f"  h={h:.0e}: Kmin={Kmin:+.6e}, Kmax={Kmax:+.6e}")

# And at n1 = n2 (s=1)
print(f"\nh-convergence for n1.n2 at n1=n2 point (pi/3, 1, pi/3, 1)")
pt = (np.pi/3, 1.0, np.pi/3, 1.0)
s_val = n1_dot_n2(*pt)
print(f"  s = {s_val:.6f}")
for h in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]:
    Kmin, Kmax = compute_K_at_point(sf, gamma, pt, h=h, scan_angles=90)
    print(f"  h={h:.0e}: Kmin={Kmin:+.6e}, Kmax={Kmax:+.6e}")
