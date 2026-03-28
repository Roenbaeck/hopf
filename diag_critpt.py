"""Quick diagnostic: compute K at critical point of x1*x2 using known-working
seam format from verify_brioschi_separable.py vs new make_product_seam."""
import numpy as np
from test_local_fine import compute_K_at_point

def make_seam_x1x2_v1(t1, p1, t2, p2):
    """x1*x2 seam using manual analytic derivatives."""
    st1, ct1 = np.sin(t1), np.cos(t1)
    sp1, cp1 = np.sin(p1), np.cos(p1)
    st2, ct2 = np.sin(t2), np.cos(t2)
    sp2, cp2 = np.sin(p2), np.cos(p2)
    
    x1, y1, z1 = st1*cp1, st1*sp1, ct1
    x2, y2, z2 = st2*cp2, st2*sp2, ct2
    s = x1 * x2
    
    # Analytic coord derivatives of x = sin(t)cos(p)
    # dx/dt = cos(t)cos(p), dx/dp = -sin(t)sin(p)
    dx1_dt1 = ct1*cp1
    dx1_dp1 = -st1*sp1
    dx2_dt2 = ct2*cp2
    dx2_dp2 = -st2*sp2
    
    ds = [None]*4
    ds[0] = dx1_dt1 * x2            # ds/dt1
    ds[1] = dx1_dp1 * x2            # ds/dp1
    ds[2] = x1 * dx2_dt2            # ds/dt2
    ds[3] = x1 * dx2_dp2            # ds/dp2
    
    # Second derivatives
    d2x1_dt1dt1 = -st1*cp1          # = -x1
    d2x1_dp1dp1 = -st1*cp1          # = -x1
    d2x1_dt1dp1 = -ct1*sp1
    d2x2_dt2dt2 = -st2*cp2          # = -x2
    d2x2_dp2dp2 = -st2*cp2          # = -x2
    d2x2_dt2dp2 = -ct2*sp2
    
    d2s = [[None]*4 for _ in range(4)]
    d2s[0][0] = d2x1_dt1dt1 * x2
    d2s[0][1] = d2x1_dt1dp1 * x2
    d2s[0][2] = dx1_dt1 * dx2_dt2   # cross
    d2s[0][3] = dx1_dt1 * dx2_dp2
    d2s[1][1] = d2x1_dp1dp1 * x2
    d2s[1][2] = dx1_dp1 * dx2_dt2
    d2s[1][3] = dx1_dp1 * dx2_dp2
    d2s[2][2] = x1 * d2x2_dt2dt2
    d2s[2][3] = x1 * d2x2_dt2dp2
    d2s[3][3] = x1 * d2x2_dp2dp2
    for i in range(4):
        for j in range(i+1, 4):
            d2s[j][i] = d2s[i][j]
    
    return s, ds, d2s


def make_seam_x1x2_v2(t1, p1, t2, p2):
    """x1*x2 seam using numerical derivatives (same approach as verify_critpt_product.py)."""
    st1, ct1 = np.sin(t1), np.cos(t1)
    sp1, cp1 = np.sin(p1), np.cos(p1)
    st2, ct2 = np.sin(t2), np.cos(t2)
    sp2, cp2 = np.sin(p2), np.cos(p2)
    
    x1, x2 = st1*cp1, st2*cp2
    s = x1 * x2
    
    def _eval(tt1, pp1, tt2, pp2):
        return np.sin(tt1)*np.cos(pp1) * np.sin(tt2)*np.cos(pp2)
    
    h = 1e-7
    coords = [t1, p1, t2, p2]
    ds = [None]*4
    d2s = [[None]*4 for _ in range(4)]
    
    for i in range(4):
        cp = list(coords); cm = list(coords)
        cp[i] = coords[i] + h; cm[i] = coords[i] - h
        ds[i] = (_eval(*cp) - _eval(*cm)) / (2*h)
    
    for i in range(4):
        for j in range(i, 4):
            if i == j:
                cp = list(coords); cm = list(coords)
                cp[i] = coords[i] + h; cm[i] = coords[i] - h
                d2s[i][j] = (_eval(*cp) - 2*s + _eval(*cm)) / h**2
            else:
                pp = list(coords); pm = list(coords); mp = list(coords); mm = list(coords)
                pp[i] = coords[i]+h; pp[j] = coords[j]+h
                pm[i] = coords[i]+h; pm[j] = coords[j]-h
                mp[i] = coords[i]-h; mp[j] = coords[j]+h
                mm[i] = coords[i]-h; mm[j] = coords[j]-h
                d2s[i][j] = (_eval(*pp) - _eval(*pm) - _eval(*mp) + _eval(*mm)) / (4*h**2)
            d2s[j][i] = d2s[i][j] if j > i else d2s[i][j]
    
    return s, ds, d2s

gamma = 0.05

# Test at critical point: theta=pi/2, phi=0 for both factors
crit = (np.pi/2, 0.0, np.pi/2, 0.0)
# Slightly perturbed to avoid exact phi=0
crit_safe = (np.pi/2, 0.01, np.pi/2, 0.01)

# Test at generic point for comparison
generic = (0.7, 0.5, 0.9, 1.2)

for label, center in [("critical (pi/2,0,pi/2,0)", crit),
                       ("critical safe (pi/2,0.01,pi/2,0.01)", crit_safe),
                       ("generic (0.7,0.5,0.9,1.2)", generic)]:
    print(f"\n=== {label} ===")
    for name, sf in [("analytic", make_seam_x1x2_v1), ("numerical", make_seam_x1x2_v2)]:
        try:
            K_min, K_max, K_coord = compute_K_at_point(sf, gamma, center, h=1e-4)
            print(f"  {name}: Kmin={K_min:+.6e} Kmax={K_max:+.6e}")
            for key in ["th1,th2", "th1,ph2", "ph1,th2", "ph1,ph2"]:
                print(f"    K({key}) = {K_coord[key]:+.6e}")
        except Exception as e:
            print(f"  {name}: ERROR: {e}")
