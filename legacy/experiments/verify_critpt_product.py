"""
Verify: R_{a alpha b beta} = 0 at simultaneous critical points of f and g,
for product seams s = f(x1) * g(x2) of ANY ell, including non-axisymmetric.

Test cases:
  - l=2: Y21c * Y21c  (f depends on theta AND phi)
  - l=2: Y22c * Y22c
  - l=2: Y21c * Y22s  (mixed m)
  - l=1: xz (non-separable in theta/phi but still product)
"""
import numpy as np
from test_local_fine import compute_K_at_point


def make_product_seam(f_func, g_func):
    """Product seam s = f(embedding1) * g(embedding2)."""
    def seam_func(t1, p1, t2, p2):
        st1, ct1 = np.sin(t1), np.cos(t1)
        sp1, cp1 = np.sin(p1), np.cos(p1)
        st2, ct2 = np.sin(t2), np.cos(t2)
        sp2, cp2 = np.sin(p2), np.cos(p2)
        x1, y1, z1 = st1*cp1, st1*sp1, ct1
        x2, y2, z2 = st2*cp2, st2*sp2, ct2

        f = f_func(x1, y1, z1)
        g = g_func(x2, y2, z2)
        s = f * g

        h = 1e-7
        coords = [t1, p1, t2, p2]
        ds = [None]*4
        d2s = [[None]*4 for _ in range(4)]

        for i in range(4):
            cp = list(coords); cm = list(coords)
            cp[i] = coords[i] + h; cm[i] = coords[i] - h
            sp = _eval(f_func, g_func, *cp)
            sm = _eval(f_func, g_func, *cm)
            ds[i] = (sp - sm) / (2*h)

        for i in range(4):
            for j in range(i, 4):
                cp_ij = list(coords); cm_ij = list(coords)
                cp_ji = list(coords); cm_ji = list(coords)
                cp_ij[i] = coords[i] + h; cp_ij[j] = coords[j] + h
                cm_ij[i] = coords[i] - h; cm_ij[j] = coords[j] - h
                cp_ji[i] = coords[i] + h; cp_ji[j] = coords[j] - h
                cm_ji[i] = coords[i] - h; cm_ji[j] = coords[j] + h
                if i == j:
                    sp = _eval(f_func, g_func, *cp_ij)
                    sm = _eval(f_func, g_func, *cm_ij)
                    d2s[i][j] = (sp - 2*s + sm) / h**2
                else:
                    d2s[i][j] = (_eval(f_func, g_func, *cp_ij)
                                 - _eval(f_func, g_func, *cp_ji)
                                 - _eval(f_func, g_func, *cm_ji)
                                 + _eval(f_func, g_func, *cm_ij)) / (4*h**2)
                    d2s[j][i] = d2s[i][j]

        return s, ds, d2s
    return seam_func


def _eval(f_func, g_func, t1, p1, t2, p2):
    st1, ct1 = np.sin(t1), np.cos(t1)
    sp1, cp1 = np.sin(p1), np.cos(p1)
    st2, ct2 = np.sin(t2), np.cos(t2)
    sp2, cp2 = np.sin(p2), np.cos(p2)
    x1, y1, z1 = st1*cp1, st1*sp1, ct1
    x2, y2, z2 = st2*cp2, st2*sp2, ct2
    return f_func(x1, y1, z1) * g_func(x2, y2, z2)


# Spherical harmonic building blocks
def Y10(x, y, z): return z          # l=1, m=0
def Y11c(x, y, z): return x         # l=1, m=1 cos
def Y21c(x, y, z): return x*z       # l=2, m=1 cos (prop to sin th cos th cos phi)
def Y21s(x, y, z): return y*z       # l=2, m=1 sin
def Y22c(x, y, z): return x**2 - y**2  # l=2, m=2 cos
def Y22s(x, y, z): return x*y       # l=2, m=2 sin
def Y20(x, y, z): return 3*z**2 - 1  # l=2, m=0


def find_critical_points_of_product(f_func, g_func):
    """Find simultaneous critical points (nabla_1 f = 0 AND nabla_2 g = 0).
    
    Critical points of f on S^2 are at stationary points of f restricted
    to the sphere. We search numerically on a grid.
    """
    from scipy.optimize import minimize

    crits = []

    # Find critical points of f on S^2
    def neg_abs_grad_f(tp):
        t, p = tp
        h = 1e-6
        st, ct = np.sin(t), np.cos(t)
        sp, cp = np.sin(p), np.cos(p)
        x, y, z = st*cp, st*sp, ct
        f0 = f_func(x, y, z)
        # d/dtheta
        xp = np.sin(t+h)*cp; yp = np.sin(t+h)*sp; zp = np.cos(t+h)
        xm = np.sin(t-h)*cp; ym = np.sin(t-h)*sp; zm = np.cos(t-h)
        df_dt = (f_func(xp, yp, zp) - f_func(xm, ym, zm)) / (2*h)
        # d/dphi (divided by sin theta if we want covariant)
        xp2 = st*np.cos(p+h); yp2 = st*np.sin(p+h)
        xm2 = st*np.cos(p-h); ym2 = st*np.sin(p-h)
        df_dp = (f_func(xp2, yp2, z) - f_func(xm2, ym2, z)) / (2*h)
        grad_sq = df_dt**2 + df_dp**2 / max(st**2, 1e-10)
        return grad_sq

    # Same for g
    def neg_abs_grad_g(tp):
        t, p = tp
        h = 1e-6
        st, ct = np.sin(t), np.cos(t)
        sp, cp = np.sin(p), np.cos(p)
        x, y, z = st*cp, st*sp, ct
        xp = np.sin(t+h)*cp; yp = np.sin(t+h)*sp; zp = np.cos(t+h)
        xm = np.sin(t-h)*cp; ym = np.sin(t-h)*sp; zm = np.cos(t-h)
        dg_dt = (g_func(xp, yp, zp) - g_func(xm, ym, zm)) / (2*h)
        xp2 = st*np.cos(p+h); yp2 = st*np.sin(p+h)
        xm2 = st*np.cos(p-h); ym2 = st*np.sin(p-h)
        dg_dp = (g_func(xp2, yp2, z) - g_func(xm2, ym2, z)) / (2*h)
        grad_sq = dg_dt**2 + dg_dp**2 / max(st**2, 1e-10)
        return grad_sq

    # Grid search for critical points of f
    f_crits = []
    for t0 in np.linspace(0.2, 2.9, 8):
        for p0 in np.linspace(0.1, 6.1, 8):
            res = minimize(neg_abs_grad_f, [t0, p0], method='Nelder-Mead',
                         options={'xatol': 1e-10, 'fatol': 1e-20})
            if res.fun < 1e-14:
                f_crits.append((res.x[0] % np.pi, res.x[1] % (2*np.pi)))

    g_crits = []
    for t0 in np.linspace(0.2, 2.9, 8):
        for p0 in np.linspace(0.1, 6.1, 8):
            res = minimize(neg_abs_grad_g, [t0, p0], method='Nelder-Mead',
                         options={'xatol': 1e-10, 'fatol': 1e-20})
            if res.fun < 1e-14:
                g_crits.append((res.x[0] % np.pi, res.x[1] % (2*np.pi)))

    # Deduplicate
    def dedup(pts, tol=0.05):
        out = []
        for p in pts:
            if not any(abs(p[0]-q[0]) < tol and abs(p[1]-q[1]) < tol for q in out):
                out.append(p)
        return out

    f_crits = dedup(f_crits)
    g_crits = dedup(g_crits)

    return f_crits, g_crits


def main():
    test_cases = [
        ("Y21c x Y21c", Y21c, Y21c),
        ("Y22c x Y22c", Y22c, Y22c),
        ("Y21c x Y22s", Y21c, Y22s),
        ("Y11c x Y11c (l=1 ref)", Y11c, Y11c),
    ]

    gamma = 0.05

    for name, f_func, g_func in test_cases:
        print(f"\n{'='*60}")
        print(f"Seam: {name}, gamma={gamma}")
        print(f"{'='*60}")

        sf = make_product_seam(f_func, g_func)
        f_crits, g_crits = find_critical_points_of_product(f_func, g_func)

        print(f"  Critical points of f: {len(f_crits)}")
        print(f"  Critical points of g: {len(g_crits)}")

        if not f_crits or not g_crits:
            print("  No simultaneous critical points found!")
            continue

        # Test at up to 4 simultaneous critical points
        count = 0
        for fc in f_crits[:3]:
            for gc in g_crits[:3]:
                center = (fc[0], fc[1], gc[0], gc[1])
                # Avoid poles
                if fc[0] < 0.05 or fc[0] > np.pi - 0.05:
                    continue
                if gc[0] < 0.05 or gc[0] > np.pi - 0.05:
                    continue

                K_min, K_max, K_coord = compute_K_at_point(sf, gamma, center, h=1e-4)
                print(f"  ({fc[0]:.3f},{fc[1]:.3f},{gc[0]:.3f},{gc[1]:.3f}):")
                for key in ["th1,th2", "th1,ph2", "ph1,th2", "ph1,ph2"]:
                    print(f"    K({key}) = {K_coord[key]:+.6e}")
                print(f"    Kmin={K_min:+.6e}  Kmax={K_max:+.6e}")
                count += 1
                if count >= 4:
                    break
            if count >= 4:
                break

    # Also test at a NON-critical point for comparison
    print(f"\n{'='*60}")
    print("COMPARISON: Y21c x Y21c at non-critical point (0.7, 0.5, 0.9, 1.2)")
    print(f"{'='*60}")
    sf = make_product_seam(Y21c, Y21c)
    center = (0.7, 0.5, 0.9, 1.2)
    K_min, K_max, K_coord = compute_K_at_point(sf, gamma, center, h=1e-4)
    for key in ["th1,th2", "th1,ph2", "ph1,th2", "ph1,ph2"]:
        print(f"  K({key}) = {K_coord[key]:+.6e}")
    print(f"  Kmin={K_min:+.6e}  Kmax={K_max:+.6e}")


if __name__ == "__main__":
    main()
