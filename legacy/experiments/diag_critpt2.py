"""Compare seam function outputs at a single point."""
import numpy as np

def make_seam_analytic(t1, p1, t2, p2):
    st1, ct1 = np.sin(t1), np.cos(t1)
    sp1, cp1 = np.sin(p1), np.cos(p1)
    st2, ct2 = np.sin(t2), np.cos(t2)
    sp2, cp2 = np.sin(p2), np.cos(p2)
    x1, x2 = st1*cp1, st2*cp2
    s = x1 * x2
    dx1_dt1 = ct1*cp1; dx1_dp1 = -st1*sp1
    dx2_dt2 = ct2*cp2; dx2_dp2 = -st2*sp2
    ds = [dx1_dt1*x2, dx1_dp1*x2, x1*dx2_dt2, x1*dx2_dp2]
    d2s = [[None]*4 for _ in range(4)]
    d2s[0][0] = -st1*cp1*x2; d2s[1][1] = -st1*cp1*x2
    d2s[2][2] = x1*(-st2*cp2); d2s[3][3] = x1*(-st2*cp2)
    d2s[0][1] = -ct1*sp1*x2; d2s[1][0] = d2s[0][1]
    d2s[0][2] = dx1_dt1*dx2_dt2; d2s[2][0] = d2s[0][2]
    d2s[0][3] = dx1_dt1*dx2_dp2; d2s[3][0] = d2s[0][3]
    d2s[1][2] = dx1_dp1*dx2_dt2; d2s[2][1] = d2s[1][2]
    d2s[1][3] = dx1_dp1*dx2_dp2; d2s[3][1] = d2s[1][3]
    d2s[2][3] = x1*(-ct2*sp2); d2s[3][2] = d2s[2][3]
    return s, ds, d2s

def make_seam_numerical(t1, p1, t2, p2):
    st1, ct1 = np.sin(t1), np.cos(t1)
    sp1, cp1 = np.sin(p1), np.cos(p1)
    st2, ct2 = np.sin(t2), np.cos(t2)
    sp2, cp2 = np.sin(p2), np.cos(p2)
    x1, x2 = st1*cp1, st2*cp2
    s = x1 * x2
    def _ev(tt1, pp1, tt2, pp2):
        return np.sin(tt1)*np.cos(pp1)*np.sin(tt2)*np.cos(pp2)
    h = 1e-7
    coords = [t1, p1, t2, p2]
    ds = [None]*4
    for i in range(4):
        c1 = list(coords); c2 = list(coords)
        c1[i] = coords[i]+h; c2[i] = coords[i]-h
        ds[i] = (_ev(*c1) - _ev(*c2))/(2*h)
    d2s = [[None]*4 for _ in range(4)]
    for i in range(4):
        for j in range(i, 4):
            if i == j:
                c1 = list(coords); c2 = list(coords)
                c1[i] = coords[i]+h; c2[i] = coords[i]-h
                d2s[i][j] = (_ev(*c1) - 2*s + _ev(*c2))/h**2
            else:
                pp = list(coords); pm = list(coords)
                mp = list(coords); mm = list(coords)
                pp[i]=coords[i]+h; pp[j]=coords[j]+h
                pm[i]=coords[i]+h; pm[j]=coords[j]-h
                mp[i]=coords[i]-h; mp[j]=coords[j]+h
                mm[i]=coords[i]-h; mm[j]=coords[j]-h
                d2s[i][j] = (_ev(*pp)-_ev(*pm)-_ev(*mp)+_ev(*mm))/(4*h**2)
            d2s[j][i] = d2s[i][j]
    return s, ds, d2s

# Test at critical point of x1*x2 on a SCALAR (not array)
t1, p1, t2, p2 = np.pi/2, 0.0, np.pi/2, 0.0

print("=== Scalar evaluation at critical point (pi/2, 0, pi/2, 0) ===")
s_a, ds_a, d2s_a = make_seam_analytic(t1, p1, t2, p2)
s_n, ds_n, d2s_n = make_seam_numerical(t1, p1, t2, p2)

print(f"s: analytic={s_a:.15e} numerical={s_n:.15e} diff={abs(s_a-s_n):.2e}")
for i in range(4):
    print(f"ds[{i}]: analytic={ds_a[i]:.15e} numerical={ds_n[i]:.15e} diff={abs(ds_a[i]-ds_n[i]):.2e}")
for i in range(4):
    for j in range(4):
        a = d2s_a[i][j]; n = d2s_n[i][j]
        if abs(a) > 1e-14 or abs(n) > 1e-14:
            print(f"d2s[{i}][{j}]: analytic={a:.15e} numerical={n:.15e} diff={abs(a-n):.2e}")

# Now test on small arrays (simulating compute_K_at_point grid)
print("\n=== Array evaluation (3x3x3x3 grid centered at critical point) ===")
h_grid = 1e-4
N = 3
t1_arr = np.pi/2 + h_grid*np.arange(-(N//2), N//2+1)
p1_arr = 0.0 + h_grid*np.arange(-(N//2), N//2+1)
t2_arr = np.pi/2 + h_grid*np.arange(-(N//2), N//2+1)
p2_arr = 0.0 + h_grid*np.arange(-(N//2), N//2+1)
T1, P1, T2, P2 = np.meshgrid(t1_arr, p1_arr, t2_arr, p2_arr, indexing='ij')

s_a, ds_a, d2s_a = make_seam_analytic(T1, P1, T2, P2)
s_n, ds_n, d2s_n = make_seam_numerical(T1, P1, T2, P2)

# Check center point (1,1,1,1)
c = (1,1,1,1)
print(f"s center: analytic={s_a[c]:.15e} numerical={s_n[c]:.15e}")
for i in range(4):
    print(f"ds[{i}] center: analytic={ds_a[i][c]:.15e} numerical={ds_n[i][c]:.15e}")
for i in range(4):
    for j in range(i, 4):
        a = d2s_a[i][j][c]; n = d2s_n[i][j][c]
        diff = abs(a-n)
        if abs(a) > 1e-14 or diff > 1e-10:
            print(f"d2s[{i}][{j}] center: analytic={a:.15e} numerical={n:.15e} diff={diff:.2e}")

# Check corners and max difference
max_diff_ds = max(np.max(np.abs(ds_a[i] - ds_n[i])) for i in range(4))
max_diff_d2s = max(np.max(np.abs(d2s_a[i][j] - d2s_n[i][j])) for i in range(4) for j in range(4))
print(f"\nMax |ds_a - ds_n| over grid: {max_diff_ds:.2e}")
print(f"Max |d2s_a - d2s_n| over grid: {max_diff_d2s:.2e}")
