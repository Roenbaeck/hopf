#!/usr/bin/env python3
"""
Test with l=2 spherical harmonics on S^2 x S^2.

The l=1 test showed K <= 0 everywhere, but l=1 harmonics have
special Hessian structure (nabla^2 Y_1^m = -Y_1^m g).
For l=2, the traceless Hessian is nontrivial.

l=2 real spherical harmonics (unnormalized):
  Y20 ~ 3z^2 - 1
  Y21c ~ xz
  Y21s ~ yz
  Y22c ~ x^2 - y^2
  Y22s ~ xy
"""

import numpy as np


def make_seam_general(basis_pairs, coeffs):
    """
    General seam: s = sum_k c_k * f_k(unit1) * g_k(unit2)
    where f_k, g_k are spherical harmonics specified as functions of
    embedding coords (x,y,z).

    basis_pairs: list of (f1_func, f2_func) pairs
    coeffs: list of coefficients
    """
    def seam_func(t1, p1, t2, p2):
        st1, ct1 = np.sin(t1), np.cos(t1)
        sp1, cp1 = np.sin(p1), np.cos(p1)
        st2, ct2 = np.sin(t2), np.cos(t2)
        sp2, cp2 = np.sin(p2), np.cos(p2)

        x1, y1, z1 = st1*cp1, st1*sp1, ct1
        x2, y2, z2 = st2*cp2, st2*sp2, ct2

        z = np.zeros_like(t1)
        h = 1e-6  # FD step for derivatives

        # Evaluate s at base point
        def s_at(tt1, pp1, tt2, pp2):
            sx1 = np.sin(tt1)*np.cos(pp1)
            sy1 = np.sin(tt1)*np.sin(pp1)
            sz1 = np.cos(tt1)
            sx2 = np.sin(tt2)*np.cos(pp2)
            sy2 = np.sin(tt2)*np.sin(pp2)
            sz2 = np.cos(tt2)
            val = z.copy()
            for k, (f1, f2) in enumerate(basis_pairs):
                val = val + coeffs[k] * f1(sx1, sy1, sz1) * f2(sx2, sy2, sz2)
            return val

        s_val = s_at(t1, p1, t2, p2)

        # First derivatives via central differences
        ds = [None]*4
        coords = [t1, p1, t2, p2]
        for i in range(4):
            args_p = list(coords)
            args_m = list(coords)
            args_p[i] = coords[i] + h
            args_m[i] = coords[i] - h
            ds[i] = (s_at(*args_p) - s_at(*args_m)) / (2*h)

        # Second derivatives via central differences
        d2s = [[None]*4 for _ in range(4)]
        for i in range(4):
            for j in range(i, 4):
                if i == j:
                    args_p = list(coords); args_m = list(coords)
                    args_p[i] = coords[i] + h
                    args_m[i] = coords[i] - h
                    d2s[i][j] = (s_at(*args_p) - 2*s_val + s_at(*args_m)) / h**2
                else:
                    args_pp = list(coords); args_pm = list(coords)
                    args_mp = list(coords); args_mm = list(coords)
                    args_pp[i] = coords[i]+h; args_pp[j] = coords[j]+h
                    args_pm[i] = coords[i]+h; args_pm[j] = coords[j]-h
                    args_mp[i] = coords[i]-h; args_mp[j] = coords[j]+h
                    args_mm[i] = coords[i]-h; args_mm[j] = coords[j]-h
                    d2s[i][j] = (s_at(*args_pp) - s_at(*args_pm) - s_at(*args_mp) + s_at(*args_mm)) / (4*h**2)
                d2s[j][i] = d2s[i][j]

        return s_val, ds, d2s

    return seam_func


def build_curvature(seam_func, gamma, N=12):
    eps_th = 0.15
    th1 = np.linspace(eps_th, np.pi - eps_th, N)
    ph1 = np.linspace(0, 2*np.pi, N, endpoint=False)
    th2 = np.linspace(eps_th, np.pi - eps_th, N)
    ph2 = np.linspace(0, 2*np.pi, N, endpoint=False)
    dxs = [th1[1]-th1[0], ph1[1]-ph1[0], th2[1]-th2[0], ph2[1]-ph2[0]]

    T1, P1, T2, P2 = np.meshgrid(th1, ph1, th2, ph2, indexing='ij')
    shape = T1.shape

    S, dS, d2S = seam_func(T1, P1, T2, P2)

    st1, ct1 = np.sin(T1), np.cos(T1)
    st2, ct2 = np.sin(T2), np.cos(T2)

    # Covariant Hessian
    H = np.zeros((4, 4) + shape)
    H[0,0] = d2S[0][0]
    H[0,1] = d2S[0][1] - (ct1/st1)*dS[1]
    H[0,2] = d2S[0][2]
    H[0,3] = d2S[0][3]
    H[1,1] = d2S[1][1] + st1*ct1*dS[0]
    H[1,2] = d2S[1][2]
    H[1,3] = d2S[1][3]
    H[2,2] = d2S[2][2]
    H[2,3] = d2S[2][3] - (ct2/st2)*dS[3]
    H[3,3] = d2S[3][3] + st2*ct2*dS[2]
    for i in range(4):
        for j in range(i+1, 4):
            H[j,i] = H[i,j]

    G = gamma * H.copy()
    G[0,0] += 1.0; G[1,1] += st1**2; G[2,2] += 1.0; G[3,3] += st2**2

    def fd(arr, axis):
        return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2*dxs[axis])

    dG = np.zeros((4, 4, 4) + shape)
    for k in range(4):
        for i in range(4):
            for j in range(i, 4):
                dG[i,j,k] = fd(G[i,j], k)
                dG[j,i,k] = dG[i,j,k]

    Gf = G.reshape(4, 4, -1).transpose(2, 0, 1)
    Gif = np.linalg.inv(Gf)
    Ginv = Gif.transpose(1, 2, 0).reshape(4, 4, *shape)

    Chr = np.zeros((4, 4, 4) + shape)
    for m in range(4):
        for i in range(4):
            for j in range(i, 4):
                val = np.zeros(shape)
                for k in range(4):
                    val += Ginv[m,k] * (dG[k,i,j] + dG[k,j,i] - dG[i,j,k])
                Chr[m,i,j] = 0.5 * val
                if j > i: Chr[m,j,i] = Chr[m,i,j]

    dChr = np.zeros((4, 4, 4, 4) + shape)
    for l in range(4):
        for m in range(4):
            for i in range(4):
                for j in range(i, 4):
                    dChr[m,i,j,l] = fd(Chr[m,i,j], l)
                    if j > i: dChr[m,j,i,l] = dChr[m,i,j,l]

    R_0202 = np.zeros(shape)
    for m in range(4):
        Rm = dChr[m,0,2,2] - dChr[m,2,2,0]
        for p in range(4):
            Rm += Chr[m,2,p]*Chr[p,0,2] - Chr[m,0,p]*Chr[p,2,2]
        R_0202 += G[0,m] * Rm

    denom = G[0,0]*G[2,2] - G[0,2]**2
    K = R_0202 / denom
    dV = st1 * st2 * dxs[0]*dxs[1]*dxs[2]*dxs[3]

    return K, dV, shape


def main():
    rng = np.random.default_rng(42)

    # l=1 spherical harmonics (embedding coords)
    f_x = lambda x,y,z: x
    f_y = lambda x,y,z: y
    f_z = lambda x,y,z: z

    # l=2 spherical harmonics (unnormalized)
    f_z2 = lambda x,y,z: 3*z**2 - 1    # Y20
    f_xz = lambda x,y,z: x*z           # Y21c
    f_yz = lambda x,y,z: y*z           # Y21s
    f_x2y2 = lambda x,y,z: x**2 - y**2  # Y22c
    f_xy = lambda x,y,z: x*y           # Y22s

    l1_funcs = [f_x, f_y, f_z]
    l2_funcs = [f_z2, f_xz, f_yz, f_x2y2, f_xy]
    all_funcs = l1_funcs + l2_funcs

    gamma = 0.01
    N = 10  # smaller grid due to FD overhead
    trim = 2

    print("=" * 70)
    print("Mixed curvature test on S^2 x S^2 with l=1 and l=2 harmonics")
    print(f"gamma = {gamma}, N = {N}")
    print("=" * 70)

    # Pure l=2 x l=2 seams
    print("\n--- Pure l=2 x l=2 product seams ---")
    print(f"{'Name':<25} {'K min':>12} {'K max':>12} {'int K':>12} {'sign':>6}")
    print("-" * 70)

    for i, f1 in enumerate(l2_funcs):
        for j, f2 in enumerate(l2_funcs):
            if i > j: continue
            name = f"l2_{i} x l2_{j}"
            sf = make_seam_general([(f1, f2)], [1.0])
            K, dV, shape = build_curvature(sf, gamma, N)
            sl = tuple(slice(trim, -trim) for _ in range(4))
            Ki, dVi = K[sl], dV[sl]
            kmin, kmax = np.min(Ki), np.max(Ki)
            int_K = np.sum(Ki * dVi)
            sign = kmin < -1e-12 and kmax > 1e-12
            print(f"{name:<25} {kmin:>12.4e} {kmax:>12.4e} {int_K:>12.4e} {'YES' if sign else 'NO':>6}")

    # Random l=1 + l=2 mixed seams
    print("\n--- Random mixed l=1,l=2 seams ---")
    print(f"{'Name':<25} {'K min':>12} {'K max':>12} {'int K':>12} {'sign':>6}")
    print("-" * 70)

    n_sign = 0
    n_total = 0
    for trial in range(15):
        # Random pairs from all harmonics
        n_terms = rng.integers(2, 6)
        pairs = []
        coeffs = []
        for _ in range(n_terms):
            i1 = rng.integers(0, len(all_funcs))
            i2 = rng.integers(0, len(all_funcs))
            pairs.append((all_funcs[i1], all_funcs[i2]))
            coeffs.append(rng.normal())

        sf = make_seam_general(pairs, coeffs)
        K, dV, shape = build_curvature(sf, gamma, N)
        sl = tuple(slice(trim, -trim) for _ in range(4))
        Ki, dVi = K[sl], dV[sl]
        kmin, kmax = np.min(Ki), np.max(Ki)
        int_K = np.sum(Ki * dVi)
        sign = kmin < -1e-12 and kmax > 1e-12
        print(f"random_{trial:<18} {kmin:>12.4e} {kmax:>12.4e} {int_K:>12.4e} {'YES' if sign else 'NO':>6}")
        n_total += 1
        if sign: n_sign += 1

    print(f"\nK changes sign: {n_sign}/{n_total}")

    # Verify first-order cancellation for l=2
    print("\n--- First-order cancellation for l=2 seam ---")
    sf = make_seam_general([(f_xz, f_yz)], [1.0])
    for g in [0.1, 0.01, 0.001]:
        K, dV, _ = build_curvature(sf, g, N=8)
        sl = tuple(slice(2, -2) for _ in range(4))
        Ki = K[sl]
        print(f"  gamma={g}: max|K|={np.max(np.abs(Ki)):.4e}, max|K/g^2|={np.max(np.abs(Ki))/g**2:.4e}")


if __name__ == '__main__':
    main()
