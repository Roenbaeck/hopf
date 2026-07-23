#!/usr/bin/env python3
"""
Test mixed curvature sign-change on S^2 x S^2 (not flat torus).

For g = h_round + gamma * Hess_h(s), compute the full Riemann tensor
R_{0202} via Christoffel symbols using analytical derivatives of
product spherical harmonics.

Tests:
1. K_{mix} changes sign for all non-equivariant seams
2. int K dV ~ 0 (i.e., integral identity survives curvature)
3. K/gamma^2 bounded as gamma -> 0 (first-order cancellation)
"""

import numpy as np
import sys


def make_seam_product_harmonics(coeffs):
    """
    Return seam_func for s = sum coeffs[i]*f_i where f_i are products of
    l=1 spherical harmonics: {x1,y1,z1} x {x2,y2,z2}.

    coeffs: 9 elements for (x1x2, x1y2, x1z2, y1x2, y1y2, y1z2, z1x2, z1y2, z1z2).

    Returns function(t1, p1, t2, p2) -> (s, ds[4], d2s[4][4])
    with analytical coordinate derivatives.
    """
    c = np.array(coeffs)

    def seam_func(t1, p1, t2, p2):
        st1, ct1 = np.sin(t1), np.cos(t1)
        sp1, cp1 = np.sin(p1), np.cos(p1)
        st2, ct2 = np.sin(t2), np.cos(t2)
        sp2, cp2 = np.sin(p2), np.cos(p2)

        # Embedding coords and their derivatives w.r.t. (t1, p1, t2, p2)
        # Factor 1: x1=st1*cp1, y1=st1*sp1, z1=ct1
        # Factor 2: x2=st2*cp2, y2=st2*sp2, z2=ct2
        emb1 = [st1*cp1, st1*sp1, ct1]
        emb2 = [st2*cp2, st2*sp2, ct2]

        # d(emb1)/d(coord_i) for i=0..3
        # d/dt1: (ct1*cp1, ct1*sp1, -st1)
        # d/dp1: (-st1*sp1, st1*cp1, 0)
        # d/dt2: (0, 0, 0)
        # d/dp2: (0, 0, 0)
        de1 = [[ct1*cp1, -st1*sp1, 0, 0],
               [ct1*sp1,  st1*cp1, 0, 0],
               [-st1,     0,       0, 0]]

        # d(emb2)/d(coord_i)
        de2 = [[0, 0, ct2*cp2, -st2*sp2],
               [0, 0, ct2*sp2,  st2*cp2],
               [0, 0, -st2,     0]]

        # d^2(emb1)/d(coord_i)d(coord_j) -- only nonzero for i,j in {0,1}
        # d2x1/dt1dt1 = -st1*cp1, d2x1/dt1dp1 = -ct1*sp1, d2x1/dp1dp1 = -st1*cp1
        # d2y1/dt1dt1 = -st1*sp1, d2y1/dt1dp1 = ct1*cp1, d2y1/dp1dp1 = -st1*sp1
        # d2z1/dt1dt1 = -ct1, rest = 0
        d2e1 = [[[None]*4 for _ in range(4)] for _ in range(3)]
        d2e1[0][0][0] = -st1*cp1; d2e1[0][0][1] = -ct1*sp1; d2e1[0][1][0] = -ct1*sp1
        d2e1[0][1][1] = -st1*cp1
        d2e1[1][0][0] = -st1*sp1; d2e1[1][0][1] = ct1*cp1; d2e1[1][1][0] = ct1*cp1
        d2e1[1][1][1] = -st1*sp1
        d2e1[2][0][0] = -ct1

        d2e2 = [[[None]*4 for _ in range(4)] for _ in range(3)]
        d2e2[0][2][2] = -st2*cp2; d2e2[0][2][3] = -ct2*sp2; d2e2[0][3][2] = -ct2*sp2
        d2e2[0][3][3] = -st2*cp2
        d2e2[1][2][2] = -st2*sp2; d2e2[1][2][3] = ct2*cp2; d2e2[1][3][2] = ct2*cp2
        d2e2[1][3][3] = -st2*sp2
        d2e2[2][2][2] = -ct2

        # Build s = sum_a sum_b c[3a+b] * emb1[a] * emb2[b]
        z = np.zeros_like(t1)
        s_val = z.copy()
        for a in range(3):
            for b in range(3):
                s_val = s_val + c[3*a+b] * emb1[a] * emb2[b]

        # ds/d(coord_i)
        ds = [z.copy() for _ in range(4)]
        for a in range(3):
            for b in range(3):
                cc = c[3*a+b]
                if cc == 0: continue
                for i in range(4):
                    ds[i] = ds[i] + cc * (de1[a][i] * emb2[b] + emb1[a] * de2[b][i])

        # d^2s / d(coord_i) d(coord_j)
        d2s = [[z.copy() for _ in range(4)] for _ in range(4)]
        for a in range(3):
            for b in range(3):
                cc = c[3*a+b]
                if cc == 0: continue
                for i in range(4):
                    for j in range(i, 4):
                        val = z.copy()
                        # Product rule: d2(f1*f2)/didj = d2f1/didj*f2 + df1/di*df2/dj + df1/dj*df2/di + f1*d2f2/didj
                        dd1 = d2e1[a][i][j]
                        if dd1 is not None:
                            val = val + dd1 * emb2[b]
                        val = val + de1[a][i] * de2[b][j]
                        val = val + de1[a][j] * de2[b][i]
                        dd2 = d2e2[b][i][j]
                        if dd2 is not None:
                            val = val + emb1[a] * dd2
                        d2s[i][j] = d2s[i][j] + cc * val
                        if j > i:
                            d2s[j][i] = d2s[i][j]

        return s_val, ds, d2s

    return seam_func


def build_metric_and_curvature(seam_func, gamma, N=12):
    """
    Build 4D metric g = h + gamma*Hess_h(s) on S^2 x S^2 and
    compute mixed sectional curvature K(d/dtheta1, d/dtheta2).

    Uses analytical seam derivatives + finite differences for Christoffel derivatives.
    """
    eps_th = 0.15
    th1 = np.linspace(eps_th, np.pi - eps_th, N)
    ph1 = np.linspace(0, 2*np.pi, N, endpoint=False)
    th2 = np.linspace(eps_th, np.pi - eps_th, N)
    ph2 = np.linspace(0, 2*np.pi, N, endpoint=False)
    dxs = [th1[1]-th1[0], ph1[1]-ph1[0], th2[1]-th2[0], ph2[1]-ph2[0]]

    T1, P1, T2, P2 = np.meshgrid(th1, ph1, th2, ph2, indexing='ij')
    shape = T1.shape  # (N, N, N, N)

    # Get seam data (analytical)
    S, dS, d2S = seam_func(T1, P1, T2, P2)

    st1, ct1 = np.sin(T1), np.cos(T1)
    st2, ct2 = np.sin(T2), np.cos(T2)

    # Covariant Hessian: H_ij = s_{,ij} - Gamma^k_ij s_{,k}
    # Background Christoffels (S^2 x S^2, coords (th1,ph1,th2,ph2)):
    # Gamma^0_{11} = -st1*ct1, Gamma^1_{01} = ct1/st1
    # Gamma^2_{33} = -st2*ct2, Gamma^3_{23} = ct2/st2
    H = np.zeros((4, 4) + shape)
    H[0,0] = d2S[0][0]
    H[0,1] = d2S[0][1] - (ct1/st1) * dS[1]
    H[0,2] = d2S[0][2]
    H[0,3] = d2S[0][3]
    H[1,1] = d2S[1][1] + st1*ct1 * dS[0]
    H[1,2] = d2S[1][2]
    H[1,3] = d2S[1][3]
    H[2,2] = d2S[2][2]
    H[2,3] = d2S[2][3] - (ct2/st2) * dS[3]
    H[3,3] = d2S[3][3] + st2*ct2 * dS[2]
    for i in range(4):
        for j in range(i+1, 4):
            H[j,i] = H[i,j]

    # Full metric g = h + gamma*H
    G = gamma * H.copy()
    G[0,0] += 1.0
    G[1,1] += st1**2
    G[2,2] += 1.0
    G[3,3] += st2**2

    # Finite difference for metric derivatives
    def fd(arr, axis):
        return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2*dxs[axis])

    # dG_{ij}/dx^k
    dG = np.zeros((4, 4, 4) + shape)
    for k in range(4):
        for i in range(4):
            for j in range(i, 4):
                dG[i,j,k] = fd(G[i,j], k)
                dG[j,i,k] = dG[i,j,k]

    # Inverse metric (vectorized)
    Gflat = G.reshape(4, 4, -1).transpose(2, 0, 1)
    Giflat = np.linalg.inv(Gflat)
    Ginv = Giflat.transpose(1, 2, 0).reshape(4, 4, *shape)

    # Christoffel symbols
    Chr = np.zeros((4, 4, 4) + shape)
    for m in range(4):
        for i in range(4):
            for j in range(i, 4):
                val = np.zeros(shape)
                for k in range(4):
                    val += Ginv[m,k] * (dG[k,i,j] + dG[k,j,i] - dG[i,j,k])
                Chr[m,i,j] = 0.5 * val
                if j > i:
                    Chr[m,j,i] = Chr[m,i,j]

    # Christoffel derivatives (finite difference)
    dChr = np.zeros((4, 4, 4, 4) + shape)
    for l in range(4):
        for m in range(4):
            for i in range(4):
                for j in range(i, 4):
                    dChr[m,i,j,l] = fd(Chr[m,i,j], l)
                    if j > i:
                        dChr[m,j,i,l] = dChr[m,i,j,l]

    # R_{0202}: mixed curvature (theta1, theta2) plane
    # R^m_{bcd}: a=0,b=2,c=0,d=2
    # R^m_{202} = dChr^m_{02}/dx^2 - dChr^m_{22}/dx^0 + sum_p [Chr^m_{2p}Chr^p_{02} - Chr^m_{0p}Chr^p_{22}]
    R_0202 = np.zeros(shape)
    for m in range(4):
        Rm = dChr[m,0,2,2] - dChr[m,2,2,0]
        for p in range(4):
            Rm += Chr[m,2,p]*Chr[p,0,2] - Chr[m,0,p]*Chr[p,2,2]
        R_0202 += G[0,m] * Rm

    # Sectional curvature
    denom = G[0,0]*G[2,2] - G[0,2]**2
    K = R_0202 / denom

    # Volume element
    dV = st1 * st2 * dxs[0]*dxs[1]*dxs[2]*dxs[3]

    return K, dV, shape


def main():
    print("=" * 70)
    print("Mixed curvature of Hessian seam metrics on S^2 x S^2")
    print("=" * 70)

    N = 12
    gamma = 0.01
    rng = np.random.default_rng(42)

    # Test seams
    test_cases = {}
    test_cases['x1*x2'] = [1,0,0, 0,0,0, 0,0,0]
    test_cases['y1*z2'] = [0,0,0, 0,0,1, 0,0,0]
    test_cases['x1*y2+z1*z2'] = [0,1,0, 0,0,0, 0,0,1]
    test_cases['x1*x2+y1*y2'] = [1,0,0, 0,1,0, 0,0,0]  # = cos(angle)
    for trial in range(10):
        c = rng.normal(size=9)
        test_cases[f'random_{trial}'] = c.tolist()

    print(f"\ngamma = {gamma}, grid N = {N}")
    print(f"{'Seam':<20} {'K min':>12} {'K max':>12} {'int K dV':>12} {'int|K|dV':>12} {'sign_chg':>9}")
    print("-" * 75)

    n_sign_change = 0
    n_total = 0
    trim = 2

    for name, coeffs in test_cases.items():
        seam_func = make_seam_product_harmonics(coeffs)
        K, dV, shape = build_metric_and_curvature(seam_func, gamma, N)

        sl = tuple(slice(trim, -trim) for _ in range(4))
        Ki = K[sl]
        dVi = dV[sl]

        int_K = np.sum(Ki * dVi)
        int_absK = np.sum(np.abs(Ki) * dVi)
        kmin, kmax = np.min(Ki), np.max(Ki)
        sign_change = kmin < -1e-12 and kmax > 1e-12

        print(f"{name:<20} {kmin:>12.4e} {kmax:>12.4e} {int_K:>12.4e} {int_absK:>12.4e} {'YES' if sign_change else 'NO':>9}")
        n_total += 1
        if sign_change:
            n_sign_change += 1

    print(f"\nK changes sign: {n_sign_change}/{n_total}")

    # First-order cancellation test
    print("\n" + "=" * 70)
    print("First-order cancellation: K/gamma^2 as gamma -> 0")
    print("=" * 70)

    coeffs = rng.normal(size=9).tolist()
    seam_func = make_seam_product_harmonics(coeffs)
    for g in [0.1, 0.01, 0.001]:
        K, dV, shape = build_metric_and_curvature(seam_func, g, N=10)
        sl = tuple(slice(2, -2) for _ in range(4))
        Ki = K[sl]
        ratio = np.max(np.abs(Ki)) / g**2
        print(f"  gamma={g:.4f}: max|K|={np.max(np.abs(Ki)):.4e}, max|K/gamma^2|={ratio:.4e}")

    # Integral identity test
    print("\n" + "=" * 70)
    print("Integral identity: int K dV / (gamma^2 * int |K| dV)")
    print("If int Q dV = 0 survives on S^2xS^2, this ratio -> 0 as gamma -> 0")
    print("=" * 70)

    for g in [0.1, 0.01, 0.001]:
        K, dV, shape = build_metric_and_curvature(seam_func, g, N=10)
        sl = tuple(slice(2, -2) for _ in range(4))
        Ki = K[sl]
        dVi = dV[sl]
        int_K = np.sum(Ki * dVi)
        int_absK = np.sum(np.abs(Ki) * dVi)
        if int_absK > 0:
            ratio = abs(int_K) / int_absK
        else:
            ratio = 0
        print(f"  gamma={g:.4f}: int K dV = {int_K:.4e}, int|K|dV = {int_absK:.4e}, |ratio| = {ratio:.4e}")


if __name__ == '__main__':
    main()
