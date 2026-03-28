#!/usr/bin/env python3
"""
Test the Bochner approach (FEEDBACK Steps 4-5) on S^2 x S^2.

For a Hessian seam metric g = h + gamma * Hess_h(s), we test:

1. Bochner identity: int Ric(grad s, grad s) dV = int [(Delta s)^2 - |Hess^g s|^2] dV
   Verify numerically that the identity holds.

2. Seam identity: gamma * Hess^g s = g - h - gamma * A^k_ij * nabla_k s
   At critical points of s: gamma * Hess^g s = g - h

3. The key inequality from positive curvature:
   int Ric(grad s, grad s) dV >= 2*K_min * int |grad s|^2 dV

4. Compute Lambda(s,g) = int |grad^g s|^2 dV / int (Delta_g s)^2 dV
   and check whether the seam constraint forces Lambda to be bounded
   in a way that contradicts positive curvature.

APPROACH: For small gamma, we use perturbation theory:
- g ≈ h + gamma*H, so g^{ij} ≈ h^{ij} - gamma*H^{ij} + O(gamma^2)
- Delta_g s ≈ Delta_h s - gamma * H^{ij} H_{ij} + ... (schematic)
- Ric_g ≈ Ric_h + O(gamma) where Ric_h = g_{S^2} on each factor
- The seam identity: gamma * Delta_g s = tr_g(g - h) = 4 - tr_g(h) ≈ 4 - (4 - gamma*tr(H)) = gamma*tr(H) = gamma*Delta_h s
  So Delta_g s ≈ Delta_h s to leading order.

Actually, let's compute everything numerically for the full metric.
"""

import numpy as np
from test_Q_S2xS2_l2 import make_seam_general, build_curvature


def compute_bochner_quantities(seam_func, gamma, N=10):
    """
    Compute all Bochner-related quantities for the seam metric
    g = h + gamma * Hess_h(s) on S^2 x S^2.

    Returns dict with:
    - Ric_grad_grad: Ric_g(grad^g s, grad^g s) at each point
    - Delta_s_sq: (Delta_g s)^2
    - Hess_sq: |Hess^g s|^2_g
    - grad_sq: |grad^g s|^2_g
    - K_min_02: minimum mixed sectional curvature K(e_0, e_2)
    - dV: volume element
    """
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

    # Covariant Hessian H_ij = s_{,ij} - Gamma^k_ij s_{,k}
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

    # Full metric
    G = gamma * H.copy()
    G[0,0] += 1.0; G[1,1] += st1**2; G[2,2] += 1.0; G[3,3] += st2**2

    def fd(arr, axis):
        return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2*dxs[axis])

    # Metric derivatives
    dG = np.zeros((4, 4, 4) + shape)
    for k in range(4):
        for i in range(4):
            for j in range(i, 4):
                dG[i,j,k] = fd(G[i,j], k)
                dG[j,i,k] = dG[i,j,k]

    # Inverse metric
    Gf = G.reshape(4, 4, -1).transpose(2, 0, 1)
    Gif = np.linalg.inv(Gf)
    Ginv = Gif.transpose(1, 2, 0).reshape(4, 4, *shape)

    # Christoffel symbols of g
    Chr = np.zeros((4, 4, 4) + shape)
    for m in range(4):
        for i in range(4):
            for j in range(i, 4):
                val = np.zeros(shape)
                for k in range(4):
                    val += Ginv[m,k] * (dG[k,i,j] + dG[k,j,i] - dG[i,j,k])
                Chr[m,i,j] = 0.5 * val
                if j > i: Chr[m,j,i] = Chr[m,i,j]

    # Christoffel derivatives
    dChr = np.zeros((4, 4, 4, 4) + shape)
    for l in range(4):
        for m in range(4):
            for i in range(4):
                for j in range(i, 4):
                    dChr[m,i,j,l] = fd(Chr[m,i,j], l)
                    if j > i: dChr[m,j,i,l] = dChr[m,i,j,l]

    # Full Riemann tensor R^m_{bcd} (all needed components for Ricci)
    # R^m_{bcd} = dChr^m_{cd,b} - dChr^m_{bd,c} + Chr^m_{be}Chr^e_{cd} - Chr^m_{ce}Chr^e_{bd}
    # Ricci: R_{bd} = R^c_{bcd} = sum_c R^c_{bcd}
    Ric = np.zeros((4, 4) + shape)
    for b in range(4):
        for d in range(b, 4):
            val = np.zeros(shape)
            for c in range(4):
                # R^c_{bcd}
                Rc = dChr[c,c,d,b] - dChr[c,b,d,c]
                for e in range(4):
                    Rc += Chr[c,b,e]*Chr[e,c,d] - Chr[c,c,e]*Chr[e,b,d]
                val += Rc
            Ric[b,d] = val
            if d > b: Ric[d,b] = val

    # grad^g s: (grad s)^i = g^{ij} s_{,j}
    grad_s = np.zeros((4,) + shape)
    for i in range(4):
        for j in range(4):
            grad_s[i] += Ginv[i,j] * dS[j]

    # |grad^g s|^2 = g_{ij} (grad s)^i (grad s)^j = g^{ij} s_i s_j
    grad_sq = np.zeros(shape)
    for i in range(4):
        for j in range(4):
            grad_sq += Ginv[i,j] * dS[i] * dS[j]

    # Ric(grad s, grad s) = Ric_{ij} (grad s)^i (grad s)^j
    Ric_gg = np.zeros(shape)
    for i in range(4):
        for j in range(4):
            Ric_gg += Ric[i,j] * grad_s[i] * grad_s[j]

    # Hess^g s: (Hess^g s)_{ij} = s_{,ij} - Gamma^g_k_{ij} s_{,k}
    Hess_g = np.zeros((4, 4) + shape)
    for i in range(4):
        for j in range(i, 4):
            val = d2S[i][j].copy()
            for k in range(4):
                val -= Chr[k,i,j] * dS[k]
            Hess_g[i,j] = val
            if j > i: Hess_g[j,i] = val

    # Delta_g s = g^{ij} Hess^g_{ij} = tr_g(Hess^g s)
    Delta_s = np.zeros(shape)
    for i in range(4):
        for j in range(4):
            Delta_s += Ginv[i,j] * Hess_g[i,j]

    # |Hess^g s|^2_g = g^{ia} g^{jb} Hess_ij Hess_ab
    Hess_sq = np.zeros(shape)
    for i in range(4):
        for j in range(4):
            for a in range(4):
                for b in range(4):
                    Hess_sq += Ginv[i,a] * Ginv[j,b] * Hess_g[i,j] * Hess_g[a,b]

    # Mixed curvature K(e_0, e_2)
    R_0202 = np.zeros(shape)
    for m in range(4):
        Rm = dChr[m,0,2,2] - dChr[m,2,2,0]
        for p in range(4):
            Rm += Chr[m,2,p]*Chr[p,0,2] - Chr[m,0,p]*Chr[p,2,2]
        R_0202 += G[0,m] * Rm
    denom_02 = G[0,0]*G[2,2] - G[0,2]**2
    K_02 = R_0202 / denom_02

    # Volume element
    det_g = np.linalg.det(Gf).reshape(shape)
    dV = np.sqrt(np.abs(det_g)) * dxs[0]*dxs[1]*dxs[2]*dxs[3]

    # tr_g(g - h): needed for the FEEDBACK inequality
    h_bg = np.zeros((4, 4) + shape)
    h_bg[0,0] = 1.0; h_bg[1,1] = st1**2; h_bg[2,2] = 1.0; h_bg[3,3] = st2**2
    tr_g_gh = np.zeros(shape)
    for i in range(4):
        for j in range(4):
            tr_g_gh += Ginv[i,j] * (G[i,j] - h_bg[i,j])

    return {
        'Ric_gg': Ric_gg,
        'Delta_sq': Delta_s**2,
        'Hess_sq': Hess_sq,
        'grad_sq': grad_sq,
        'K_02': K_02,
        'dV': dV,
        'Delta_s': Delta_s,
        'tr_g_gh': tr_g_gh,
        'tr_g_gh_sq': tr_g_gh**2,
    }


def main():
    print("=" * 70)
    print("Bochner approach test on S^2 x S^2 (FEEDBACK Steps 4-5)")
    print("=" * 70)

    # Spherical harmonic basis functions
    f_x = lambda x,y,z: x
    f_y = lambda x,y,z: y
    f_z = lambda x,y,z: z
    f_z2 = lambda x,y,z: 3*z**2 - 1
    f_xz = lambda x,y,z: x*z
    f_yz = lambda x,y,z: y*z
    f_x2y2 = lambda x,y,z: x**2 - y**2
    f_xy = lambda x,y,z: x*y
    all_funcs = [f_x, f_y, f_z, f_z2, f_xz, f_yz, f_x2y2, f_xy]

    rng = np.random.default_rng(42)
    N = 10
    trim = 2
    sl = tuple(slice(trim, -trim) for _ in range(4))

    gammas = [0.01, 0.05, 0.1]

    print("\n--- Bochner identity verification ---")
    print("int Ric(grad s, grad s) dV = int [(Delta s)^2 - |Hess^g s|^2] dV")
    print()

    # Test with a few seams
    test_seams = {}
    test_seams['xz*yz'] = ([(f_xz, f_yz)], [1.0])
    test_seams['x*x+y*y'] = ([(f_x, f_x), (f_y, f_y)], [1.0, 1.0])
    for t in range(5):
        n_terms = rng.integers(2, 5)
        pairs = [(all_funcs[rng.integers(0, len(all_funcs))],
                  all_funcs[rng.integers(0, len(all_funcs))]) for _ in range(n_terms)]
        coeffs = rng.normal(size=n_terms).tolist()
        test_seams[f'rand_{t}'] = (pairs, coeffs)

    gamma = 0.01
    print(f"gamma = {gamma}, N = {N}")
    print(f"{'Seam':<15} {'LHS(Ric)':>12} {'RHS(Lap-Hess)':>14} {'ratio':>10}")
    print("-" * 55)

    for name, (pairs, coeffs) in test_seams.items():
        sf = make_seam_general(pairs, coeffs)
        q = compute_bochner_quantities(sf, gamma, N)

        lhs = np.sum(q['Ric_gg'][sl] * q['dV'][sl])
        rhs = np.sum((q['Delta_sq'][sl] - q['Hess_sq'][sl]) * q['dV'][sl])
        rat = lhs / rhs if abs(rhs) > 1e-20 else float('nan')
        print(f"{name:<15} {lhs:>12.4e} {rhs:>14.4e} {rat:>10.4f}")

    print("\n--- Key quantities for the inequality ---")
    print("If sec > 0: int Ric(grad s, grad s) >= 2*K_min * int |grad s|^2")
    print("Combined with Bochner: 2*K_min * int |grad s|^2 <= int [(Delta s)^2 - |Hess s|^2]")
    print()
    print(f"{'Seam':<15} {'int|grad|^2':>12} {'int(Delta)^2':>13} {'int|Hess|^2':>12} {'Lambda':>10} {'K02 min':>12}")
    print("-" * 75)

    for name, (pairs, coeffs) in test_seams.items():
        sf = make_seam_general(pairs, coeffs)
        q = compute_bochner_quantities(sf, gamma, N)

        int_grad = np.sum(q['grad_sq'][sl] * q['dV'][sl])
        int_delta = np.sum(q['Delta_sq'][sl] * q['dV'][sl])
        int_hess = np.sum(q['Hess_sq'][sl] * q['dV'][sl])
        k_min = np.min(q['K_02'][sl])
        Lambda = int_grad / int_delta if int_delta > 1e-20 else float('nan')

        print(f"{name:<15} {int_grad:>12.4e} {int_delta:>13.4e} {int_hess:>12.4e} {Lambda:>10.4f} {k_min:>12.4e}")

    print("\n--- Seam identity check ---")
    print("gamma * Delta_g s ≈ tr_g(g - h)  (FEEDBACK Corollary 1)")
    print()
    for name, (pairs, coeffs) in list(test_seams.items())[:3]:
        sf = make_seam_general(pairs, coeffs)
        q = compute_bochner_quantities(sf, gamma, N)
        lhs_vals = gamma * q['Delta_s'][sl]
        rhs_vals = q['tr_g_gh'][sl]
        rel_err = np.max(np.abs(lhs_vals - rhs_vals)) / (np.max(np.abs(rhs_vals)) + 1e-20)
        print(f"  {name}: max|gamma*Delta_s - tr_g(g-h)| / |tr_g(g-h)| = {rel_err:.4e}")

    print("\n--- FEEDBACK inequality (*) test ---")
    print("If sec>0: int |grad s|^2 <= (3/(8*K_min*gamma^2)) * int (tr_g(g-h))^2 dV")
    print("Check feasibility: what K_min is required?")
    print()
    print(f"{'Seam':<15} {'int|grad|^2':>12} {'int(tr)^2':>12} {'K_min_req':>12} {'K02_actual':>12}")
    print("-" * 65)

    for name, (pairs, coeffs) in test_seams.items():
        sf = make_seam_general(pairs, coeffs)
        q = compute_bochner_quantities(sf, gamma, N)

        int_grad = np.sum(q['grad_sq'][sl] * q['dV'][sl])
        int_tr_sq = np.sum(q['tr_g_gh_sq'][sl] * q['dV'][sl])
        k02_min = np.min(q['K_02'][sl])

        # From inequality: K_min <= 3*int(tr^2) / (8*gamma^2 * int|grad|^2)
        if int_grad > 1e-20:
            K_min_req = 3 * int_tr_sq / (8 * gamma**2 * int_grad)
        else:
            K_min_req = float('inf')

        print(f"{name:<15} {int_grad:>12.4e} {int_tr_sq:>12.4e} {K_min_req:>12.4e} {k02_min:>12.4e}")

    print("\n--- Scaling with gamma ---")
    print("Check if the required K_min bound changes with gamma")
    print()
    sf = make_seam_general([(f_xz, f_yz), (f_x, f_z2)], [1.0, -0.5])
    print(f"{'gamma':>8} {'int|grad|^2':>12} {'int(tr)^2':>12} {'K_req':>12} {'K02 min':>12} {'K02 max':>12}")
    print("-" * 70)
    for g in [0.2, 0.1, 0.05, 0.01, 0.005]:
        q = compute_bochner_quantities(sf, g, N)
        int_grad = np.sum(q['grad_sq'][sl] * q['dV'][sl])
        int_tr_sq = np.sum(q['tr_g_gh_sq'][sl] * q['dV'][sl])
        k02_min = np.min(q['K_02'][sl])
        k02_max = np.max(q['K_02'][sl])
        K_req = 3 * int_tr_sq / (8 * g**2 * int_grad) if int_grad > 1e-20 else float('inf')
        print(f"{g:>8.4f} {int_grad:>12.4e} {int_tr_sq:>12.4e} {K_req:>12.4e} {k02_min:>12.4e} {k02_max:>12.4e}")


if __name__ == '__main__':
    main()
