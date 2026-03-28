"""
Verify the FEEDBACK's claimed proof: at the maximum of
Phi = log(lambda_1 * lambda_2 - eta^2) = log det g_orb,
does the Gaussian curvature K of the orbit metric have a definite sign?

The claim: Hess(Phi) <= 0 at the max implies K <= 0 there.

We test this on random equivariant seams s(theta1, theta2).
"""
import numpy as np
from itertools import product as iprod

def compute_orbit_metric_and_K(s_func, th1, th2, gamma=0.1, h=1e-5):
    """
    For an equivariant seam s(theta1, theta2) on S^2 x S^2,
    compute the orbit metric components and Gaussian curvature
    at (th1, th2) using finite differences.
    
    The seam metric diagonal blocks are:
      g_11 = 1 + gamma * d^2 s / d theta1^2   (conformal factor on factor 1)
      g_22 = 1 + gamma * d^2 s / d theta2^2   (conformal factor on factor 2)
    Wait -- for a general equivariant seam, the Hessian on S^2 has both 
    diagonal and off-diagonal parts. But restricted to the orbit 
    (theta1, theta2), the metric is:
      lambda_1 = 1 + gamma * (d^2s/dtheta1^2)  [the theta1-theta1 component]
      lambda_2 = 1 + gamma * (d^2s/dtheta2^2)  [the theta2-theta2 component]  
      eta = gamma * (d^2s / dtheta1 dtheta2)    [the cross component]
    
    Actually for the Hessian metric g = h + gamma * nabla^2 s on S^2 x S^2,
    in coordinates (theta1, phi1, theta2, phi2), the orbit metric (fixing phi's)
    is the 2x2 submatrix in (theta1, theta2):
      lambda_1 = 1 + gamma * s_{,11}  (covariant Hessian component)
      lambda_2 = 1 + gamma * s_{,22}  
      eta = gamma * s_{,12}           (mixed partial = mixed Hessian for product)
    
    For a product seam s = f(theta1)*g(theta2), s_{,12} = f'*g' so eta != 0.
    """
    # Compute s and its derivatives via FD
    s00 = s_func(th1, th2)
    sp0 = s_func(th1 + h, th2)
    sm0 = s_func(th1 - h, th2)
    s0p = s_func(th1, th2 + h)
    s0m = s_func(th1, th2 - h)
    spp = s_func(th1 + h, th2 + h)
    spm = s_func(th1 + h, th2 - h)
    smp = s_func(th1 - h, th2 + h)
    smm = s_func(th1 - h, th2 - h)
    
    s11 = (sp0 - 2*s00 + sm0) / h**2
    s22 = (s0p - 2*s00 + s0m) / h**2
    s12 = (spp - spm - smp + smm) / (4*h**2)
    
    lam1 = 1 + gamma * s11
    lam2 = 1 + gamma * s22
    eta = gamma * s12
    
    return lam1, lam2, eta


def compute_K_brioschi(s_func, th1, th2, gamma=0.1, h=1e-5):
    """
    Compute Gaussian curvature of the 2D orbit metric 
    g_orb = [[lam1, eta], [eta, lam2]] using finite differences
    of metrics components, via the standard Brioschi formula.
    """
    # Get metric components on a grid
    def get_metric(t1, t2):
        return compute_orbit_metric_and_K(s_func, t1, t2, gamma, h=1e-6)
    
    lam1, lam2, eta = get_metric(th1, th2)
    det = lam1 * lam2 - eta**2
    if det <= 0:
        return None, lam1, lam2, eta, det
    
    # For Gaussian curvature of a general 2D metric, use Christoffel symbols
    # and Riemann tensor. This is the standard computation.
    dh = 1e-5
    
    # Metric at nearby points
    l1p, l2p, ep = get_metric(th1 + dh, th2)
    l1m, l2m, em = get_metric(th1 - dh, th2)
    l1_0p, l2_0p, e_0p = get_metric(th1, th2 + dh)
    l1_0m, l2_0m, e_0m = get_metric(th1, th2 - dh)
    
    # First derivatives of metric components
    # g = [[g11, g12], [g12, g22]] = [[lam1, eta], [eta, lam2]]
    g11_1 = (l1p - l1m) / (2*dh)   # d(lam1)/dtheta1
    g11_2 = (l1_0p - l1_0m) / (2*dh)  # d(lam1)/dtheta2
    g22_1 = (l2p - l2m) / (2*dh)
    g22_2 = (l2_0p - l2_0m) / (2*dh)
    g12_1 = (ep - em) / (2*dh)
    g12_2 = (e_0p - e_0m) / (2*dh)
    
    # Second derivatives
    l1pp, l2pp, epp = get_metric(th1 + dh, th2 + dh)
    l1pm, l2pm, epm = get_metric(th1 + dh, th2 - dh)
    l1mp, l2mp, emp = get_metric(th1 - dh, th2 + dh)
    l1mm, l2mm, emm = get_metric(th1 - dh, th2 - dh)
    
    g11_11 = (l1p - 2*lam1 + l1m) / dh**2
    g11_22 = (l1_0p - 2*lam1 + l1_0m) / dh**2
    g22_11 = (l2p - 2*lam2 + l2m) / dh**2
    g22_22 = (l2_0p - 2*lam2 + l2_0m) / dh**2
    g12_12 = (epp - epm - emp + emm) / (4*dh**2)
    g11_12 = (l1pp - l1pm - l1mp + l1mm) / (4*dh**2)
    g22_12 = (l2pp - l2pm - l2mp + l2mm) / (4*dh**2)
    
    # Brioschi formula for Gaussian curvature of 2D metric
    # K = (R_1212) / det(g)
    # Using the standard formula:
    # K = [g] * [ ... ] where the formula involves g_ij and their derivatives
    #
    # Standard result: for 2D metric g_ij,
    # K = (1/det) * [Gamma terms]
    #
    # Let's use the explicit formula via Christoffel symbols
    # Gamma^1_11 = (g22*g11_1 - 2*g12*g12_1 + g12*g11_2) / (2*det) -- wrong
    # Let me use the correct Christoffel symbols for 2D
    
    # g^{ij}: inverse metric
    gi11 = lam2 / det
    gi22 = lam1 / det
    gi12 = -eta / det
    
    # Christoffel symbols Gamma^k_{ij}
    # Gamma^1_{11} = 0.5 * g^{1k} * (g_{k1,1} + g_{k1,1} - g_{11,k})
    #              = 0.5 * (g^{11}*g_{11,1} + g^{12}*(2*g_{12,1} - g_{11,2}))
    G1_11 = 0.5 * (gi11 * g11_1 + gi12 * (2*g12_1 - g11_2))
    G1_12 = 0.5 * (gi11 * g11_2 + gi12 * g22_1)
    G1_22 = 0.5 * (gi11 * (2*g12_2 - g22_1) + gi12 * g22_2)
    G2_11 = 0.5 * (gi12 * g11_1 + gi22 * (2*g12_1 - g11_2))
    G2_12 = 0.5 * (gi12 * g11_2 + gi22 * g22_1)
    G2_22 = 0.5 * (gi12 * (2*g12_2 - g22_1) + gi22 * g22_2)
    
    # Riemann tensor R^1_{212} = Gamma^1_{22,1} - Gamma^1_{12,2} + Gamma^1_{1k}Gamma^k_{22} - Gamma^1_{2k}Gamma^k_{12}
    # But for numerical stability, use the direct formula:
    # K = (R_{1212}) / det(g)
    # R_{1212} = g_{1k} R^k_{212}
    
    # Actually, for a 2D metric, the simplest formula is:
    # K = -1/(2*sqrt(det)) * [d/d1(1/sqrt(det) * d(g22)/d1 - ...) + ...]
    # 
    # Let me just use the Brioschi formula directly:
    # K = 1/(4*det^2) * | g11  g11_1  g11_2 |     1/(2*det) * | 0      g11_2  g22_1 |
    #                    | g11_1 g11_11 g11_12|  -              | g11_2  g11_22 g12_22|  
    #                    | g11_2 g11_12 g11_22|                 | g22_1  g12_12 g22_22|  -- not quite right
    #
    # Actually the Brioschi formula for the general (non-diagonal) 2D case is complex.
    # Let me use the Christoffel approach properly.
    
    # K = (Gamma^1_{22,1} - Gamma^1_{12,2} + Gamma^1_{1a}*Gamma^a_{22} - Gamma^1_{2a}*Gamma^a_{12}) / det ... no
    # Actually K * det = R_{1212} = g_{1a} R^a_{212}
    
    # Simpler: just compute R^1_{212} numerically by differentiating Christoffel symbols
    # But for double FD this gets noisy. Instead:
    
    # Use the formula for 2D Gaussian curvature via the metric determinant and 
    # Christoffel symbols:
    # K = (1/det) * (partial_1 Gamma^2_{12} - partial_2 Gamma^2_{11} + Gamma^1_{12}*Gamma^2_{11} - Gamma^1_{11}*Gamma^2_{12} + Gamma^2_{12}*Gamma^2_{12} - Gamma^2_{11}*Gamma^2_{22})
    # Hmm, this is getting messy. Let me just compute K via the log-det formula.
    
    # For 2D, K = -1/(2*sqrt(det)) * laplacian_g(log(sqrt(det)))... no that's only for conformal metrics.
    
    # OK let's use the most robust approach: the Gauss formula
    # K = 1/(2*det) * [ -g22_11 - g11_22 + 2*g12_12 + ... first-order terms ]
    #
    # The exact formula (Brioschi for general 2D metric):
    # 4*det^2 * K = 
    #   det * (2*g12_12 - g11_22 - g22_11)
    #   + g22 * (g11_1 * g22_1 + g11_2^2 - g11_1 * ... ) -- this gets complicated
    #
    # Let me just fall back to the Riemann tensor via Christoffel FD.
    
    # Actually, the cleanest formula: compute Gamma's at shifted points and differentiate.
    # But let me just compute K from the inverse-metric approach.
    
    # For a general 2D metric g_ij, the Gaussian curvature is:
    # K = R_{1212} / det(g)
    # where R_{ijkl} = g_{il,jk} - g_{ik,jl} - g_{jl,ik} + g_{jk,il} 
    #                  + g^{mn}(Gamma_{nil} Gamma_{mjk} - Gamma_{nik} Gamma_{mjl})
    # ... too complex. Let me use the totally standard formula.
    
    # 2D Gaussian curvature explicit formula (Weinstein, do Carmo):
    # det(g) * K = Gamma^1_{12,1} - Gamma^1_{11,2} + Gamma^k_{12}*Gamma^1_{k1} - Gamma^k_{11}*Gamma^1_{k2}
    # with k summed over 1,2.
    
    # Actually the standard relation is:
    # R^1_{212} = d_1(Gamma^1_{22}) - d_2(Gamma^1_{12}) + Gamma^1_{1m}Gamma^m_{22} - Gamma^1_{2m}*Gamma^m_{12}
    # K = R_{1212}/det = g_{11}*R^1_{212}/det (since R_{1212} = g_{1a}*R^a_{212} and only R^1_{212} contributes in 2D)
    # Wait, R_{1212} = g_{11}R^1_{212} + g_{12}R^2_{212}
    # And R^2_{212} = d_1(Gamma^2_{22}) - d_2(Gamma^2_{12}) + Gamma^2_{1m}Gamma^m_{22} - Gamma^2_{2m}*Gamma^m_{12}
    
    # This is getting unwieldy for inline code. Let me just use a robust numerical K computation.
    # Use the formula: K = 1/(2*det) * (stuff involving first and second partials of g_ij)
    
    # Actually, the Brioschi-style formula for general 2D metric is:
    # K = 1/(det^2) * [ det * (g12_12 - g11_22/2 - g22_11/2)
    #   + (g11_2 * g22_2 + g22_1 * g11_1)/4 * det
    #   - ... ]
    # This is a mess. Let me just compute numerically from the 4D Riemann tensor approach
    # that already works in the existing scripts.
    pass
    
    # SIMPLIFICATION: for the purpose of this test, we only need K at the maximum
    # of Phi. Let me compute K via the formula K = R_{0202} / (g_{00}g_{22} - g_{02}^2)
    # using the 4D approach from test_Q_S2xS2.py.
    # But that's overkill. Let me just use a clean 2D Gaussian curvature.
    
    # Final approach: use the coordinate-free formula via area element.
    # sqrt(det) * K = -d/d1[(g22*g11_1 - g12*g11_2 + g12*g22_1 - ?) / (2*sqrt(det))]
    # ... I realize I'm going in circles. Let me just do it the Christoffel way cleanly.
    
    R1_212 = 0  # Will compute via FD of Christoffel symbols
    R2_212 = 0
    
    # Get Gamma at shifted points
    def christoffel_at(t1, t2):
        dh2 = 1e-5
        l1c, l2c, ec = get_metric(t1, t2)
        l1p2, l2p2, ep2 = get_metric(t1 + dh2, t2)
        l1m2, l2m2, em2 = get_metric(t1 - dh2, t2)
        l1_p2, l2_p2, e_p2 = get_metric(t1, t2 + dh2)
        l1_m2, l2_m2, e_m2 = get_metric(t1, t2 - dh2)
        
        d = l1c * l2c - ec * ec
        if d <= 0:
            return None
        
        gi11 = l2c / d
        gi22 = l1c / d
        gi12 = -ec / d
        
        dg11_1 = (l1p2 - l1m2) / (2*dh2)
        dg11_2 = (l1_p2 - l1_m2) / (2*dh2)
        dg22_1 = (l2p2 - l2m2) / (2*dh2)
        dg22_2 = (l2_p2 - l2_m2) / (2*dh2)
        dg12_1 = (ep2 - em2) / (2*dh2)
        dg12_2 = (e_p2 - e_m2) / (2*dh2)
        
        G1_11 = 0.5 * (gi11 * dg11_1 + gi12 * (2*dg12_1 - dg11_2))
        G1_12 = 0.5 * (gi11 * dg11_2 + gi12 * dg22_1)
        G1_22 = 0.5 * (gi11 * (2*dg12_2 - dg22_1) + gi12 * dg22_2)
        G2_11 = 0.5 * (gi12 * dg11_1 + gi22 * (2*dg12_1 - dg11_2))
        G2_12 = 0.5 * (gi12 * dg11_2 + gi22 * dg22_1)
        G2_22 = 0.5 * (gi12 * (2*dg12_2 - dg22_1) + gi22 * dg22_2)
        
        return (G1_11, G1_12, G1_22, G2_11, G2_12, G2_22)
    
    c0 = christoffel_at(th1, th2)
    cp = christoffel_at(th1 + dh, th2)
    cm = christoffel_at(th1 - dh, th2)
    c0p = christoffel_at(th1, th2 + dh)
    c0m = christoffel_at(th1, th2 - dh)
    
    if any(x is None for x in [c0, cp, cm, c0p, c0m]):
        return None, lam1, lam2, eta, det
    
    G1_11, G1_12, G1_22, G2_11, G2_12, G2_22 = c0
    
    # d_1(Gamma^1_{22})
    dG1_22_d1 = (cp[2] - cm[2]) / (2*dh)
    # d_2(Gamma^1_{12})  
    dG1_12_d2 = (c0p[1] - c0m[1]) / (2*dh)
    # d_1(Gamma^2_{22})
    dG2_22_d1 = (cp[5] - cm[5]) / (2*dh)
    # d_2(Gamma^2_{12})
    dG2_12_d2 = (c0p[4] - c0m[4]) / (2*dh)
    
    # R^1_{212} = d_1(G1_22) - d_2(G1_12) + G1_1m*G_m_22 - G1_2m*Gm_12
    R1_212 = (dG1_22_d1 - dG1_12_d2 
              + G1_11*G1_22 + G1_12*G2_22 
              - G1_12*G1_12 - G1_22*G2_12)
    
    R2_212 = (dG2_22_d1 - dG2_12_d2
              + G2_11*G1_22 + G2_12*G2_22
              - G2_12*G1_12 - G2_22*G2_12)
    
    # R_{1212} = g_{11}*R^1_{212} + g_{12}*R^2_{212}
    R_1212 = lam1 * R1_212 + eta * R2_212
    
    K = R_1212 / det
    
    return K, lam1, lam2, eta, det


def test_logdet_maxprinciple():
    """
    Test the FEEDBACK claim: at the maximum of Phi = log(lam1*lam2 - eta^2),
    the Gaussian curvature K <= 0.
    """
    np.random.seed(42)
    gamma = 0.1
    N_grid = 60
    th1_vals = np.linspace(0.15, np.pi - 0.15, N_grid)
    th2_vals = np.linspace(0.15, np.pi - 0.15, N_grid)
    
    n_tests = 0
    n_violations = 0
    violation_details = []
    
    # Test various seams
    for trial in range(100):
        # Random seam: sum of product harmonics
        # s = sum a_ij * f_i(theta1) * g_j(theta2)
        # where f, g are from {cos(theta), sin(theta), cos(2*theta), sin(2*theta), ...}
        n_modes = np.random.randint(1, 5)
        coeffs = np.random.randn(n_modes)
        ells1 = np.random.randint(1, 4, size=n_modes)
        ells2 = np.random.randint(1, 4, size=n_modes)
        
        def make_seam(cs, e1s, e2s):
            def s_func(t1, t2):
                val = 0.0
                for c, l1, l2 in zip(cs, e1s, e2s):
                    # Use associated Legendre-like functions (simplified)
                    f1 = np.cos(l1 * t1) if np.random.random() > 0.5 else np.sin(l1 * t1)
                    f2 = np.cos(l2 * t2) if np.random.random() > 0.5 else np.sin(l2 * t2)
                    val += c * f1 * f2
                return val
            return s_func
        
        # Deterministic seam construction
        phases1 = np.random.choice([0, 1], size=n_modes)
        phases2 = np.random.choice([0, 1], size=n_modes)
        
        def s_func(t1, t2, _c=coeffs.copy(), _e1=ells1.copy(), _e2=ells2.copy(), 
                   _p1=phases1.copy(), _p2=phases2.copy()):
            val = 0.0
            for c, l1, l2, p1, p2 in zip(_c, _e1, _e2, _p1, _p2):
                f1 = np.cos(l1 * t1) if p1 == 0 else np.sin(l1 * t1)
                f2 = np.cos(l2 * t2) if p2 == 0 else np.sin(l2 * t2)
                val += c * f1 * f2
            return val
        
        # Find maximum of Phi = log(lam1*lam2 - eta^2)
        best_phi = -np.inf
        best_t1, best_t2 = None, None
        
        for t1 in th1_vals:
            for t2 in th2_vals:
                l1, l2, e = compute_orbit_metric_and_K(s_func, t1, t2, gamma)
                d = l1 * l2 - e**2
                if d > 0:
                    phi = np.log(d)
                    if phi > best_phi:
                        best_phi = phi
                        best_t1, best_t2 = t1, t2
        
        if best_t1 is None:
            continue
        
        # Refine maximum with finer grid around best point
        dt = (th1_vals[1] - th1_vals[0])
        for t1 in np.linspace(best_t1 - dt, best_t1 + dt, 20):
            for t2 in np.linspace(best_t2 - dt, best_t2 + dt, 20):
                if t1 < 0.05 or t1 > np.pi - 0.05 or t2 < 0.05 or t2 > np.pi - 0.05:
                    continue
                l1, l2, e = compute_orbit_metric_and_K(s_func, t1, t2, gamma)
                d = l1 * l2 - e**2
                if d > 0:
                    phi = np.log(d)
                    if phi > best_phi:
                        best_phi = phi
                        best_t1, best_t2 = t1, t2
        
        # Compute K at the max of Phi
        K, l1, l2, e, d = compute_K_brioschi(s_func, best_t1, best_t2, gamma)
        
        if K is None:
            continue
            
        n_tests += 1
        if K > 1e-6:  # K > 0 at max of Phi => violation
            n_violations += 1
            violation_details.append({
                'trial': trial,
                'K': K,
                'Phi': best_phi,
                'th1': best_t1,
                'th2': best_t2,
                'lam1': l1,
                'lam2': l2,
                'eta': e,
            })
    
    print(f"\n{'='*60}")
    print(f"FEEDBACK claim: K <= 0 at max of Phi = log(det g_orb)")
    print(f"{'='*60}")
    print(f"Tests run: {n_tests}")
    print(f"Violations (K > 0 at max Phi): {n_violations}")
    print(f"Success rate: {n_tests - n_violations}/{n_tests}")
    
    if n_violations > 0:
        print(f"\nFIRST 10 VIOLATIONS:")
        for v in violation_details[:10]:
            print(f"  trial={v['trial']}: K={v['K']:.6e}, Phi={v['Phi']:.4f}, "
                  f"th=({v['th1']:.3f},{v['th2']:.3f}), "
                  f"lam=({v['lam1']:.4f},{v['lam2']:.4f}), eta={v['eta']:.4f}")
    
    if n_violations > 0:
        print(f"\nCONCLUSION: FEEDBACK's claimed proof is WRONG.")
        print(f"  K > 0 at the max of Phi = log(det g_orb) in {n_violations}/{n_tests} cases.")
    else:
        print(f"\nCONCLUSION: No violations found (but this doesn't prove the claim).")

if __name__ == "__main__":
    test_logdet_maxprinciple()
