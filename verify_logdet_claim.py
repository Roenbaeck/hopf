"""
Verify the FEEDBACK's claimed proof:
  "At the maximum of Phi = log(lam1*lam2 - eta^2) = log det g_orb,
   the Gaussian curvature K <= 0."

This was already partially tested (TODO.md: "Psi-argument, 441/593 violations")
but let's reproduce with the exact formulation from FEEDBACK, using the
general (non-diagonal) orbit metric from S^2 x S^2 seam metrics.

Approach: use the same compute_mixed_curvature infrastructure on S^2 x S^2.
For equivariant seams s = s(theta1, theta2), the orbit metric is the 2x2
block g_orb = [[g_11, g_12], [g_12, g_22]] in the (theta1, theta2) subspace.
The Gaussian curvature of g_orb equals the mixed sectional curvature K_mix.

For a Hessian seam g = h + gamma * nabla^2 s on round S^2 x S^2:
  g_11 = 1 + gamma * s_{;11}  (covariant Hessian on factor 1)
  g_22 = 1 + gamma * s_{;22}  (covariant Hessian on factor 2)
  g_12 = gamma * s_{;12} = gamma * s_{,12} (mixed partial = covariant for product)

For a product-harmonic seam s = f(theta1)*g(theta2):
  s_{;11} = f''*g  but on S^2: nabla^2_{theta,theta} f = f'' (in theta coords)
  Actually for the round metric ds^2 = d theta^2 + sin^2 theta d phi^2,
  the (theta,theta) component of nabla^2 f is just f'' (since Gamma^theta_{theta,theta} = 0).
  The mixed partial is s_{,12} = f' * g'.

So:
  lam1 = 1 + gamma * f'' * g
  lam2 = 1 + gamma * f * g''
  eta  = gamma * f' * g'

We compute K of the 2x2 metric [[lam1, eta], [eta, lam2]] via FD,
then check sign at the max of Phi = log(lam1*lam2 - eta^2).
"""
import numpy as np

def test_logdet_claim():
    np.random.seed(42)
    N = 200
    theta = np.linspace(0.05, np.pi - 0.05, N)
    dt = theta[1] - theta[0]
    T1, T2 = np.meshgrid(theta, theta, indexing='ij')
    
    gamma = 0.15
    n_valid = 0
    n_violations = 0  # K > 0 at max of Phi
    
    for trial in range(300):
        # Generate random product-harmonic seam s = f(theta1) * g(theta2)
        # f, g are random linear combinations of cos(k*theta), sin(k*theta)
        n_f = np.random.randint(1, 4)
        n_g = np.random.randint(1, 4)
        
        cf = np.random.randn(n_f) * 0.5
        kf = np.random.randint(1, 5, size=n_f)
        pf = np.random.choice([0, 1], size=n_f)
        
        cg = np.random.randn(n_g) * 0.5
        kg = np.random.randint(1, 5, size=n_g)
        pg = np.random.choice([0, 1], size=n_g)
        
        # Compute f, f', f'' on theta1 grid
        f = np.zeros(N)
        fp = np.zeros(N)
        fpp = np.zeros(N)
        for c, k, p in zip(cf, kf, pf):
            if p == 0:
                f += c * np.cos(k * theta)
                fp += -c * k * np.sin(k * theta)
                fpp += -c * k**2 * np.cos(k * theta)
            else:
                f += c * np.sin(k * theta)
                fp += c * k * np.cos(k * theta)
                fpp += -c * k**2 * np.sin(k * theta)
        
        g_arr = np.zeros(N)
        gp = np.zeros(N)
        gpp = np.zeros(N)
        for c, k, p in zip(cg, kg, pg):
            if p == 0:
                g_arr += c * np.cos(k * theta)
                gp += -c * k * np.sin(k * theta)
                gpp += -c * k**2 * np.cos(k * theta)
            else:
                g_arr += c * np.sin(k * theta)
                gp += c * k * np.cos(k * theta)
                gpp += -c * k**2 * np.sin(k * theta)
        
        # Build 2D arrays: f(theta1), g(theta2), etc.
        F = f[:, None] * np.ones(N)[None, :]
        FP = fp[:, None] * np.ones(N)[None, :]
        FPP = fpp[:, None] * np.ones(N)[None, :]
        G = np.ones(N)[:, None] * g_arr[None, :]
        GP = np.ones(N)[:, None] * gp[None, :]
        GPP = np.ones(N)[:, None] * gpp[None, :]
        
        # Orbit metric components
        lam1 = 1 + gamma * FPP * G   # 1 + gamma * f'' * g
        lam2 = 1 + gamma * F * GPP   # 1 + gamma * f * g''
        eta  = gamma * FP * GP       # gamma * f' * g'
        
        det_g = lam1 * lam2 - eta**2
        
        # Check metric is positive definite everywhere
        if np.min(lam1) <= 0 or np.min(det_g) <= 0:
            continue
        
        # Compute Gaussian curvature K of [[lam1, eta], [eta, lam2]]
        # via the standard Brioschi formula for general 2D metric.
        #
        # For g = [[E, F], [F, G]], the Gaussian curvature is:
        # K = (1 / (2*det)) * [ ... involving E, F, G and their partials ... ]
        #
        # Use the formula from do Carmo / Brioschi:
        # K * det^2 = det * (E_22 + G_11 - 2*F_12) / 2
        #           + (E_1*G_1 + E_2^2)/4 * ... 
        # Actually this is complex. Let's use the Christoffel approach.
        #
        # Simpler: compute via the formula
        # K = -1/(2*sqrt(det)) * [ d/d1((2*F_2 - G_1)/(2*sqrt(det))) 
        #                        + d/d2((2*F_1 - E_2)/(2*sqrt(det))) ]  -- Gauss's formula
        # Wait, that's for orthogonal coordinates only (F=0).
        #
        # For general 2D metric, the cleanest approach is:
        # Use the Liouville formula / connection coefficients.
        # 
        # Let me use the explicit formula:
        # 2*det^2 * K = 
        #   det * (2*F_12 - E_22 - G_11) 
        #   + (2*E_2*F_2 - E_2*G_1 - 2*F_1*F_2 + ... )
        # This is getting messy. Numeric approach via log(det).
        
        # Alternative: For a diagonal metric (eta=0), we'd use Brioschi.
        # For general metric, compute numerically via Christoffel symbols.
        
        # Let's compute K via the standard formula using finite differences
        # of the metric components. We'll compute Christoffel symbols at each 
        # grid point, then compute R_{1212}.
        
        # Metric derivatives via FD
        E = lam1   # g_{11}
        F_m = eta   # g_{12} 
        G_m = lam2  # g_{22}
        
        # First derivatives
        E_1 = np.gradient(E, dt, axis=0)
        E_2 = np.gradient(E, dt, axis=1)
        F_1 = np.gradient(F_m, dt, axis=0)
        F_2 = np.gradient(F_m, dt, axis=1)
        G_1 = np.gradient(G_m, dt, axis=0)
        G_2 = np.gradient(G_m, dt, axis=1)
        
        # Second derivatives
        E_11 = np.gradient(E_1, dt, axis=0)
        E_22 = np.gradient(E_2, dt, axis=1)
        E_12 = np.gradient(E_1, dt, axis=1)
        F_11 = np.gradient(F_1, dt, axis=0)
        F_12 = np.gradient(F_1, dt, axis=1)
        F_22 = np.gradient(F_2, dt, axis=1)
        G_11 = np.gradient(G_1, dt, axis=0)
        G_22 = np.gradient(G_2, dt, axis=1)
        G_12 = np.gradient(G_1, dt, axis=1)
        
        # Brioschi formula for general 2D metric:
        # K = 1/det * [R_{1212}]
        # where
        # R_{1212} = -0.5*(E_22 + G_11 - 2*F_12) + 
        #            (1/det)*[...Christoffel product terms...]
        #
        # The exact formula (from e.g. Kreyszig "Differential Geometry"):
        # R_{1212} = F_12 - 0.5*E_22 - 0.5*G_11 + det * (Gamma terms)
        # 
        # Actually, the cleanest exact formula from the metric determinant:
        # 4*det^2*K =  det*(2*F_12 - E_22 - G_11) 
        #            + E*(E_2*G_2 - 2*F_2*G_1 + G_1^2) / 4    -- no this isn't right either
        #
        # Let me just use Christoffel symbols and the Riemann tensor formula.
        
        # Inverse metric  
        gi11 = G_m / det_g
        gi22 = E / det_g
        gi12 = -F_m / det_g
        
        # Christoffel Gamma^k_{ij}
        # Gamma^1_{11} = 0.5*(gi11*E_1 + gi12*(2*F_1 - E_2))
        G1_11 = 0.5*(gi11*E_1 + gi12*(2*F_1 - E_2))
        G1_12 = 0.5*(gi11*E_2 + gi12*G_1)
        G1_22 = 0.5*(gi11*(2*F_2 - G_1) + gi12*G_2)
        G2_11 = 0.5*(gi12*E_1 + gi22*(2*F_1 - E_2))
        G2_12 = 0.5*(gi12*E_2 + gi22*G_1)
        G2_22 = 0.5*(gi12*(2*F_2 - G_1) + gi22*G_2)
        
        # R^1_{212} = d_1(G1_22) - d_2(G1_12) + G1_{1m}*Gm_{22} - G1_{2m}*Gm_{12}
        dG1_22_d1 = np.gradient(G1_22, dt, axis=0)
        dG1_12_d2 = np.gradient(G1_12, dt, axis=1)
        dG2_22_d1 = np.gradient(G2_22, dt, axis=0)
        dG2_12_d2 = np.gradient(G2_12, dt, axis=1)
        
        R1_212 = (dG1_22_d1 - dG1_12_d2 
                  + G1_11*G1_22 + G1_12*G2_22 
                  - G1_12*G1_12 - G1_22*G2_12)
        
        R2_212 = (dG2_22_d1 - dG2_12_d2
                  + G2_11*G1_22 + G2_12*G2_22
                  - G2_12*G1_12 - G2_22*G2_12)
        
        R_1212 = E * R1_212 + F_m * R2_212
        K = R_1212 / det_g
        
        # Work in interior to avoid boundary FD artifacts
        m = 15
        K_int = K[m:-m, m:-m]
        det_int = det_g[m:-m, m:-m]
        
        # Find maximum of Phi = log(det_g) in interior
        Phi = np.log(det_int)
        idx_max = np.unravel_index(np.argmax(Phi), Phi.shape)
        
        K_at_max = K_int[idx_max]
        
        n_valid += 1
        if K_at_max > 1e-4:  # K > 0 at max of Phi
            n_violations += 1
        
        if n_valid <= 5 or (n_valid % 50 == 0):
            print(f"  trial {trial}: K_at_max_Phi = {K_at_max:+.6e}, "
                  f"max_Phi = {Phi[idx_max]:.4f}, "
                  f"K_range = [{np.min(K_int):.4e}, {np.max(K_int):.4e}]")
    
    print(f"\n{'='*60}")
    print(f"FEEDBACK claim: K <= 0 at max of Phi = log(det g_orb)")
    print(f"{'='*60}")
    print(f"Valid tests: {n_valid}")
    print(f"Violations (K > 0 at max Phi): {n_violations}/{n_valid}")
    print(f"Violation rate: {n_violations/n_valid*100:.1f}%")
    
    if n_violations > 0:
        print(f"\n*** CONCLUSION: FEEDBACK's claimed proof is WRONG. ***")
        print(f"    K > 0 at the max of log(det g_orb) in {n_violations}/{n_valid} cases.")
        print(f"    (Consistent with TODO.md: Psi-argument had 441/593 violations)")
    else:
        print(f"\nNo violations found.")

if __name__ == "__main__":
    test_logdet_claim()
