#!/usr/bin/env python3
"""
Test Pathway 2: Hessian seam metrics on T^2.

For a 2D metric g = A dx^2 + B dy^2 on the 2-torus,
where A = 1 + gamma * s_xx and B = 1 + gamma * s_yy,
the Gaussian curvature involves 4th derivatives of s.

Key structural identity: A_yy = B_xx = gamma * s_xxyy.

Strategy: at the maximum of Phi = s_xy on T^2,
  Phi_x = s_xxy = 0  =>  A_y = 0
  Phi_y = s_xyy = 0  =>  B_x = 0
  => K = -gamma * s_xxyy / (A * B)

But s_xxyy has indefinite sign at this point.

Alternative: at the minimum of B over x (fixed y):
  B_x = 0, B_xx >= 0  =>  s_xyy = 0, s_xxyy >= 0
  => leading term is <= 0, but A_y != 0 gives correction terms.

We test whether K <= 0 at these auxiliary points.
"""

import numpy as np
from itertools import product as iprod

def make_random_fourier(N_modes, rng):
    """Generate random Fourier coefficients for a function on T^2."""
    coeffs = {}
    for k1 in range(-N_modes, N_modes + 1):
        for k2 in range(-N_modes, N_modes + 1):
            if k1 == 0 and k2 == 0:
                coeffs[(k1, k2)] = 0.0
            else:
                coeffs[(k1, k2)] = rng.normal() / (1 + k1**2 + k2**2)
    return coeffs

def eval_fourier(coeffs, x, y):
    """Evaluate Fourier series and its derivatives at (x, y)."""
    s = 0; sx = 0; sy = 0
    sxx = 0; sxy = 0; syy = 0
    sxxx = 0; sxxy = 0; sxyy = 0; syyy = 0
    sxxyy = 0
    for (k1, k2), c in coeffs.items():
        phase = k1 * x + k2 * y
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)
        s += c * cos_p
        sx += -c * k1 * sin_p
        sy += -c * k2 * sin_p
        sxx += -c * k1**2 * cos_p
        sxy += -c * k1 * k2 * cos_p
        syy += -c * k2**2 * cos_p
        sxxx += c * k1**3 * sin_p
        sxxy += c * k1**2 * k2 * sin_p
        sxyy += c * k1 * k2**2 * sin_p
        syyy += c * k2**3 * sin_p
        sxxyy += c * k1**2 * k2**2 * cos_p
    return {
        's': s, 'sx': sx, 'sy': sy,
        'sxx': sxx, 'sxy': sxy, 'syy': syy,
        'sxxx': sxxx, 'sxxy': sxxy, 'sxyy': sxyy, 'syyy': syyy,
        'sxxyy': sxxyy
    }

def gaussian_curvature_diagonal(A, B, Ax, Ay, Bx, By, Axx, Ayy, Bxx, Byy):
    """
    Gaussian curvature of g = A dx^2 + B dy^2.

    Standard formula:
    K = -1/(2*sqrt(AB)) * [d/dx(Gx/sqrt(EG)) + d/dy(Ey/sqrt(EG))]

    Expanded:
    K = -(Bxx + Ayy)/(2AB)
        + Ax*Bx/(4A^2*B) + Bx^2/(4A*B^2)
        + Ay^2/(4A^2*B) + Ay*By/(4A*B^2)
    """
    AB = A * B
    K = -(Bxx + Ayy) / (2 * AB) \
        + Ax * Bx / (4 * A**2 * B) + Bx**2 / (4 * A * B**2) \
        + Ay**2 / (4 * A**2 * B) + Ay * By / (4 * A * B**2)
    return K

def run_test(gamma, N_modes, N_grid, n_trials, rng):
    """
    For random Fourier seams, compute Gaussian curvature of the
    Hessian seam metric and test various auxiliary function strategies.
    """
    xx = np.linspace(0, 2*np.pi, N_grid, endpoint=False)
    yy = np.linspace(0, 2*np.pi, N_grid, endpoint=False)
    X, Y = np.meshgrid(xx, yy, indexing='ij')

    results = {
        'max_sxy': {'success': 0, 'fail': 0, 'skip': 0},
        'min_sxy': {'success': 0, 'fail': 0, 'skip': 0},
        'sequential_minB_minA': {'success': 0, 'fail': 0, 'skip': 0},
        'min_sxx_plus_syy': {'success': 0, 'fail': 0, 'skip': 0},
        'always_sign_changes': {'yes': 0, 'no': 0},
    }

    for trial in range(n_trials):
        coeffs = make_random_fourier(N_modes, rng)

        # Evaluate on grid
        S = np.zeros_like(X)
        SXX = np.zeros_like(X)
        SXY = np.zeros_like(X)
        SYY = np.zeros_like(X)
        SXXY = np.zeros_like(X)
        SXYY = np.zeros_like(X)
        SXXYY = np.zeros_like(X)
        SXXX = np.zeros_like(X)
        SYYY = np.zeros_like(X)

        for (k1, k2), c in coeffs.items():
            phase = k1 * X + k2 * Y
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)
            SXX += -c * k1**2 * cos_p
            SXY += -c * k1 * k2 * cos_p
            SYY += -c * k2**2 * cos_p
            SXXY += c * k1**2 * k2 * sin_p
            SXYY += c * k1 * k2**2 * sin_p
            SXXYY += c * k1**2 * k2**2 * cos_p
            SXXX += c * k1**3 * sin_p
            SYYY += c * k2**3 * sin_p

        # Metric coefficients
        A = 1 + gamma * SXX
        B = 1 + gamma * SYY
        Ax = gamma * SXXX
        Ay = gamma * SXXY
        Bx = gamma * SXYY
        By = gamma * SYYY

        # Check Riemannian condition
        if np.min(A) <= 0 or np.min(B) <= 0:
            for key in results:
                if key != 'always_sign_changes':
                    results[key]['skip'] += 1
            continue

        # Ayy = Bxx = gamma * sxxyy (the KEY structural identity!)
        Ayy = gamma * SXXYY
        Bxx = gamma * SXXYY

        # Compute Gaussian curvature on the full grid
        K = gaussian_curvature_diagonal(A, B, Ax, Ay, Bx, By, Ax*0, Ayy, Bxx, By*0)
        # Wait, I need Axx and Byy too for the full formula...
        # Axx = gamma * s_xxxx, Byy = gamma * s_yyyy
        # But these don't appear in the Brioschi formula for K!
        # Let me recompute properly.

        # Actually, in the Brioschi formula:
        # K = -(Bxx + Ayy)/(2AB) + Ax*Bx/(4A^2*B) + Bx^2/(4A*B^2)
        #     + Ay^2/(4A^2*B) + Ay*By/(4A*B^2)
        # where Bxx = d^2B/dx^2 and Ayy = d^2A/dy^2
        # These are gamma*s_xxyy for both! (the structural identity)
        # Ax = dA/dx = gamma*s_xxx, Bx = dB/dx = gamma*s_xyy
        # Ay = dA/dy = gamma*s_xxy, By = dB/dy = gamma*s_yyy

        K = -(Bxx + Ayy) / (2 * A * B) \
            + Ax * Bx / (4 * A**2 * B) + Bx**2 / (4 * A * B**2) \
            + Ay**2 / (4 * A**2 * B) + Ay * By / (4 * A * B**2)

        # Check if K always changes sign
        if np.min(K) < -1e-12 and np.max(K) > 1e-12:
            results['always_sign_changes']['yes'] += 1
        else:
            results['always_sign_changes']['no'] += 1

        # ========================================
        # Strategy 1: max of s_xy
        # At max: s_xxy = 0 (Ay=0), s_xyy = 0 (Bx=0)
        # => K = -gamma*s_xxyy / (A*B)
        # ========================================
        idx_max_sxy = np.unravel_index(np.argmax(SXY), SXY.shape)
        K_at_max_sxy = K[idx_max_sxy]
        if K_at_max_sxy <= 1e-10:
            results['max_sxy']['success'] += 1
        else:
            results['max_sxy']['fail'] += 1

        # Strategy 1b: min of s_xy
        idx_min_sxy = np.unravel_index(np.argmin(SXY), SXY.shape)
        K_at_min_sxy = K[idx_min_sxy]
        if K_at_min_sxy <= 1e-10:
            results['min_sxy']['success'] += 1
        else:
            results['min_sxy']['fail'] += 1

        # ========================================
        # Strategy 2: Sequential min of B over x, then min of A over y
        # Step 1: For each y, find x that minimizes B(x,y)
        # Step 2: Among those (x*,y), find y that minimizes A(x*,y)
        # ========================================
        # Step 1: for each y (column j), find x index minimizing B
        min_B_x_idx = np.argmin(B, axis=0)  # shape (N_grid,)
        # Step 2: evaluate A at (x*(y), y) for each y, then find y minimizing it
        A_at_minB = np.array([A[min_B_x_idx[j], j] for j in range(N_grid)])
        j_star = np.argmin(A_at_minB)
        i_star = min_B_x_idx[j_star]
        K_sequential = K[i_star, j_star]
        if K_sequential <= 1e-10:
            results['sequential_minB_minA']['success'] += 1
        else:
            results['sequential_minB_minA']['fail'] += 1

        # ========================================
        # Strategy 3: min of (s_xx + s_yy) = min of Laplacian
        # At min of Delta_s: s_xxx + s_xyy = 0 and s_xxy + s_yyy = 0
        # => Ax + Bx = 0 and Ay + By = 0
        # ========================================
        LAPL = SXX + SYY
        idx_min_lapl = np.unravel_index(np.argmin(LAPL), LAPL.shape)
        K_at_min_lapl = K[idx_min_lapl]
        if K_at_min_lapl <= 1e-10:
            results['min_sxx_plus_syy']['success'] += 1
        else:
            results['min_sxx_plus_syy']['fail'] += 1

    return results

def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("Pathway 2: Hessian seam metric on T^2")
    print("g = (1 + gamma*s_xx) dx^2 + (1 + gamma*s_yy) dy^2")
    print("Key identity: A_yy = B_xx = gamma * s_xxyy")
    print("=" * 70)

    for gamma in [0.01, 0.05, 0.1]:
        print(f"\n--- gamma = {gamma} ---")
        res = run_test(gamma=gamma, N_modes=3, N_grid=200,
                      n_trials=500, rng=rng)

        total = 500
        for strategy, counts in res.items():
            if strategy == 'always_sign_changes':
                print(f"  K always changes sign: {counts['yes']}/{counts['yes']+counts['no']}")
            else:
                valid = counts['success'] + counts['fail']
                if valid > 0:
                    print(f"  {strategy}: K<=0 in {counts['success']}/{valid}"
                          f" ({100*counts['success']/valid:.1f}%)"
                          f" [skipped {counts['skip']}]")

    # Detailed analysis of failures for the best strategy
    print("\n" + "=" * 70)
    print("Detailed failure analysis (gamma=0.05, max_sxy strategy)")
    print("=" * 70)
    rng2 = np.random.default_rng(123)
    xx = np.linspace(0, 2*np.pi, 200, endpoint=False)
    yy = np.linspace(0, 2*np.pi, 200, endpoint=False)
    X, Y = np.meshgrid(xx, yy, indexing='ij')
    gamma = 0.05
    n_fail = 0

    for trial in range(200):
        coeffs = make_random_fourier(3, rng2)
        SXX = np.zeros_like(X); SXY = np.zeros_like(X)
        SYY = np.zeros_like(X); SXXY = np.zeros_like(X)
        SXYY = np.zeros_like(X); SXXYY = np.zeros_like(X)
        SXXX = np.zeros_like(X); SYYY = np.zeros_like(X)
        for (k1, k2), c in coeffs.items():
            phase = k1 * X + k2 * Y
            cos_p = np.cos(phase); sin_p = np.sin(phase)
            SXX += -c * k1**2 * cos_p
            SXY += -c * k1 * k2 * cos_p
            SYY += -c * k2**2 * cos_p
            SXXY += c * k1**2 * k2 * sin_p
            SXYY += c * k1 * k2**2 * sin_p
            SXXYY += c * k1**2 * k2**2 * cos_p
            SXXX += c * k1**3 * sin_p
            SYYY += c * k2**3 * sin_p
        A = 1 + gamma * SXX; B = 1 + gamma * SYY
        if np.min(A) <= 0 or np.min(B) <= 0:
            continue
        Ax = gamma*SXXX; Ay = gamma*SXXY
        Bx = gamma*SXYY; By = gamma*SYYY
        Ayy = gamma*SXXYY; Bxx = gamma*SXXYY
        K = -(Bxx + Ayy)/(2*A*B) + Ax*Bx/(4*A**2*B) + Bx**2/(4*A*B**2) \
            + Ay**2/(4*A**2*B) + Ay*By/(4*A*B**2)

        idx = np.unravel_index(np.argmax(SXY), SXY.shape)
        K_val = K[idx]
        if K_val > 1e-10:
            n_fail += 1
            if n_fail <= 5:
                print(f"  Trial {trial}: K = {K_val:.6f} at max(s_xy)")
                print(f"    s_xxy = {SXXY[idx]:.6f}, s_xyy = {SXYY[idx]:.6f}")
                print(f"    s_xxyy = {SXXYY[idx]:.6f}")
                print(f"    A = {A[idx]:.6f}, B = {B[idx]:.6f}")
                # Decompose K into leading vs correction
                leading = -gamma * SXXYY[idx] / (A[idx] * B[idx])
                correction = K_val - leading
                print(f"    Leading term: {leading:.6f}")
                print(f"    Correction:   {correction:.6f}")

    print(f"\n  Total failures: {n_fail}/200")

    # ========================================
    # NEW STRATEGY: Simultaneously zero out Ay and Bx
    # by finding saddle/extremal of s_xy AND checking sign of s_xxyy
    # ========================================
    print("\n" + "=" * 70)
    print("Combined strategy: max(s_xy) OR min(s_xy), take whichever gives K<=0")
    print("=" * 70)
    rng3 = np.random.default_rng(456)
    gamma = 0.05
    n_success = 0; n_fail = 0; n_skip = 0
    for trial in range(500):
        coeffs = make_random_fourier(3, rng3)
        SXX = np.zeros_like(X); SXY = np.zeros_like(X)
        SYY = np.zeros_like(X); SXXY = np.zeros_like(X)
        SXYY = np.zeros_like(X); SXXYY = np.zeros_like(X)
        SXXX = np.zeros_like(X); SYYY = np.zeros_like(X)
        for (k1, k2), c in coeffs.items():
            phase = k1 * X + k2 * Y
            cos_p = np.cos(phase); sin_p = np.sin(phase)
            SXX += -c * k1**2 * cos_p
            SXY += -c * k1 * k2 * cos_p
            SYY += -c * k2**2 * cos_p
            SXXY += c * k1**2 * k2 * sin_p
            SXYY += c * k1 * k2**2 * sin_p
            SXXYY += c * k1**2 * k2**2 * cos_p
            SXXX += c * k1**3 * sin_p
            SYYY += c * k2**3 * sin_p
        A = 1 + gamma * SXX; B = 1 + gamma * SYY
        if np.min(A) <= 0 or np.min(B) <= 0:
            n_skip += 1; continue
        Ax = gamma*SXXX; Ay = gamma*SXXY
        Bx = gamma*SXYY; By = gamma*SYYY
        Ayy = gamma*SXXYY; Bxx = gamma*SXXYY
        K = -(Bxx + Ayy)/(2*A*B) + Ax*Bx/(4*A**2*B) + Bx**2/(4*A*B**2) \
            + Ay**2/(4*A**2*B) + Ay*By/(4*A*B**2)

        # At max(s_xy): s_xxyy can have either sign
        # At min(s_xy): s_xxyy can have either sign, but OPPOSITE
        # If s_xxyy > 0 at max => K = -gamma*s_xxyy/AB < 0 at max (GOOD)
        # If s_xxyy < 0 at max => try min, where s_xxyy might be > 0

        idx_max = np.unravel_index(np.argmax(SXY), SXY.shape)
        idx_min = np.unravel_index(np.argmin(SXY), SXY.shape)
        K_max = K[idx_max]
        K_min = K[idx_min]

        if K_max <= 1e-10 or K_min <= 1e-10:
            n_success += 1
        else:
            n_fail += 1

    print(f"  gamma = {gamma}: {n_success}/{n_success+n_fail} "
          f"({100*n_success/(n_success+n_fail):.1f}%) [skip {n_skip}]")

    # Also test: at ANY critical point of s_xy, is K <= 0?
    # Critical points include maxima, minima, and saddles.
    # At ALL critical points of s_xy: Ay = Bx = 0, K = -gamma*s_xxyy/(AB)
    # We just need s_xxyy > 0 at SOME critical point.
    # But wait — s_xxyy is part of the Hessian of s_xy.
    # At a max: Hessian <= 0, so trace s_xxxy + s_xyyy <= 0, but s_xxyy is off-diagonal.
    # At a saddle: mixed entry s_xxyy can have either sign.
    # Since s_xxyy is continuous and integral of s_xxyy over T^2 = sum c*k1^2*k2^2 (often > 0),
    # there must be a critical point of s_xy where s_xxyy > 0.

    print("\n" + "=" * 70)
    print("BEST STRATEGY: find ANY critical point of s_xy where s_xxyy > 0")
    print("(Then K = -gamma*s_xxyy/AB < 0 exactly)")
    print("=" * 70)
    rng4 = np.random.default_rng(789)
    gamma = 0.05
    n_success = 0; n_fail = 0; n_skip = 0
    for trial in range(500):
        coeffs = make_random_fourier(3, rng4)
        SXY = np.zeros_like(X)
        SXXY = np.zeros_like(X)
        SXYY = np.zeros_like(X)
        SXXYY = np.zeros_like(X)
        for (k1, k2), c in coeffs.items():
            phase = k1 * X + k2 * Y
            cos_p = np.cos(phase); sin_p = np.sin(phase)
            SXY += -c * k1 * k2 * cos_p
            SXXY += c * k1**2 * k2 * sin_p
            SXYY += c * k1 * k2**2 * sin_p
            SXXYY += c * k1**2 * k2**2 * cos_p

        # Find approximate critical points of s_xy
        # (where |grad(s_xy)| is small)
        grad_mag = np.sqrt(SXXY**2 + SXYY**2)
        # Find local minima of grad_mag (i.e., near-critical points)
        threshold = np.percentile(grad_mag, 1)  # bottom 1%
        crit_mask = grad_mag < max(threshold, 1e-8)

        if not np.any(crit_mask):
            n_skip += 1
            continue

        # At critical points, check if s_xxyy > 0
        sxxyy_at_crit = SXXYY[crit_mask]
        if np.any(sxxyy_at_crit > 1e-10):
            n_success += 1
        else:
            n_fail += 1
            if n_fail <= 3:
                print(f"  Trial {trial}: all s_xxyy at crit pts <= 0")
                print(f"    s_xxyy range at crit: [{np.min(sxxyy_at_crit):.6f}, {np.max(sxxyy_at_crit):.6f}]")
                print(f"    Number of crit points: {np.sum(crit_mask)}")

    print(f"\n  Result: {n_success}/{n_success+n_fail} "
          f"({100*n_success/(n_success+n_fail):.1f}%) [skip {n_skip}]")

if __name__ == '__main__':
    main()
