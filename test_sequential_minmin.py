"""
Test the sequential min-min proof approach:
1. For each theta_1, find theta_2*(theta_1) = argmin_{theta_2} log(lambda_1(theta_1, theta_2))
2. Then find theta_1* = argmin_{theta_1} log(lambda_2(theta_1, theta_2*(theta_1)))
3. At the point (theta_1*, theta_2*(theta_1*)), check:
   - lambda_1,2 = 0 (from step 1)
   - lambda_1,22 >= 0 (log-convexity at min)
   - What about lambda_2,1 and lambda_2,11?

Also test the "product min" approach:
- Find (theta_1*, theta_2*) = argmin log(lambda_1) over whole domain
  => lambda_1,1 = lambda_1,2 = 0, lambda_1,11 >= 0, lambda_1,22 >= 0
- At this SAME point, what is lambda_2,11?
- K = -(lambda_1,22 + lambda_2,11)/(2*lambda_1*lambda_2)
- We know lambda_1,22 >= 0; is lambda_2,11 also >= 0?

And the "interleaved" approach:
- Pick theta_2* = argmin_{theta_2} lambda_1(theta_1^0, theta_2)
- Pick theta_1* = argmin_{theta_1} lambda_2(theta_1, theta_2*)
- At (theta_1*, theta_2*): lambda_1,2=0 at (theta_1^0, theta_2*) but NOT at (theta_1*, theta_2*)
- So this doesn't directly work.

But what about: pick the point where lambda_1*lambda_2 is minimized,
using the correct K formula?
"""

import numpy as np

np.random.seed(42)

N_theta = 200
theta = np.linspace(0.01, np.pi - 0.01, N_theta)  # avoid boundary

def make_seam_metric(n_modes=3):
    """Generate random seam metric functions lambda_1, lambda_2 (diagonal)."""
    # lambda_k = alpha_k + beta_k * s_k^2 + gamma_k * s_kk
    # For O(2)xO(2) equivariance, s = s(theta_1, theta_2)
    # Use random Fourier series for s
    
    coeffs = np.random.randn(n_modes, n_modes) * 0.3
    
    # Base alpha values (must keep lambda positive)
    alpha1 = 1.0 + 0.3 * np.random.rand()
    alpha2 = 1.0 + 0.3 * np.random.rand()
    beta1 = 0.5 * np.random.randn()
    beta2 = 0.5 * np.random.randn()
    gamma1 = 0.3 * np.random.randn()
    gamma2 = 0.3 * np.random.randn()
    
    T1, T2 = np.meshgrid(theta, theta, indexing='ij')
    
    # Build s from Fourier modes
    s = np.zeros_like(T1)
    for i in range(n_modes):
        for j in range(n_modes):
            s += coeffs[i, j] * np.cos(i * T1) * np.cos(j * T2)
    
    # Compute derivatives numerically
    dt = theta[1] - theta[0]
    s1 = np.gradient(s, dt, axis=0)
    s2 = np.gradient(s, dt, axis=1)
    s11 = np.gradient(s1, dt, axis=0)
    s22 = np.gradient(s2, dt, axis=1)
    
    L1 = alpha1 + beta1 * s1**2 + gamma1 * s11
    L2 = alpha2 + beta2 * s2**2 + gamma2 * s22
    
    # Ensure positivity
    if np.min(L1) <= 0 or np.min(L2) <= 0:
        return None, None, None, None
    
    return L1, L2, T1, T2


def compute_K(L1, L2, dt):
    """Compute Gaussian curvature of diagonal 2D metric."""
    L1_1 = np.gradient(L1, dt, axis=0)
    L1_2 = np.gradient(L1, dt, axis=1)
    L2_1 = np.gradient(L2, dt, axis=0)
    L2_2 = np.gradient(L2, dt, axis=1)
    L1_22 = np.gradient(L1_2, dt, axis=1)
    L2_11 = np.gradient(L2_1, dt, axis=0)
    
    N = (L1_1*L2_1 + L1_2**2)*L2 + (L1_2*L2_2 + L2_1**2)*L1 - 2*(L1_22 + L2_11)*L1*L2
    K = N / (4*L1**2 * L2**2)
    
    return K, L1_1, L1_2, L2_1, L2_2, L1_22, L2_11


def test_approach(n_trials=500):
    dt = theta[1] - theta[0]
    
    n_valid = 0
    n_seqmin_works = 0
    n_globalmin_works = 0
    n_seqmin_K_nonpos = 0
    
    for trial in range(n_trials):
        L1, L2, T1, T2 = make_seam_metric()
        if L1 is None:
            continue
        n_valid += 1
        
        K, L1_1, L1_2, L2_1, L2_2, L1_22, L2_11 = compute_K(L1, L2, dt)
        
        # Interior region (avoid boundary effects from numerical derivatives)
        margin = 10
        K_int = K[margin:-margin, margin:-margin]
        L1_int = L1[margin:-margin, margin:-margin]
        L2_int = L2[margin:-margin, margin:-margin]
        L1_2_int = L1_2[margin:-margin, margin:-margin]
        L2_1_int = L2_1[margin:-margin, margin:-margin]
        L1_22_int = L1_22[margin:-margin, margin:-margin]
        L2_11_int = L2_11[margin:-margin, margin:-margin]
        
        # Approach 1: Sequential min-min
        # For each theta_1 (row), find argmin_theta_2 of log(L1)
        logL1 = np.log(L1[margin:-margin, margin:-margin])
        logL2 = np.log(L2[margin:-margin, margin:-margin])
        
        # Step 1: for each row i, find j*(i) = argmin_j logL1[i,j]
        j_star = np.argmin(logL1, axis=1)
        
        # Step 2: evaluate logL2 along the curve (i, j*(i))
        n_int = logL1.shape[0]
        logL2_curve = np.array([logL2[i, j_star[i]] for i in range(n_int)])
        i_star = np.argmin(logL2_curve)
        
        # Point: (i_star, j_star[i_star])
        ii, jj = i_star, j_star[i_star]
        
        # At this point, check:
        # lambda_1,2 should be ~ 0 (from step 1, this is where logL1 is min in theta_2)
        # lambda_1,22 should be >= 0
        check_L1_2 = abs(L1_2_int[ii, jj])
        check_L1_22 = L1_22_int[ii, jj]
        check_L2_11 = L2_11_int[ii, jj]
        
        # K at this point
        K_at_point = K_int[ii, jj]
        
        if K_at_point <= 1e-10:  # K <= 0 (with numerical tolerance)
            n_seqmin_K_nonpos += 1
        
        # Approach 2: Global minimum of L1*L2
        prod = L1_int * L2_int
        idx_min = np.unravel_index(np.argmin(prod), prod.shape)
        K_at_prodmin = K_int[idx_min]
        
        if K_at_prodmin <= 1e-10:
            n_globalmin_works += 1
    
    print(f"Valid trials: {n_valid}")
    print(f"Sequential min-min: K<=0 at found point: {n_seqmin_K_nonpos}/{n_valid}")
    print(f"Global min of L1*L2: K<=0 there: {n_globalmin_works}/{n_valid}")


# Also test the specific formula at the sequential min point more carefully
def test_formula_at_seqmin(n_trials=200):
    """
    At the sequential min point:
    - lambda_1 is minimized over theta_2 (for fixed theta_1)
      => lambda_1,2 = 0, lambda_1,22 >= 0
    - logL2 is minimized over theta_1 along the curve theta_2 = theta_2*(theta_1)
    
    The KEY question: does lambda_2,1 = 0 at this point?
    If not, does the full K formula still give K <= 0?
    """
    dt = theta[1] - theta[0]
    n_valid = 0
    n_L2_1_zero = 0
    n_K_nonpos = 0
    
    for trial in range(n_trials):
        L1, L2, T1, T2 = make_seam_metric()
        if L1 is None:
            continue
        n_valid += 1
        
        K, L1_1, L1_2, L2_1, L2_2, L1_22, L2_11 = compute_K(L1, L2, dt)
        
        margin = 10
        logL1 = np.log(L1[margin:-margin, margin:-margin])
        logL2 = np.log(L2[margin:-margin, margin:-margin])
        L1_2_int = L1_2[margin:-margin, margin:-margin]
        L2_1_int = L2_1[margin:-margin, margin:-margin]
        L1_22_int = L1_22[margin:-margin, margin:-margin]
        L2_11_int = L2_11[margin:-margin, margin:-margin]
        K_int = K[margin:-margin, margin:-margin]
        
        # Sequential min: min_theta2 logL1 then min_theta1 logL2
        j_star = np.argmin(logL1, axis=1)
        n_int = logL1.shape[0]
        logL2_curve = np.array([logL2[i, j_star[i]] for i in range(n_int)])
        i_star = np.argmin(logL2_curve)
        ii, jj = i_star, j_star[i_star]
        
        K_here = K_int[ii, jj]
        L2_1_here = L2_1_int[ii, jj]
        L1_2_here = L1_2_int[ii, jj]
        
        if abs(L2_1_here) < 0.05:
            n_L2_1_zero += 1
        if K_here <= 1e-10:
            n_K_nonpos += 1
    
    print(f"\n--- Detailed sequential min analysis ---")
    print(f"Valid: {n_valid}")
    print(f"|lambda_2,1| < 0.05 at seq-min point: {n_L2_1_zero}/{n_valid}")
    print(f"K <= 0 at seq-min point: {n_K_nonpos}/{n_valid}")


# Test approach where we INDEPENDENTLY minimize
def test_independent_min(n_trials=200):
    """
    Independent minimization:
    - Find (a, b) = argmin_{theta_1,theta_2} logL1  (global min of logL1)
      => L1,1=0, L1,2=0, L1,11>=0, L1,22>=0
    - At this SAME point (a,b), K = -(L1,22 + L2,11)/(2*L1*L2)
      since L1,2=0 and L2,1 may not be 0.
    Wait, L2,1 != 0 in general. So K != -(L1,22+L2,11)/(2L1L2).
    
    With L1,1=0, L1,2=0, but L2,1,L2,2 arbitrary:
    K = [0*L2 + (0 + L2,1^2)*L1 - 2*(L1,22 + L2,11)*L1*L2] / (4*L1^2*L2^2)
      = [L2,1^2*L1 - 2*(L1,22+L2,11)*L1*L2] / (4*L1^2*L2^2)
      = [L2,1^2 - 2*(L1,22+L2,11)*L2] / (4*L1*L2^2)
    
    We know L1,22 >= 0, but L2,11 has no constraint and L2,1^2 >= 0 adds a positive term.
    Hmm, this doesn't directly work.
    """
    dt = theta[1] - theta[0]
    n_valid = 0
    n_K_nonpos = 0
    
    for trial in range(n_trials):
        L1, L2, T1, T2 = make_seam_metric()
        if L1 is None:
            continue
        n_valid += 1
        
        K, L1_1, L1_2, L2_1, L2_2, L1_22, L2_11 = compute_K(L1, L2, dt)
        
        margin = 10
        K_int = K[margin:-margin, margin:-margin]
        logL1 = np.log(L1[margin:-margin, margin:-margin])
        
        # Global min of logL1
        idx = np.unravel_index(np.argmin(logL1), logL1.shape)
        K_here = K_int[idx]
        
        if K_here <= 1e-10:
            n_K_nonpos += 1
    
    print(f"\n--- Global min of logL1 ---")
    print(f"Valid: {n_valid}")
    print(f"K <= 0 at global min of logL1: {n_K_nonpos}/{n_valid}")


# The CORRECT sequential approach:
def test_correct_sequential(n_trials=500):
    """
    1. For each theta_1, find theta_2*(theta_1) = argmin_{theta_2} logL1(theta_1, theta_2)
       At this point: L1,2 = 0, L1,22/L1 >= 0 (i.e. L1,22 >= 0)
    2. On the curve (theta_1, theta_2*(theta_1)), find min of logL2
       At theta_1*: d/dtheta_1[logL2(theta_1, theta_2*(theta_1))] = 0
    
    At the point (theta_1*, theta_2*):
    - L1,2 = 0 at (theta_1*, theta_2*) because theta_2* = argmin_{theta_2} logL1(theta_1*, .)
    - L1,22 >= 0
    - BUT lambda_2,1 is NOT necessarily zero
    
    K with L1,2 = 0:
    K = [L1,1*L2,1*L2 + L2,1^2*L1 - 2*(L1,22+L2,11)*L1*L2] / (4*L1^2*L2^2)
    
    Not clear this is <= 0.
    
    OK what about flipping the order?
    1. For each theta_2, find theta_1*(theta_2) = argmin_{theta_1} logL2(theta_1, theta_2)
       => L2,1 = 0, L2,11 >= 0
    2. On the curve, find min of logL1 => at theta_2*
    
    At (theta_1*, theta_2*):
    - L2,1 = 0, L2,11 >= 0
    - L1,2 may not be 0
    
    K with L2,1 = 0:
    K = [L1,2^2*L2 + L1,2*L2,2*L1 - 2*(L1,22+L2,11)*L1*L2] / (4*L1^2*L2^2)
    
    We know L2,11 >= 0 but L1,22 could be anything. Hmm.
    
    What if we combine BOTH orderings?
    """
    dt = theta[1] - theta[0]
    n_valid = 0
    results = {"seqmin12": 0, "seqmin21": 0, "either": 0}
    
    for trial in range(n_trials):
        L1, L2, T1, T2 = make_seam_metric()
        if L1 is None:
            continue
        n_valid += 1
        
        K, _, _, _, _, _, _ = compute_K(L1, L2, dt)
        
        margin = 10
        K_int = K[margin:-margin, margin:-margin]
        logL1 = np.log(L1[margin:-margin, margin:-margin])
        logL2 = np.log(L2[margin:-margin, margin:-margin])
        n_int = logL1.shape[0]
        
        # Order 1: min_theta2(logL1) then min_theta1(logL2)
        j_star = np.argmin(logL1, axis=1)
        logL2_curve1 = np.array([logL2[i, j_star[i]] for i in range(n_int)])
        i_star1 = np.argmin(logL2_curve1)
        K1 = K_int[i_star1, j_star[i_star1]]
        
        # Order 2: min_theta1(logL2) then min_theta2(logL1) 
        i_star = np.argmin(logL2, axis=0)
        logL1_curve2 = np.array([logL1[i_star[j], j] for j in range(n_int)])
        j_star2 = np.argmin(logL1_curve2)
        K2 = K_int[i_star[j_star2], j_star2]
        
        if K1 <= 1e-10:
            results["seqmin12"] += 1
        if K2 <= 1e-10:
            results["seqmin21"] += 1
        if K1 <= 1e-10 or K2 <= 1e-10:
            results["either"] += 1
    
    print(f"\n--- Correct sequential min test ---")
    print(f"Valid: {n_valid}")
    print(f"Order 1 (min_t2 L1, min_t1 L2): K<=0: {results['seqmin12']}/{n_valid}")
    print(f"Order 2 (min_t1 L2, min_t2 L1): K<=0: {results['seqmin21']}/{n_valid}")
    print(f"Either order gives K<=0: {results['either']}/{n_valid}")


if __name__ == "__main__":
    print("=== Testing proof approaches for K <= 0 ===\n")
    test_approach(n_trials=500)
    test_formula_at_seqmin(n_trials=300)
    test_independent_min(n_trials=300)
    test_correct_sequential(n_trials=500)
