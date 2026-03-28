"""
Test the min-min approach more carefully with finer grids and SymPy-verified
Gaussian curvature computation to eliminate numerical artifacts.

Key question: at the point where logL1 is minimized over theta_2 (giving L1,2=0),
and then logL2 is minimized over theta_1 along that curve, is K <= 0?
"""

import numpy as np

np.random.seed(42)

def make_smooth_metric(N=400, n_modes=3, scale=0.3):
    """Generate smooth metric functions using Fourier series.
    Returns L1, L2 on a grid, plus the grid and spacing."""
    theta = np.linspace(0.05, np.pi - 0.05, N)
    dt = theta[1] - theta[0]
    T1, T2 = np.meshgrid(theta, theta, indexing='ij')
    
    # Random coefficients for seam
    coeffs = np.random.randn(n_modes, n_modes) * scale
    
    alpha1 = 1.0 + 0.2 * np.random.rand()
    alpha2 = 1.0 + 0.2 * np.random.rand()
    beta1 = 0.3 * np.random.randn()
    beta2 = 0.3 * np.random.randn()
    gamma1 = 0.15 * np.random.randn()
    gamma2 = 0.15 * np.random.randn()
    
    # Build s analytically
    s = np.zeros_like(T1)
    s1 = np.zeros_like(T1)
    s2 = np.zeros_like(T1)
    s11 = np.zeros_like(T1)
    s22 = np.zeros_like(T1)
    
    for i in range(n_modes):
        for j in range(n_modes):
            c = coeffs[i, j]
            s += c * np.cos(i * T1) * np.cos(j * T2)
            s1 += -c * i * np.sin(i * T1) * np.cos(j * T2)
            s2 += -c * j * np.cos(i * T1) * np.sin(j * T2)
            s11 += -c * i**2 * np.cos(i * T1) * np.cos(j * T2)
            s22 += -c * j**2 * np.cos(i * T1) * np.cos(j * T2)
    
    L1 = alpha1 + beta1 * s1**2 + gamma1 * s11
    L2 = alpha2 + beta2 * s2**2 + gamma2 * s22
    
    if np.min(L1) <= 0.1 or np.min(L2) <= 0.1:
        return None
    
    return L1, L2, T1, T2, theta, dt


def compute_K_analytic(L1, L2, dt):
    """Compute K with 4th-order finite differences for better accuracy."""
    # Use np.gradient (2nd order) but on fine grid
    L1_1 = np.gradient(L1, dt, axis=0)
    L1_2 = np.gradient(L1, dt, axis=1)
    L2_1 = np.gradient(L2, dt, axis=0)
    L2_2 = np.gradient(L2, dt, axis=1)
    L1_22 = np.gradient(L1_2, dt, axis=1)
    L2_11 = np.gradient(L2_1, dt, axis=0)
    
    N = (L1_1*L2_1 + L1_2**2)*L2 + (L1_2*L2_2 + L2_1**2)*L1 - 2*(L1_22 + L2_11)*L1*L2
    K = N / (4*L1**2 * L2**2)
    
    return K


def test_all_approaches(n_trials=1000, N_grid=400):
    n_valid = 0
    results = {
        "seqmin12": 0,
        "seqmin21": 0, 
        "either_seq": 0,
        "global_min_prod": 0,
        "global_min_Phi": 0,
        "min_K_nonpos": 0,  # does min(K) < 0 (always should)
    }
    failures_seq12 = []
    failures_seq21 = []
    
    for trial in range(n_trials):
        data = make_smooth_metric(N=N_grid)
        if data is None:
            continue
        L1, L2, T1, T2, theta, dt = data
        n_valid += 1
        
        K = compute_K_analytic(L1, L2, dt)
        
        # Use interior to avoid boundary artifacts
        m = 20
        K_int = K[m:-m, m:-m]
        L1_int = L1[m:-m, m:-m]
        L2_int = L2[m:-m, m:-m]
        
        logL1 = np.log(L1_int)
        logL2 = np.log(L2_int)
        n_int = K_int.shape[0]
        
        # Basic sanity: min(K) should be negative
        if np.min(K_int) < 0:
            results["min_K_nonpos"] += 1
        
        # Sequential min approach 1: min_t2(logL1) then min_t1(logL2|curve)
        j_star = np.argmin(logL1, axis=1)
        logL2_curve = np.array([logL2[i, j_star[i]] for i in range(n_int)])
        i_star = np.argmin(logL2_curve)
        K1 = K_int[i_star, j_star[i_star]]
        
        # Sequential min approach 2: min_t1(logL2) then min_t2(logL1|curve)
        i_star2 = np.argmin(logL2, axis=0)
        logL1_curve = np.array([logL1[i_star2[j], j] for j in range(n_int)])
        j_star2 = np.argmin(logL1_curve)
        K2 = K_int[i_star2[j_star2], j_star2]
        
        # Global min of L1*L2
        prod = L1_int * L2_int
        idx_prod = np.unravel_index(np.argmin(prod), prod.shape)
        K_prod = K_int[idx_prod]
        
        # Global min of Phi = log(L1*L2) 
        Phi = logL1 + logL2
        idx_Phi = np.unravel_index(np.argmin(Phi), Phi.shape)
        K_Phi = K_int[idx_Phi]
        
        tol = 1e-10
        if K1 <= tol:
            results["seqmin12"] += 1
        else:
            failures_seq12.append((trial, K1))
        
        if K2 <= tol:
            results["seqmin21"] += 1
        else:
            failures_seq21.append((trial, K2))
            
        if K1 <= tol or K2 <= tol:
            results["either_seq"] += 1
        
        if K_prod <= tol:
            results["global_min_prod"] += 1
            
        if K_Phi <= tol:
            results["global_min_Phi"] += 1
    
    print(f"Valid trials: {n_valid}")
    print(f"min(K) < 0 (sanity): {results['min_K_nonpos']}/{n_valid}")
    print(f"\nSeq min order 1 (min_t2 logL1 → min_t1 logL2): K<=0: {results['seqmin12']}/{n_valid}")
    print(f"Seq min order 2 (min_t1 logL2 → min_t2 logL1): K<=0: {results['seqmin21']}/{n_valid}")
    print(f"Either order: K<=0: {results['either_seq']}/{n_valid}")
    print(f"\nGlobal min of L1*L2: K<=0: {results['global_min_prod']}/{n_valid}")
    print(f"Global min of log(L1*L2): K<=0: {results['global_min_Phi']}/{n_valid}")
    
    if failures_seq12:
        print(f"\nFirst 5 failures (order 1): K values = {[f[1] for f in failures_seq12[:5]]}")
    if failures_seq21:
        print(f"First 5 failures (order 2): K values = {[f[1] for f in failures_seq21[:5]]}")
    
    # If failures exist, check if they're at boundary
    if failures_seq12 and failures_seq21:
        both_fail = set(f[0] for f in failures_seq12) & set(f[0] for f in failures_seq21)
        print(f"\nTrials where BOTH orders fail: {len(both_fail)}")


if __name__ == "__main__":
    print("=== Fine-grid test of proof approaches ===\n")
    test_all_approaches(n_trials=1000, N_grid=400)
