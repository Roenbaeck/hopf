"""
Test whether the boundary integral from Gauss-Bonnet on the 2D orbit space
has a definite sign.

K*sqrt(L1*L2) = -1/2 * [d/dt1(L2_1/sqrt(L1*L2)) + d/dt2(L1_2/sqrt(L1*L2))]

Integrating over [0,pi]^2:
∫∫ K*sqrt(L1*L2) dt1 dt2 = -1/2 * [∫(L2_1/sqrt(L1*L2)|_{t1=pi} - L2_1/sqrt(L1*L2)|_{t1=0}) dt2
                                     + ∫(L1_2/sqrt(L1*L2)|_{t2=pi} - L1_2/sqrt(L1*L2)|_{t2=0}) dt1]

For seam metrics on S^2, smoothness at poles imposes parity conditions.
"""

import numpy as np

np.random.seed(42)

N = 500
theta = np.linspace(0, np.pi, N)
dt = theta[1] - theta[0]
T1, T2 = np.meshgrid(theta, theta, indexing='ij')

def make_smooth_seam(n_modes=4, scale=0.3):
    """Generate a smooth O(2)xO(2)-equivariant seam and metric.
    
    For smoothness on S^2, use cos(k*theta) modes (which are smooth at poles).
    """
    coeffs = np.random.randn(n_modes, n_modes) * scale
    
    alpha1 = 1.0 + 0.3*np.random.rand()
    alpha2 = 1.0 + 0.3*np.random.rand()
    beta1 = 0.3*np.random.randn()
    beta2 = 0.3*np.random.randn()
    gamma1 = 0.2*np.random.randn()
    gamma2 = 0.2*np.random.randn()
    
    # s = sum c_{ij} cos(i*t1)*cos(j*t2) -- smooth at all poles
    s = np.zeros_like(T1)
    s1 = np.zeros_like(T1)  # exact ds/dt1
    s2 = np.zeros_like(T1)  # exact ds/dt2
    s11 = np.zeros_like(T1)
    s22 = np.zeros_like(T1)
    s12 = np.zeros_like(T1)
    
    for i in range(n_modes):
        for j in range(n_modes):
            c = coeffs[i, j]
            ci = np.cos(i*T1)
            si = np.sin(i*T1)
            cj = np.cos(j*T2)
            sj = np.sin(j*T2)
            s += c * ci * cj
            s1 += -c * i * si * cj
            s2 += -c * j * ci * sj
            s11 += -c * i**2 * ci * cj
            s22 += -c * j**2 * ci * cj
            s12 += c * i * j * si * sj
    
    L1 = alpha1 + beta1 * s1**2 + gamma1 * s11
    L2 = alpha2 + beta2 * s2**2 + gamma2 * s22
    
    if np.min(L1) <= 0.05 or np.min(L2) <= 0.05:
        return None
    
    # Also compute derivatives of L1, L2 analytically
    # dL1/dt1 = 2*beta1*s1*s11 + gamma1*s111 (need s111)
    # dL1/dt2 = 2*beta1*s1*s12 + gamma1*s112 (need s112)
    # etc.
    # For simplicity, use numerical derivatives on fine grid
    L1_1 = np.gradient(L1, dt, axis=0)
    L1_2 = np.gradient(L1, dt, axis=1)
    L2_1 = np.gradient(L2, dt, axis=0)
    L2_2 = np.gradient(L2, dt, axis=1)
    
    return L1, L2, L1_1, L1_2, L2_1, L2_2


def compute_boundary_integral(L1, L2, L1_2, L2_1):
    """
    Boundary integral = -1/2 * [∫_0^pi (L2_1/sqrt(L1*L2))|_{t1=pi} dt2
                                - ∫_0^pi (L2_1/sqrt(L1*L2))|_{t1=0} dt2
                                + ∫_0^pi (L1_2/sqrt(L1*L2))|_{t2=pi} dt1
                                - ∫_0^pi (L1_2/sqrt(L1*L2))|_{t2=0} dt1]
    """
    sqrtLprod = np.sqrt(L1 * L2)
    
    # Boundary at t1 = 0 (first row)
    b1_at_0 = L2_1[0, :] / sqrtLprod[0, :]  # function of t2
    # Boundary at t1 = pi (last row)
    b1_at_pi = L2_1[-1, :] / sqrtLprod[-1, :]
    
    # Boundary at t2 = 0 (first column)
    b2_at_0 = L1_2[:, 0] / sqrtLprod[:, 0]
    # Boundary at t2 = pi (last column)
    b2_at_pi = L1_2[:, -1] / sqrtLprod[:, -1]
    
    I_boundary = -0.5 * (np.trapz(b1_at_pi - b1_at_0, dx=dt) 
                         + np.trapz(b2_at_pi - b2_at_0, dx=dt))
    
    return I_boundary


def compute_area_integral(L1, L2):
    """Compute ∫∫ K * sqrt(L1*L2) dt1 dt2 directly via computing K."""
    L1_1 = np.gradient(L1, dt, axis=0)
    L1_2 = np.gradient(L1, dt, axis=1)
    L2_1 = np.gradient(L2, dt, axis=0)
    L2_2 = np.gradient(L2, dt, axis=1)
    L1_22 = np.gradient(L1_2, dt, axis=1)
    L2_11 = np.gradient(L2_1, dt, axis=0)
    
    N = (L1_1*L2_1 + L1_2**2)*L2 + (L1_2*L2_2 + L2_1**2)*L1 - 2*(L1_22 + L2_11)*L1*L2
    K = N / (4*L1**2 * L2**2)
    
    integrand = K * np.sqrt(L1 * L2)
    I = np.trapz(np.trapz(integrand, dx=dt, axis=1), dx=dt)
    return I, K


def test_boundary_integral(n_trials=500):
    n_valid = 0
    n_bdy_nonpos = 0
    n_bdy_nonneg = 0
    bdy_vals = []
    area_vals = []
    
    for trial in range(n_trials):
        data = make_smooth_seam()
        if data is None:
            continue
        L1, L2, L1_1, L1_2, L2_1, L2_2 = data
        n_valid += 1
        
        I_bdy = compute_boundary_integral(L1, L2, L1_2, L2_1)
        I_area, K = compute_area_integral(L1, L2)
        
        bdy_vals.append(I_bdy)
        area_vals.append(I_area)
        
        if I_bdy <= 1e-10:
            n_bdy_nonpos += 1
        if I_bdy >= -1e-10:
            n_bdy_nonneg += 1
    
    bdy_vals = np.array(bdy_vals)
    area_vals = np.array(area_vals)
    
    print(f"Valid trials: {n_valid}")
    print(f"\nBoundary integral statistics:")
    print(f"  min = {bdy_vals.min():.6f}")
    print(f"  max = {bdy_vals.max():.6f}")
    print(f"  mean = {bdy_vals.mean():.6f}")
    print(f"  <= 0: {n_bdy_nonpos}/{n_valid}")
    print(f"  >= 0: {n_bdy_nonneg}/{n_valid}")
    
    print(f"\nArea integral (∫∫ K sqrt(L1L2) dt1dt2) statistics:")
    print(f"  min = {area_vals.min():.6f}")
    print(f"  max = {area_vals.max():.6f}")
    print(f"  mean = {area_vals.mean():.6f}")
    
    # Check if boundary integral = area integral (consistency check)
    diff = np.abs(bdy_vals - area_vals)
    print(f"\n|boundary - area| max = {diff.max():.6f}, mean = {diff.mean():.6f}")
    
    # Now check smoothness at poles for our metrics
    print(f"\nSmoothness check at poles (sample):")
    data = make_smooth_seam()
    if data:
        L1, L2, L1_1, L1_2, L2_1, L2_2 = data
        print(f"  L2_1[0,:5] = {L2_1[0,:5]}")
        print(f"  L2_1[-1,:5] = {L2_1[-1,:5]}")
        print(f"  L1_2[:5,0] = {L1_2[:5,0]}")
        print(f"  L1_2[:5,-1] = {L1_2[:5,-1]}")


if __name__ == "__main__":
    print("=== Boundary integral analysis ===\n")
    test_boundary_integral(n_trials=500)
