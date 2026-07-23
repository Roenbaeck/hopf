"""
Investigate the boundary behavior of the orbit metric near theta_k = 0 or pi.

For the orbit space [0,pi]^2, the boundary corresponds to degenerate orbits
on S^2 x S^2. The metric coefficients must satisfy regularity conditions
for the metric to extend smoothly to S^2 x S^2.

Here we analyze what happens to the orbit metric g_2D and its curvature
near the boundary.
"""
import numpy as np

N = 500
th1_1d = np.linspace(0.001, np.pi - 0.001, N)
th2_1d = np.linspace(0.001, np.pi - 0.001, N)
dt = th1_1d[1] - th1_1d[0]
TH1, TH2 = np.meshgrid(th1_1d, th2_1d, indexing='ij')


def seam_derivs_grid(s_func, h=1e-5):
    s = s_func(TH1, TH2)
    s1 = (s_func(TH1+h, TH2) - s_func(TH1-h, TH2)) / (2*h)
    s2 = (s_func(TH1, TH2+h) - s_func(TH1, TH2-h)) / (2*h)
    s11 = (s_func(TH1+h, TH2) - 2*s + s_func(TH1-h, TH2)) / h**2
    s22 = (s_func(TH1, TH2+h) - 2*s + s_func(TH1, TH2-h)) / h**2
    s12 = (s_func(TH1+h, TH2+h) - s_func(TH1+h, TH2-h)
           - s_func(TH1-h, TH2+h) + s_func(TH1-h, TH2-h)) / (4*h**2)
    return s, s1, s2, s11, s22, s12


def gauss_K_brioschi(L1, L2, E_coeff):
    h = dt
    pL1 = np.pad(L1, 1, mode='edge')
    pL2 = np.pad(L2, 1, mode='edge')
    pE  = np.pad(E_coeff, 1, mode='edge')

    L1_1 = (pL1[2:,1:-1] - pL1[:-2,1:-1]) / (2*h)
    L1_2 = (pL1[1:-1,2:] - pL1[1:-1,:-2]) / (2*h)
    L2_1 = (pL2[2:,1:-1] - pL2[:-2,1:-1]) / (2*h)
    L2_2 = (pL2[1:-1,2:] - pL2[1:-1,:-2]) / (2*h)
    E_1  = (pE[2:,1:-1]  - pE[:-2,1:-1])  / (2*h)
    E_2  = (pE[1:-1,2:]  - pE[1:-1,:-2])  / (2*h)

    L1_22 = (pL1[1:-1,2:] - 2*L1 + pL1[1:-1,:-2]) / h**2
    L2_11 = (pL2[2:,1:-1] - 2*L2 + pL2[:-2,1:-1]) / h**2
    E_12  = (pE[2:,2:] - pE[2:,:-2] - pE[:-2,2:] + pE[:-2,:-2]) / (4*h**2)

    D = L1*L2 - E_coeff**2
    a11 = -L1_22/2 + E_12 - L2_11/2
    a12 = L1_1/2; a13 = E_1 - L1_2/2
    a21 = E_2 - L2_1/2; a22 = L1; a23 = E_coeff
    a31 = L2_2/2; a32 = E_coeff; a33 = L2
    detA = a11*(a22*a33-a23*a32) - a12*(a21*a33-a23*a31) + a13*(a21*a32-a22*a31)
    detB = -L1_2/2*(L1_2/2*L2 - E_coeff*L2_1/2) + L2_1/2*(L1_2/2*E_coeff - L1*L2_1/2)
    K = (detA - detB) / D**2
    return K


# For the seam s = cos(theta_1)*cos(theta_2):
# s_1 = -sin(t1)*cos(t2), s_2 = cos(t1)*(-sin(t2)),
# s_11 = -cos(t1)*cos(t2), s_22 = -cos(t1)*cos(t2) [sic, actually cos(t1)*(-cos(t2))...wait]
# Actually: s_1 = -sin(t1)*cos(t2), s_11 = -cos(t1)*cos(t2)
#           s_2 = -cos(t1)*sin(t2), s_22 = -cos(t1)*cos(t2)
# Near t1=0: s_1 ~ 0, s_11 ~ -cos(t2), s(0,t2) = cos(t2)

# So lambda_1(0, theta_2) = a1 + b1*0 + g1*(-cos(t2)) = a1 - g1*cos(t2)
# These are smooth and bounded.

# What matters is: do lambda_1 and lambda_2 have specific boundary values
# that constrain the Gaussian curvature?

# Let's just visualize K near the boundary
s_func = lambda t1, t2: np.cos(t1) * np.cos(t2)
params = (1.0, 1.0, 0.1, 0.1, 0, 0.05, 0.05, 0)
a1, a2, b1, b2, b12, g1, g2, g12 = params
s, s1, s2, s11, s22, s12 = seam_derivs_grid(s_func)
L1 = a1 + b1*s1**2 + g1*s11
L2 = a2 + b2*s2**2 + g2*s22
E  = b12*s1*s2 + g12*s12

K = gauss_K_brioschi(L1, L2, E)

print("K profile along theta_1 for fixed theta_2 = pi/2:")
mid = N // 2
for i in range(0, N, N//20):
    print(f"  theta_1={th1_1d[i]:.4f}: K={K[i,mid]:.6f}")

print(f"\nK at corners:")
print(f"  (0,0):  K={K[0,0]:.6f}")
print(f"  (0,pi): K={K[0,-1]:.6f}")
print(f"  (pi,0): K={K[-1,0]:.6f}")
print(f"  (pi,pi): K={K[-1,-1]:.6f}")

print(f"\nK range: [{np.min(K):.6f}, {np.max(K):.6f}]")
print(f"K at center (pi/2, pi/2): {K[mid, mid]:.6f}")

# Check the boundary values of lambda_1, lambda_2
print(f"\nBoundary values:")
print(f"  L1 at theta_1=0: {L1[0, mid]:.6f} at theta_1=pi: {L1[-1, mid]:.6f}")
print(f"  L2 at theta_2=0: {L2[mid, 0]:.6f} at theta_2=pi: {L2[mid, -1]:.6f}")
print(f"  L1 range: [{np.min(L1):.6f}, {np.max(L1):.6f}]")
print(f"  L2 range: [{np.min(L2):.6f}, {np.max(L2):.6f}]")
