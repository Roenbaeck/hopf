"""
Verify K = O(γ³) for ℓ=1 seams vs K = O(γ²) for ℓ=2 seams.
Uses the existing test_Q_S2xS2 infrastructure.
"""
import numpy as np
import sys
sys.path.insert(0, '/Users/lars/Documents/GitHub/hopf')

from test_Q_S2xS2 import make_seam_product_harmonics, build_metric_and_curvature

def Kmax_at_point(seam_func, gamma, t1, t2, n_angles=60):
    """Compute max |K_mix| at a single point (t1, φ1=0, t2, φ2=0)."""
    # Build on a small grid centered at the point
    N = 8
    eps = 0.3
    
    # We need the curvature at interior points of the grid.
    # Use the full grid approach but extract a central point.
    result = build_metric_and_curvature(seam_func, gamma, N=N)
    
    # result is (K_mix array, ...) — let me check what it returns
    return result

# Actually, let's use a simpler direct approach: compute the metric
# and curvature tensor at a single point using FD on a small stencil.

def compute_K_single_point(seam_func, gamma, t1, p1, t2, p2, h=1e-5, n_angles=60):
    """Compute K_mix at a single point using central FD for Christoffels."""
    coords = np.array([t1, p1, t2, p2])
    
    st1, ct1 = np.sin(t1), np.cos(t1)
    st2, ct2 = np.sin(t2), np.cos(t2)
    
    # Background metric h
    h_bg = np.diag([1.0, st1**2, 1.0, st2**2])
    
    # Seam at central point
    s0, ds0, d2s0 = seam_func(t1, p1, t2, p2)
    
    # Covariant Hessian
    H = np.zeros((4,4))
    H[0,0] = d2s0[0][0]
    H[0,1] = d2s0[0][1] - (ct1/st1)*ds0[1]
    H[0,2] = d2s0[0][2]
    H[0,3] = d2s0[0][3]
    H[1,0] = H[0,1]
    H[1,1] = d2s0[1][1] + st1*ct1*ds0[0]
    H[1,2] = d2s0[1][2]
    H[1,3] = d2s0[1][3]
    H[2,0] = H[0,2]
    H[2,1] = H[1,2]
    H[2,2] = d2s0[2][2]
    H[2,3] = d2s0[2][3] - (ct2/st2)*ds0[3]
    H[3,0] = H[0,3]
    H[3,1] = H[1,3]
    H[3,2] = H[2,3]
    H[3,3] = d2s0[3][3] + st2*ct2*ds0[2]
    
    G0 = h_bg + gamma * H
    
    # Build metric at shifted points for FD of Christoffel symbols
    def metric_at(dt):
        """Metric at coords + dt"""
        c = coords + dt
        st1_, ct1_ = np.sin(c[0]), np.cos(c[0])
        st2_, ct2_ = np.sin(c[2]), np.cos(c[2])
        h_ = np.diag([1.0, st1_**2, 1.0, st2_**2])
        s_, ds_, d2s_ = seam_func(c[0], c[1], c[2], c[3])
        H_ = np.zeros((4,4))
        H_[0,0] = d2s_[0][0]
        H_[0,1] = d2s_[0][1] - (ct1_/st1_)*ds_[1]
        H_[0,2] = d2s_[0][2]
        H_[0,3] = d2s_[0][3]
        H_[1,0] = H_[0,1]
        H_[1,1] = d2s_[1][1] + st1_*ct1_*ds_[0]
        H_[1,2] = d2s_[1][2]
        H_[1,3] = d2s_[1][3]
        H_[2,0] = H_[0,2]
        H_[2,1] = H_[1,2]
        H_[2,2] = d2s_[2][2]
        H_[2,3] = d2s_[2][3] - (ct2_/st2_)*ds_[3]
        H_[3,0] = H_[0,3]
        H_[3,1] = H_[1,3]
        H_[3,2] = H_[2,3]
        H_[3,3] = d2s_[3][3] + st2_*ct2_*ds_[2]
        return h_ + gamma * H_
    
    # dG/dx^k by central FD
    dG = np.zeros((4,4,4))
    for k in range(4):
        dt_p = np.zeros(4); dt_p[k] = h
        dt_m = np.zeros(4); dt_m[k] = -h
        dG[:,:,k] = (metric_at(dt_p) - metric_at(dt_m)) / (2*h)
    
    # Inverse metric
    Ginv = np.linalg.inv(G0)
    
    # Christoffel symbols
    Gamma = np.zeros((4,4,4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                Gamma[i,j,k] = 0.5 * sum(Ginv[i,l] * (dG[l,j,k] + dG[l,k,j] - dG[j,k,l]) for l in range(4))
    
    # dGamma/dx^l by central FD
    def christoffel_at(dt):
        c = coords + dt
        Gc = metric_at(dt)
        Gc_inv = np.linalg.inv(Gc)
        dGc = np.zeros((4,4,4))
        for k in range(4):
            dtp = np.zeros(4); dtp[k] = h
            dtm = np.zeros(4); dtm[k] = -h
            dGc[:,:,k] = (metric_at(dt+dtp) - metric_at(dt+dtm)) / (2*h)
        Gam = np.zeros((4,4,4))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    Gam[i,j,k] = 0.5 * sum(Gc_inv[i,l] * (dGc[l,j,k] + dGc[l,k,j] - dGc[j,k,l]) for l in range(4))
        return Gam
    
    dGamma = np.zeros((4,4,4,4))
    for l in range(4):
        dt_p = np.zeros(4); dt_p[l] = h
        dt_m = np.zeros(4); dt_m[l] = -h
        dGamma[:,:,:,l] = (christoffel_at(dt_p) - christoffel_at(dt_m)) / (2*h)
    
    # Riemann tensor R^i_{jkl} = dGamma^i_{jl}/dx^k - dGamma^i_{jk}/dx^l + Gamma*Gamma terms
    R = np.zeros((4,4,4,4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    R[i,j,k,l] = dGamma[i,j,l,k] - dGamma[i,j,k,l]
                    for m in range(4):
                        R[i,j,k,l] += Gamma[i,k,m]*Gamma[m,j,l] - Gamma[i,l,m]*Gamma[m,j,k]
    
    # Lower first index: R_{ijkl} = G_{im} R^m_{jkl}
    Riem = np.zeros((4,4,4,4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    Riem[i,j,k,l] = sum(G0[i,m]*R[m,j,k,l] for m in range(4))
    
    # Scan mixed planes: e1 in T_1, e2 in T_2
    Kmax = -1e30
    Kmin = 1e30
    for i in range(n_angles):
        alpha = np.pi * i / n_angles
        e1 = np.array([np.cos(alpha), np.sin(alpha), 0, 0])
        for j in range(n_angles):
            beta = np.pi * j / n_angles
            e2 = np.array([0, 0, np.cos(beta), np.sin(beta)])
            
            # R_{abcd} e1^a e2^b e1^c e2^d  (for K = R_{1212}/(g11*g22 - g12²))
            num = 0.0
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        for d in range(4):
                            num += Riem[a,b,c,d] * e1[a] * e2[b] * e1[c] * e2[d]
            
            g11 = e1 @ G0 @ e1
            g22 = e2 @ G0 @ e2
            g12 = e1 @ G0 @ e2
            denom = g11*g22 - g12**2
            
            if abs(denom) > 1e-15:
                K = num / denom
                Kmax = max(Kmax, K)
                Kmin = min(Kmin, K)
    
    return Kmax, Kmin

# Test point (generic, away from poles)
t1, t2 = 1.0, 2.0
p1, p2 = 0.5, 1.3

print("=" * 70)
print("γ-SCALING TEST: K_max for ℓ=1 seams (expect O(γ³))")
print("=" * 70)

# z₁z₂ seam (rank-1)
coeffs_z1z2 = [0,0,0, 0,0,0, 0,0,1]
seam_z1z2 = make_seam_product_harmonics(coeffs_z1z2)

gammas = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]

print("\ns = z₁z₂ (rank-1 equivariant):")
print(f"{'γ':>8} {'Kmax':>12} {'Kmin':>12} {'Kmax/γ²':>12} {'Kmax/γ³':>12}")
for gam in gammas:
    Kmax, Kmin = compute_K_single_point(seam_z1z2, gam, t1, p1, t2, p2)
    r2 = Kmax/gam**2 if abs(gam) > 1e-15 else 0
    r3 = Kmax/gam**3 if abs(gam) > 1e-15 else 0
    print(f"{gam:8.3f} {Kmax:+12.6e} {Kmin:+12.6e} {r2:+12.6e} {r3:+12.6e}")

# x₁x₂ + y₁y₂ + z₁z₂ = n₁·n₂ (full rank)
coeffs_full = [1,0,0, 0,1,0, 0,0,1]
seam_full = make_seam_product_harmonics(coeffs_full)

print("\ns = n₁·n₂ (full rank, diagonal SO(3) symmetry):")
print(f"{'γ':>8} {'Kmax':>12} {'Kmin':>12} {'Kmax/γ²':>12} {'Kmax/γ³':>12}")
for gam in gammas:
    Kmax, Kmin = compute_K_single_point(seam_full, gam, t1, p1, t2, p2)
    r2 = Kmax/gam**2 if abs(gam) > 1e-15 else 0
    r3 = Kmax/gam**3 if abs(gam) > 1e-15 else 0
    print(f"{gam:8.3f} {Kmax:+12.6e} {Kmin:+12.6e} {r2:+12.6e} {r3:+12.6e}")

# Distinct singular values (genuinely non-equivariant!)
coeffs_distinct = [3,0,0, 0,2,0, 0,0,1]  # σ₁=3, σ₂=2, σ₃=1
seam_distinct = make_seam_product_harmonics(coeffs_distinct)

print("\ns = 3x₁x₂ + 2y₁y₂ + z₁z₂ (distinct σ, genuinely non-equivariant):")
print(f"{'γ':>8} {'Kmax':>12} {'Kmin':>12} {'Kmax/γ²':>12} {'Kmax/γ³':>12}")
for gam in gammas:
    Kmax, Kmin = compute_K_single_point(seam_distinct, gam, t1, p1, t2, p2)
    r2 = Kmax/gam**2 if abs(gam) > 1e-15 else 0
    r3 = Kmax/gam**3 if abs(gam) > 1e-15 else 0
    print(f"{gam:8.3f} {Kmax:+12.6e} {Kmin:+12.6e} {r2:+12.6e} {r3:+12.6e}")

print("\n" + "=" * 70)
print("COMPARISON: ℓ=2 seam (expect O(γ²))")
print("=" * 70)

# ℓ=2: use cos²θ - 1/3 type. Need to build manually.
# Y₂⁰ ∝ 3cos²θ - 1. Let's use s = (3cos²θ₁-1)(3cos²θ₂-1)/4
def make_l2_seam():
    def seam_func(t1, p1, t2, p2):
        st1, ct1 = np.sin(np.atleast_1d(t1)), np.cos(np.atleast_1d(t1))
        st2, ct2 = np.sin(np.atleast_1d(t2)), np.cos(np.atleast_1d(t2))
        
        f1 = 3*ct1**2 - 1
        f2 = 3*ct2**2 - 1
        s = f1 * f2 / 4
        
        df1 = -6*ct1*st1
        df2 = -6*ct2*st2
        d2f1 = -6*(ct1**2 - st1**2)
        d2f2 = -6*(ct2**2 - st2**2)
        
        ds = [None]*4
        ds[0] = df1 * f2 / 4
        ds[1] = np.zeros_like(s)
        ds[2] = f1 * df2 / 4
        ds[3] = np.zeros_like(s)
        
        d2s = [[np.zeros_like(s) for _ in range(4)] for _ in range(4)]
        d2s[0][0] = d2f1 * f2 / 4
        d2s[2][2] = f1 * d2f2 / 4
        d2s[0][2] = df1 * df2 / 4
        d2s[2][0] = d2s[0][2]
        
        if np.ndim(t1) == 0:
            s = s.item()
            ds = [d.item() for d in ds]
            d2s = [[d.item() for d in row] for row in d2s]
        
        return s, ds, d2s
    return seam_func

seam_l2 = make_l2_seam()
print(f"\ns = Y₂⁰(θ₁)·Y₂⁰(θ₂):")
print(f"{'γ':>8} {'Kmax':>12} {'Kmin':>12} {'Kmax/γ²':>12} {'Kmax/γ³':>12}")
for gam in gammas:
    Kmax, Kmin = compute_K_single_point(seam_l2, gam, t1, p1, t2, p2)
    r2 = Kmax/gam**2 if abs(gam) > 1e-15 else 0
    r3 = Kmax/gam**3 if abs(gam) > 1e-15 else 0
    print(f"{gam:8.3f} {Kmax:+12.6e} {Kmin:+12.6e} {r2:+12.6e} {r3:+12.6e}")
