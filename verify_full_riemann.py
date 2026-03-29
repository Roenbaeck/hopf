"""Sanity check: verify Riemann computation against known results,
then scan K for all mixed planes using FULL Riemann (not quadratic formula).
"""
import numpy as np

h_fd = 1e-5

def metric_general(pt, gamma_val, seam_type='isotropic'):
    t1, p1, t2, p2 = pt
    st1, ct1 = np.sin(t1), np.cos(t1)
    st2, ct2 = np.sin(t2), np.cos(t2)
    dp = p1 - p2
    cd, sd = np.cos(dp), np.sin(dp)
    s = st1*st2*cd + ct1*ct2
    lam = 1 - gamma_val * s
    A = ct1*ct2*cd + st1*st2
    B = ct1*st2*sd
    C = -st1*ct2*sd
    D = st1*st2*cd
    g = np.zeros((4, 4))
    g[0, 0] = lam
    g[1, 1] = lam * st1**2
    g[2, 2] = lam
    g[3, 3] = lam * st2**2
    g[0, 2] = g[2, 0] = gamma_val * A
    g[0, 3] = g[3, 0] = gamma_val * B
    g[1, 2] = g[2, 1] = gamma_val * C
    g[1, 3] = g[3, 1] = gamma_val * D
    return g

def full_riemann_tensor(pt, gamma_val):
    """Compute R_{ijkl} via finite differences."""
    def met(p):
        return metric_general(p, gamma_val)
    
    g = met(pt)
    gi = np.linalg.inv(g)
    
    # Christoffel symbols of first kind
    dg = np.zeros((4, 4, 4))
    for k in range(4):
        ep = pt.copy(); ep[k] += h_fd
        em = pt.copy(); em[k] -= h_fd
        dg[:, :, k] = (met(ep) - met(em)) / (2*h_fd)
    
    G1 = np.zeros((4, 4, 4))
    for K in range(4):
        for I in range(4):
            for J in range(I, 4):
                v = 0.5*(dg[K,I,J] + dg[K,J,I] - dg[I,J,K])
                G1[K,I,J] = v; G1[K,J,I] = v
    
    # Upper Christoffel
    Gup = np.einsum('km,mij->kij', gi, G1)
    
    # Derivative of upper Christoffel
    def Gup_at(p):
        gp = met(p)
        gip = np.linalg.inv(gp)
        dgp = np.zeros((4,4,4))
        for k in range(4):
            ep2 = p.copy(); ep2[k] += h_fd
            em2 = p.copy(); em2[k] -= h_fd
            dgp[:,:,k] = (met(ep2)-met(em2))/(2*h_fd)
        G1p = np.zeros((4,4,4))
        for K in range(4):
            for I in range(4):
                for J in range(I,4):
                    v = 0.5*(dgp[K,I,J]+dgp[K,J,I]-dgp[I,J,K])
                    G1p[K,I,J]=v; G1p[K,J,I]=v
        return np.einsum('km,mij->kij', gip, G1p)
    
    dGup = np.zeros((4,4,4,4))
    for d in range(4):
        ep = pt.copy(); ep[d] += h_fd
        em = pt.copy(); em[d] -= h_fd
        dGup[:,:,:,d] = (Gup_at(ep) - Gup_at(em))/(2*h_fd)
    
    # R^m_{jkl} = dG^m_{jl,k} - dG^m_{jk,l} + G^m_{ke}G^e_{jl} - G^m_{le}G^e_{jk}
    R4 = np.zeros((4,4,4,4))
    for j in range(4):
        for k in range(4):
            for l in range(4):
                Rm = np.zeros(4)
                for m in range(4):
                    Rm[m] = dGup[m,j,l,k] - dGup[m,j,k,l]
                    for e in range(4):
                        Rm[m] += Gup[m,k,e]*Gup[e,j,l] - Gup[m,l,e]*Gup[e,j,k]
                for i in range(4):
                    R4[i,j,k,l] += g[i,:] @ Rm  # should be sum g[i,m]*Rm[m]
                    break  # compute once, assign
                for i in range(4):
                    R4[i,j,k,l] = sum(g[i,m]*Rm[m] for m in range(4))
    return R4, g

def sectional_K(R4, g, X, Y):
    """K(X,Y) from precomputed Riemann tensor."""
    R_XYXY = np.einsum('ijkl,i,j,k,l', R4, X, Y, X, Y)
    gXX = X @ g @ X
    gYY = Y @ g @ Y
    gXY = X @ g @ Y
    denom = gXX*gYY - gXY**2
    if abs(denom) < 1e-14:
        return float('nan')
    return R_XYXY / denom

# ==========================================
# SANITY CHECK 1: Round S^2 x S^2 (gamma=0)
# ==========================================
print("="*60)
print("SANITY CHECK: gamma=0 (round product)")
print("="*60)
pt = np.array([1.0, 0.5, 1.2, 0.8])
R4, g = full_riemann_tensor(pt, 0.0)

# Same-factor plane on S^2_1: should give K=1
X = np.array([1.0, 0, 0, 0])
Y = np.array([0, 1.0, 0, 0])
print(f"K(e_t1, e_p1) = {sectional_K(R4, g, X, Y):.6f}  (expect +1)")

X = np.array([0, 0, 1.0, 0])
Y = np.array([0, 0, 0, 1.0])
print(f"K(e_t2, e_p2) = {sectional_K(R4, g, X, Y):.6f}  (expect +1)")

# Mixed plane: should give K=0
X = np.array([1.0, 0, 0, 0])
Y = np.array([0, 0, 1.0, 0])
print(f"K(e_t1, e_t2) = {sectional_K(R4, g, X, Y):.6f}  (expect 0)")

X = np.array([0, 1.0, 0, 0])
Y = np.array([0, 0, 0, 1.0])
print(f"K(e_p1, e_p2) = {sectional_K(R4, g, X, Y):.6f}  (expect 0)")

# General mixed: also K=0
X = np.array([0.6, 0.8, 0, 0])
Y = np.array([0, 0, 0.7, 0.3])
print(f"K(gen_mix)    = {sectional_K(R4, g, X, Y):.6f}  (expect 0)")

# ==========================================
# MAIN TEST: gamma=0.4, scan all mixed planes
# ==========================================
print()
print("="*60)
print("MAIN TEST: gamma=0.4, full Riemann scan")
print("="*60)
gamma_val = 0.4
min_K = float('inf')
min_info = None
n_total = 0

for t1 in np.linspace(0.3, np.pi-0.3, 5):
    for t2 in np.linspace(0.3, np.pi-0.3, 5):
        for p1 in [0.0, 1.0, 2.5]:
            for p2 in [0.0, 1.0, 2.5]:
                pt = np.array([t1, p1, t2, p2])
                R4, g = full_riemann_tensor(pt, gamma_val)
                
                for theta in np.linspace(0, np.pi, 8):
                    for phi in np.linspace(0, np.pi, 8):
                        ct, st = np.cos(theta), np.sin(theta)
                        cp, sp = np.cos(phi), np.sin(phi)
                        X = np.array([ct, st, 0, 0])
                        Y = np.array([0, 0, cp, sp])
                        K = sectional_K(R4, g, X, Y)
                        if np.isnan(K):
                            continue
                        n_total += 1
                        if K < min_K:
                            min_K = K
                            min_info = (t1, p1, t2, p2, theta, phi, K)

print(f"Planes tested: {n_total}")
print(f"Min K = {min_K:.10f}")
if min_info:
    print(f"  at ({min_info[0]:.2f},{min_info[1]:.2f},{min_info[2]:.2f},"
          f"{min_info[3]:.2f}, th={min_info[4]:.2f}, ph={min_info[5]:.2f})")
if min_K > -1e-6:
    print("RESULT: K >= 0 for all tested planes (consistent with conjecture)")
else:
    print(f"RESULT: K < 0 found! Conjecture may be FALSE")
