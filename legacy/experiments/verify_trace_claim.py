"""
Test FEEDBACK's key claim: for any fixed V in T_{p2}S^2,
the trace A + C of K_mix(X, V) as X varies vanishes.

A = K_mix(e1, V), C = K_mix(e2, V) for orthonormal basis {e1, e2} of T_{p1}S^2.

This contradicts K_mix >= 0 (proved analytically) + K_max > 0 (verified), 
which forces A, C >= 0 and hence A + C > 0 for the "strong" direction V.
"""
import numpy as np

def seam_metric_components(th1, ph1, th2, ph2, sigma, gamma):
    """
    Compute the 4x4 metric g = h + gamma * nabla^2 s on S^2 x S^2
    at point (th1, ph1, th2, ph2) in spherical coordinates.
    s = sigma[0]*x1*x2 + sigma[1]*y1*y2 + sigma[2]*z1*z2
    """
    s1, s2, s3 = sigma
    ct1, st1 = np.cos(th1), np.sin(th1)
    cp1, sp1 = np.cos(ph1), np.sin(ph1)
    ct2, st2 = np.cos(th2), np.sin(th2)
    cp2, sp2 = np.cos(ph2), np.sin(ph2)
    
    # Cartesian coordinates
    x1, y1, z1 = st1*cp1, st1*sp1, ct1
    x2, y2, z2 = st2*cp2, st2*sp2, ct2
    
    s_val = s1*x1*x2 + s2*y1*y2 + s3*z1*z2
    
    # Background metric: diag(1, sin^2(th1), 1, sin^2(th2))
    h = np.diag([1.0, st1**2, 1.0, st2**2])
    
    # Compute nabla^2 s numerically via finite differences
    eps = 1e-6
    coords = np.array([th1, ph1, th2, ph2])
    
    def s_at(c):
        ct1_, st1_ = np.cos(c[0]), np.sin(c[0])
        cp1_, sp1_ = np.cos(c[1]), np.sin(c[1])
        ct2_, st2_ = np.cos(c[2]), np.sin(c[2])
        cp2_, sp2_ = np.cos(c[3]), np.sin(c[3])
        x1_ = st1_*cp1_; y1_ = st1_*sp1_; z1_ = ct1_
        x2_ = st2_*cp2_; y2_ = st2_*sp2_; z2_ = ct2_
        return s1*x1_*x2_ + s2*y1_*y2_ + s3*z1_*z2_
    
    # Compute Christoffel symbols of the background metric
    # For S^2 in (theta, phi): Gamma^theta_{phi,phi} = -sin(theta)*cos(theta)
    #                           Gamma^phi_{theta,phi} = cos(theta)/sin(theta)
    # All others zero.
    
    # Covariant Hessian: nabla_i nabla_j s = d_i d_j s - Gamma^k_{ij} d_k s
    # Compute partial derivatives
    ds = np.zeros(4)
    for i in range(4):
        cp = coords.copy(); cm = coords.copy()
        cp[i] += eps; cm[i] -= eps
        ds[i] = (s_at(cp) - s_at(cm)) / (2*eps)
    
    d2s = np.zeros((4,4))
    for i in range(4):
        for j in range(i, 4):
            cp = coords.copy(); cm = coords.copy()
            if i == j:
                cp[i] += eps; cm[i] -= eps
                d2s[i,j] = (s_at(cp) - 2*s_at(coords) + s_at(cm)) / eps**2
            else:
                cpp = coords.copy(); cpm = coords.copy()
                cmp = coords.copy(); cmm = coords.copy()
                cpp[i] += eps; cpp[j] += eps
                cpm[i] += eps; cpm[j] -= eps
                cmp[i] -= eps; cmp[j] += eps
                cmm[i] -= eps; cmm[j] -= eps
                d2s[i,j] = (s_at(cpp) - s_at(cpm) - s_at(cmp) + s_at(cmm)) / (4*eps**2)
            d2s[j,i] = d2s[i,j]
    
    # Christoffel symbols (nonzero ones for product S^2 x S^2)
    Gamma = np.zeros((4,4,4))
    # Factor 1: (theta1=0, phi1=1)
    Gamma[0,1,1] = -st1*ct1  # Gamma^theta1_{phi1,phi1}
    Gamma[1,0,1] = ct1/st1 if abs(st1) > 1e-10 else 0  # Gamma^phi1_{theta1,phi1}
    Gamma[1,1,0] = Gamma[1,0,1]
    # Factor 2: (theta2=2, phi2=3)
    Gamma[2,3,3] = -st2*ct2
    Gamma[3,2,3] = ct2/st2 if abs(st2) > 1e-10 else 0
    Gamma[3,3,2] = Gamma[3,2,3]
    
    # Covariant Hessian
    H = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            H[i,j] = d2s[i,j]
            for k in range(4):
                H[i,j] -= Gamma[k,i,j] * ds[k]
    
    g = h + gamma * H
    return g


def mixed_sectional_curvature(th1, ph1, th2, ph2, sigma, gamma, X, V):
    """
    Compute K_mix for mixed plane spanned by X (tangent to S^2_1) and V (tangent to S^2_2).
    X is a 2-vector in (d/dth1, d/dph1) basis.
    V is a 2-vector in (d/dth2, d/dph2) basis.
    Uses finite differences on the Riemann tensor.
    """
    eps = 1e-5
    coords = np.array([th1, ph1, th2, ph2])
    
    def get_g(c):
        return seam_metric_components(c[0], c[1], c[2], c[3], sigma, gamma)
    
    g = get_g(coords)
    
    # Full tangent vectors in 4D
    Xfull = np.array([X[0], X[1], 0, 0])
    Vfull = np.array([0, 0, V[0], V[1]])
    
    # Inverse metric
    ginv = np.linalg.inv(g)
    
    # Christoffel symbols via FD
    def christoffel(c):
        gc = get_g(c)
        gcinv = np.linalg.inv(gc)
        dg = np.zeros((4,4,4))  # dg[k,i,j] = d g_{ij}/d x^k
        for k in range(4):
            cp = c.copy(); cm = c.copy()
            cp[k] += eps; cm[k] -= eps
            dg[k] = (get_g(cp) - get_g(cm)) / (2*eps)
        
        G = np.zeros((4,4,4))  # Gamma^a_{bc}
        for a in range(4):
            for b in range(4):
                for c_ in range(4):
                    for d in range(4):
                        G[a,b,c_] += 0.5 * gcinv[a,d] * (dg[b,d,c_] + dg[c_,d,b] - dg[d,b,c_])
        return G
    
    Gam = christoffel(coords)
    
    # Riemann tensor via FD of Christoffel symbols
    # R^a_{bcd} = d_c Gamma^a_{bd} - d_d Gamma^a_{bc} + Gamma^a_{ce}Gamma^e_{bd} - Gamma^a_{de}Gamma^e_{bc}
    
    dGam = np.zeros((4,4,4,4))  # dGam[k,a,b,c] = d_k Gamma^a_{bc}
    for k in range(4):
        cp = coords.copy(); cm = coords.copy()
        cp[k] += eps; cm[k] -= eps
        dGam[k] = (christoffel(cp) - christoffel(cm)) / (2*eps)
    
    R = np.zeros((4,4,4,4))  # R^a_{bcd}
    for a in range(4):
        for b in range(4):
            for c_ in range(4):
                for d in range(4):
                    R[a,b,c_,d] = dGam[c_,a,b,d] - dGam[d,a,b,c_]
                    for e in range(4):
                        R[a,b,c_,d] += Gam[a,c_,e]*Gam[e,b,d] - Gam[a,d,e]*Gam[e,b,c_]
    
    # R_{abcd} = g_{ae} R^e_{bcd}
    Rdown = np.zeros((4,4,4,4))
    for a in range(4):
        for b in range(4):
            for c_ in range(4):
                for d in range(4):
                    for e in range(4):
                        Rdown[a,b,c_,d] += g[a,e] * R[e,b,c_,d]
    
    # K = R(X,V,V,X) / (g(X,X)*g(V,V) - g(X,V)^2)
    num = 0
    for a in range(4):
        for b in range(4):
            for c_ in range(4):
                for d in range(4):
                    num += Rdown[a,b,c_,d] * Xfull[a] * Vfull[b] * Vfull[c_] * Xfull[d]
    
    gXX = Xfull @ g @ Xfull
    gVV = Vfull @ g @ Vfull
    gXV = Xfull @ g @ Vfull
    denom = gXX * gVV - gXV**2
    
    return num / denom if abs(denom) > 1e-15 else 0


def test_trace_claim():
    """Test: for fixed V, does K_mix(e1, V) + K_mix(e2, V) = 0?"""
    print("="*60)
    print("Test: does tr Q = K(e1,V) + K(e2,V) = 0 for fixed V?")
    print("="*60)
    
    gamma = 0.1
    
    test_cases = [
        ("isotropic", [1, 1, 1]),
        ("distinct", [3, 2, 1]),
        ("partial", [2, 1, 1]),
    ]
    
    for name, sigma in test_cases:
        print(f"\n--- Case: {name}, sigma = {sigma} ---")
        
        # Several test points
        points = [
            (0.7, 0.5, 1.2, 0.8),
            (1.0, 0.0, 1.5, 1.0),
            (0.5, 1.5, 0.8, 2.0),
            (1.3, 0.3, 0.6, 1.7),
        ]
        
        for th1, ph1, th2, ph2 in points:
            st1 = np.sin(th1)
            st2 = np.sin(th2)
            
            # Orthonormal basis for T_{p1}S^2: e1 = d/dth1, e2 = (1/sin th1) d/dph1
            # In coordinate basis: e1 = (1, 0), e2 = (0, 1/sin(th1))
            e1 = np.array([1.0, 0.0])
            e2 = np.array([0.0, 1.0/st1])
            
            # Several choices of V
            for beta in [0, np.pi/4, np.pi/3, np.pi/2]:
                # V = cos(beta) * f1 + sin(beta) * f2
                # f1 = d/dth2, f2 = (1/sin th2) d/dph2
                f1 = np.array([1.0, 0.0])
                f2 = np.array([0.0, 1.0/st2])
                V = np.cos(beta) * f1 + np.sin(beta) * f2
                
                K1 = mixed_sectional_curvature(th1, ph1, th2, ph2, sigma, gamma, e1, V)
                K2 = mixed_sectional_curvature(th1, ph1, th2, ph2, sigma, gamma, e2, V)
                
                trace = K1 + K2
                print(f"  pt=({th1:.1f},{ph1:.1f},{th2:.1f},{ph2:.1f}), "
                      f"beta={beta:.2f}: K(e1,V)={K1:+.6e}, K(e2,V)={K2:+.6e}, "
                      f"TRACE={trace:+.6e}  {'ZERO' if abs(trace) < 1e-6 else 'NONZERO!'}")
    
    print("\n" + "="*60)
    print("If trace != 0, the FEEDBACK's proof is WRONG.")
    print("Note: we proved K_mix >= 0 for all l=1 seams.")
    print("If trace = 0 and K_mix >= 0, then ALL K_mix = 0,")
    print("contradicting K_max > 0.")
    print("="*60)


if __name__ == "__main__":
    test_trace_claim()
