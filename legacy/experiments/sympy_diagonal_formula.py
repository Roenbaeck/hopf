"""
Derive the correct mixed curvature formula for the diagonal case (eta=0)
using SymPy, directly from the 4×4 metric.
"""
import sympy as sp

t1, t2 = sp.symbols('theta_1 theta_2')

# Diagonal metric functions as generic functions of (t1, t2)
L1 = sp.Function('lambda_1')(t1, t2)
L2 = sp.Function('lambda_2')(t1, t2)
M1 = sp.Function('mu_1')(t1, t2)
M2 = sp.Function('mu_2')(t1, t2)

# 4x4 metric: diag(L1, M1*sin^2(t1), L2, M2*sin^2(t2))
# Coordinates: x^0=t1, x^1=phi1, x^2=t2, x^3=phi2
# On orbit space, phi1=phi2=0 (or any fixed value), and the metric
# only depends on (t1, t2).

# The 4x4 metric is diagonal in the diagonal case
g = sp.Matrix([
    [L1, 0, 0, 0],
    [0, M1*sp.sin(t1)**2, 0, 0],
    [0, 0, L2, 0],
    [0, 0, 0, M2*sp.sin(t2)**2]
])

ginv = g.inv()

n = 4
coords = [t1, sp.Symbol('phi_1'), t2, sp.Symbol('phi_2')]

# Only t1 and t2 derivatives are non-zero (everything is independent of phi_k)
# So d/d(phi_k) of any metric component = 0

# Christoffel symbols Gamma^i_{jk} = 1/2 g^{il}(g_{lj,k} + g_{lk,j} - g_{jk,l})
# Since everything is independent of phi_k, only d/dt1 and d/dt2 matter.

def dg(i, j, k):
    """Derivative of g_{ij} with respect to coords[k]"""
    if k in [1, 3]:  # phi derivatives
        # g components depend only on t1, t2
        # But g_{11} = M1*sin^2(t1) depends on t1
        # And g_{33} = M2*sin^2(t2) depends on t2
        return sp.Integer(0)
    return sp.diff(g[i,j], coords[k])

def Gamma(i, j, k):
    """Christoffel symbol Gamma^i_{jk}"""
    val = sp.Integer(0)
    for l in range(n):
        val += sp.Rational(1,2) * ginv[i,l] * (dg(l,j,k) + dg(l,k,j) - dg(j,k,l))
    return val

# R^i_{jkl} = d_k Gamma^i_{jl} - d_l Gamma^i_{jk} + Gamma^i_{km}Gamma^m_{jl} - Gamma^i_{lm}Gamma^m_{jk}
def Riemann_up(i, j, k, l):
    # d_k Gamma^i_{jl}
    term1 = sp.diff(Gamma(i, j, l), coords[k]) if k in [0, 2] else sp.Integer(0)
    # d_l Gamma^i_{jk}
    term2 = sp.diff(Gamma(i, j, k), coords[l]) if l in [0, 2] else sp.Integer(0)
    # Gamma^i_{km}Gamma^m_{jl}
    term3 = sum(Gamma(i, k, m) * Gamma(m, j, l) for m in range(n))
    # Gamma^i_{lm}Gamma^m_{jk}
    term4 = sum(Gamma(i, l, m) * Gamma(m, j, k) for m in range(n))
    return term1 - term2 + term3 - term4

# R_{0202} = g_{0m} R^m_{202}
print("Computing R_{0202}...")
R_0202 = sum(g[0,m] * Riemann_up(m, 2, 0, 2) for m in range(n))
R_0202 = sp.simplify(R_0202)
print(f"R_0202 = {R_0202}")

# Sectional curvature K = R_0202 / (g_00*g_22 - g_02^2) = R_0202 / (L1*L2)
K = sp.simplify(R_0202 / (L1 * L2))
print(f"\nK_{'{02}'} = R_0202/(L1*L2) = {K}")

# Now let's see what it simplifies to in terms of log derivatives
# d^2(log L1)/dt2^2 = (L1*L1_{22} - L1_2^2) / L1^2
# d^2(log L2)/dt1^2 = (L2*L2_{11} - L2_1^2) / L2^2
L1_2 = sp.diff(L1, t2)
L1_22 = sp.diff(L1, t2, 2)
L2_1 = sp.diff(L2, t1)
L2_11 = sp.diff(L2, t1, 2)

logL1_22 = (L1*L1_22 - L1_2**2) / L1**2
logL2_11 = (L2*L2_11 - L2_1**2) / L2**2

formula_A = -sp.Rational(1,2) * (logL1_22 + logL2_11)
formula_B = (-L2/2*logL1_22 - L1/2*logL2_11) / (L1*L2)

print(f"\nFormula A = -1/2*(d2^2 logL1 + d1^2 logL2) = {sp.simplify(formula_A)}")
print(f"Formula B = [-L2*d2^2(logL1) - L1*d1^2(logL2)]/(2*L1*L2) = {sp.simplify(formula_B)}")

# Check if K equals formula A or B
print(f"\nK - formula_A = {sp.simplify(K - formula_A)}")
print(f"K - formula_B = {sp.simplify(K - formula_B)}")
