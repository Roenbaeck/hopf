"""
Analyze R_{0202} at the critical point of Psi = (1/2) log(lam1*lam2 - eta^2).

Strategy:
1. Compute R_{0202} for the 4x4 metric with off-diagonal eta (same as
   compute_mixed_curvature.py but using plain symbols for derivatives).
2. At the maximum of Psi, impose:
   - First-order: lam2 * d_k(lam1) + lam1 * d_k(lam2) - 2*eta * d_k(eta) = 0
   - Second-order: d_k^2 Psi <= 0  (Hessian negative semidefinite)
3. Substitute and check the sign of R_{0202}.

We use abstract symbols for the metric coefficients and their derivatives
at a single point, avoiding Function objects for cleaner algebra.
"""

import sympy as sp
from sympy import symbols, Matrix, Rational, simplify, expand, collect, factor, latex

# ── Metric coefficients at a point ──
L1, L2, M1, M2, E = symbols('lambda1 lambda2 mu1 mu2 eta', positive=True)

# First derivatives (d1 = d/dtheta1, d2 = d/dtheta2)
L1_1, L1_2 = symbols('lambda1_1 lambda1_2')  # d_1 lam1, d_2 lam1
L2_1, L2_2 = symbols('lambda2_1 lambda2_2')
M1_1, M1_2 = symbols('mu1_1 mu1_2')
M2_1, M2_2 = symbols('mu2_1 mu2_2')
E_1, E_2   = symbols('eta_1 eta_2')

# Second derivatives
L1_11, L1_22, L1_12 = symbols('lambda1_11 lambda1_22 lambda1_12')
L2_11, L2_22, L2_12 = symbols('lambda2_11 lambda2_22 lambda2_12')
M1_1d, M1_2d = symbols('mu1_1d mu1_2d')  # not needed for R_{0202}
M2_1d, M2_2d = symbols('mu2_1d mu2_2d')
E_11, E_22, E_12 = symbols('eta_11 eta_22 eta_12')

# ── Metric at theta1=theta1_0, theta2=theta2_0 ──
# For the mixed curvature R_{0202} we only need the (theta1, theta2) block
# of the 4x4 metric, since phi1, phi2 decouple in the Christoffel symbols
# that contribute to R_{0202}.
#
# Actually, the full 4x4 metric is needed because Gamma^i_{jk} involves
# the full inverse. BUT g_{1j} = 0 for j != 1 and g_{3j} = 0 for j != 3,
# and g_{11}, g_{33} only depend on theta's (plus sin^2 factor).
# The Riemann component R_{0202} only involves Christoffel symbols with
# indices 0 and 2 in the derivative slots, so phi-direction Christoffels
# don't contribute. Let me verify this by doing the full computation.

# Actually, for a block-diagonal structure where phi1, phi2 are Killing
# directions (the metric coefficients don't depend on phi), the mixed
# curvature R_{0202} only involves the 2x2 block (theta1, theta2).
# This is because Gamma^1_{02} = Gamma^3_{02} = 0, etc.
#
# Wait - let me be more careful. Since mu1, mu2 depend on (theta1, theta2),
# there CAN be nonzero Christoffels like Gamma^1_{01} ~ d(mu1*sin^2)/dtheta
# but these don't appear in R_{0202} = R^i_{202} g_{0i}.
#
# Let me just redo the full 4x4 computation with symbols.

# We need sin(theta1), sin(theta2) values at the critical point.
# But these are just constants at the evaluation point. Let's call them s1, s2.
s1, s2 = symbols('s1 s2', positive=True)

# 4x4 metric  (indices: 0=theta1, 1=phi1, 2=theta2, 3=phi2)
g = Matrix([
    [L1,           0,      E,            0          ],
    [0,            M1*s1**2, 0,          0          ],
    [E,            0,      L2,           0          ],
    [0,            0,      0,            M2*s2**2   ]
])

# Inverse
Delta = L1*L2 - E**2   # determinant of the (0,2) block
ginv = Matrix([
    [ L2/Delta,    0,           -E/Delta,      0            ],
    [ 0,           1/(M1*s1**2), 0,            0            ],
    [-E/Delta,     0,            L1/Delta,     0            ],
    [ 0,           0,            0,            1/(M2*s2**2) ]
])

# Verify g * ginv = I
assert simplify(g * ginv - sp.eye(4)) == sp.zeros(4)
print("Inverse metric verified.")

# ── Derivatives of metric components ──
# Build a dictionary: dg[i,j,k] = d(g_{ij})/d(x^k)  (k=0 or k=2 only matter)
# Since nothing depends on phi1 (x^1) or phi2 (x^3), those partials are zero.

# We need cos(theta) for d(sin^2(theta))/dtheta = 2 sin cos = sin(2*theta)
c1, c2 = symbols('c1 c2')  # cos(theta1), cos(theta2) at the point

# d(g_{ij})/dx^0  and  d(g_{ij})/dx^2
# g_{00} = L1:   dg00/dx0 = L1_1,  dg00/dx2 = L1_2
# g_{11} = M1*s1^2:  dg11/dx0 = M1_1*s1^2 + M1*2*s1*c1 = (M1_1 + 2*M1*c1/s1)*s1^2
#                      actually dg11/dx0 = d(M1*sin^2(theta1))/dtheta1
#                      = M1_1 * sin^2 + M1 * 2*sin*cos = s1^2 * M1_1 + 2*M1*s1*c1
# etc.

# For Christoffel symbols Gamma^i_{jk}, we need d(g_{lj})/dx^k for various l,j,k.
# Since R_{0202} = R^i_{202},  only Christoffel symbols Gamma^i_{j,k} with
# certain index combinations matter:
#   R^i_{202} = d_0 Gamma^i_{22} - d_2 Gamma^i_{20} + Gamma^i_{0m} Gamma^m_{22} - Gamma^i_{2m} Gamma^m_{20}
#
# The key insight: for m=1 or m=3 (phi directions), Gamma^m_{jk} = 0 when j,k in {0,2}
# because dg_{lj}/dx^k would need l=1 or l=3, and those metric components only
# have g_{11} and g_{33} nonzero, but dg_{11}/dx^2 and dg_{33}/dx^0 contribute.
#
# Actually, let me just be careful and compute everything.
# Gamma^m_{jk} with m=1: Gamma^1_{jk} = (1/2) g^{1l} (dg_{lj}/dx^k + dg_{lk}/dx^j - dg_{jk}/dx^l)
#   g^{1l} = 0 except g^{11} = 1/(M1*s1^2)
#   So Gamma^1_{jk} = (1/2)/(M1*s1^2) * (dg_{1j}/dx^k + dg_{1k}/dx^j - dg_{jk}/dx^1)
#   g_{1j} = 0 unless j=1, so dg_{1j}/dx^k = 0 unless j=1
#   dg_{jk}/dx^1 = 0 (nothing depends on phi1)
#   So Gamma^1_{jk} = 0 unless j=1 or k=1.
#   For R^i_{202}: j,k in {0,2}, so Gamma^1_{02} = Gamma^1_{20} = Gamma^1_{00} = Gamma^1_{22} = 0
#   Similarly Gamma^3_{jk} = 0 for j,k in {0,2}.
#
# Also need Gamma^i_{0m} Gamma^m_{22} and Gamma^i_{2m} Gamma^m_{20}:
#   For m=1: Gamma^m_{22} = Gamma^1_{22} = 0, Gamma^m_{20} = Gamma^1_{20} = 0
#   For m=3: same = 0
#   So these sums only have m=0, m=2 contributions.
#
# Conclusion: R_{0202} depends only on Gamma^i_{jk} with i,j,k in {0,2}.
# These only involve g^{il} for l in {0,2}, and dg_{lj}/dx^k for l,j,k in {0,2}.
# So we only need the 2x2 block!

print("R_{0202} depends only on the (theta1, theta2) 2x2 block.")
print()

# ── 2x2 block computation ──
g2 = Matrix([[L1, E], [E, L2]])
g2inv = Matrix([[L2, -E], [-E, L1]]) / Delta

assert simplify(g2 * g2inv - sp.eye(2)) == sp.zeros(2)
print("2x2 inverse verified.")

# Coordinates for the 2x2 block: x^0 = theta1, x^1 = theta2
# (renaming to avoid confusion with the 4x4 indices)
# Derivatives: subscript 1 -> d/dtheta1, subscript 2 -> d/dtheta2

# Metric derivatives dict: dg2[a,b,c] = d(g2_{ab})/dx^c
# a,b,c in {0,1} in the 2x2 sense
# x^0 = theta1, x^1 = theta2

# g2_{00} = L1: dg2[0,0,0] = L1_1,  dg2[0,0,1] = L1_2
# g2_{01} = E:  dg2[0,1,0] = E_1,   dg2[0,1,1] = E_2
# g2_{11} = L2: dg2[1,1,0] = L2_1,  dg2[1,1,1] = L2_2

dg = {}
dg[0,0,0] = L1_1;  dg[0,0,1] = L1_2
dg[0,1,0] = E_1;   dg[0,1,1] = E_2
dg[1,0,0] = E_1;   dg[1,0,1] = E_2
dg[1,1,0] = L2_1;  dg[1,1,1] = L2_2

# Second derivatives: ddg[a,b,c,d] = d^2(g2_{ab})/dx^c dx^d
ddg = {}
ddg[0,0,0,0] = L1_11;  ddg[0,0,0,1] = L1_12;  ddg[0,0,1,0] = L1_12;  ddg[0,0,1,1] = L1_22
ddg[0,1,0,0] = E_11;   ddg[0,1,0,1] = E_12;   ddg[0,1,1,0] = E_12;   ddg[0,1,1,1] = E_22
ddg[1,0,0,0] = E_11;   ddg[1,0,0,1] = E_12;   ddg[1,0,1,0] = E_12;   ddg[1,0,1,1] = E_22
ddg[1,1,0,0] = L2_11;  ddg[1,1,0,1] = L2_12;  ddg[1,1,1,0] = L2_12;  ddg[1,1,1,1] = L2_22

# Christoffel symbols for 2x2 block
def Gamma2(i, j, k):
    val = sp.S(0)
    for l in range(2):
        val += Rational(1,2) * g2inv[i,l] * (dg[l,j,k] + dg[l,k,j] - dg[j,k,l])
    return val

# Precompute
G = [[[None]*2 for _ in range(2)] for _ in range(2)]
for i in range(2):
    for j in range(2):
        for k in range(j, 2):
            v = sp.cancel(Gamma2(i, j, k))
            G[i][j][k] = v
            G[i][k][j] = v

print("Christoffel symbols computed.")

# Derivative of Christoffel wrt x^c:
# d_c Gamma^i_{jk} = (terms involving second derivatives of g and products of first derivatives)
# Instead of differentiating symbolically (we'd need derivatives of ginv too),
# let's compute Gamma^i_{jk} as rational expressions and differentiate manually.
#
# Actually, it's easier to compute d_c Gamma^i_{jk} directly from its definition.
# d_c [(1/2) g^{il} (dg_{lj,k} + dg_{lk,j} - dg_{jk,l})]
#   = (1/2) (d_c g^{il}) (dg_{lj,k} + ...) + (1/2) g^{il} (ddg_{lj,kc} + ddg_{lk,jc} - ddg_{jk,lc})
#
# And d_c g^{il} = - g^{im} Gamma^l_{mc} ... no wait, that's circular.
# Better: d_c g^{il} = - g^{im} g^{ln} dg_{mn,c}

# Let's define dginv[i,l,c] = d(g2inv_{il})/dx^c
def dginv(i, l, c):
    val = sp.S(0)
    for m in range(2):
        for n in range(2):
            val -= g2inv[i,m] * g2inv[l,n] * dg[m,n,c]
    return val

# Derivative of Christoffel:  d_c Gamma^i_{jk}
def dGamma(i, j, k, c):
    val = sp.S(0)
    for l in range(2):
        val += Rational(1,2) * dginv(i, l, c) * (dg[l,j,k] + dg[l,k,j] - dg[j,k,l])
        val += Rational(1,2) * g2inv[i,l] * (ddg[l,j,k,c] + ddg[l,k,j,c] - ddg[j,k,l,c])
    return val

# Riemann tensor in 2D:  R^i_{jkl} = d_k G^i_{jl} - d_l G^i_{jk} + G^i_{km} G^m_{jl} - G^i_{lm} G^m_{jk}
# We want R^0_{101} and R^1_{101} (in 2x2 indices), then
# R_{0101} = g_{00} R^0_{101} + g_{01} R^1_{101}  (= R_{0202} in 4x4)

# R^i_{101}: j=1, k=0, l=1
def Riemann_up(i, j, k, l):
    val = dGamma(i, j, l, k) - dGamma(i, j, k, l)
    for m in range(2):
        val += G[i][k][m] * G[m][j][l] - G[i][l][m] * G[m][j][k]
    return val

print("Computing R^0_{101} (= R^{theta1}_{theta2, theta1, theta2} in 2x2)...")
R0_101 = Riemann_up(0, 1, 0, 1)
R0_101 = sp.cancel(R0_101)
print("Done.")

print("Computing R^1_{101}...")
R1_101 = Riemann_up(1, 1, 0, 1)
R1_101 = sp.cancel(R1_101)
print("Done.")

# R_{0101} = g_{00} R^0_{101} + g_{01} R^1_{101}
R_0101 = g2[0,0] * R0_101 + g2[0,1] * R1_101
R_0101 = sp.cancel(R_0101)

print("\n" + "="*60)
print("R_{0202} (= R_{0101} in 2x2 notation) =")
print("="*60)

# Factor out Delta from denominator
# R_0101 should be a rational function with denominator involving Delta^2
R_num, R_den = sp.fraction(sp.cancel(R_0101))
print(f"\nNumerator has {len(sp.Add.make_args(sp.expand(R_num)))} terms")
print(f"Denominator = {sp.factor(R_den)}")

# ── Critical point conditions for Psi = (1/2) log(lam1*lam2 - eta^2) ──
# Let D = lam1*lam2 - eta^2 = Delta
# d_1 Psi = (L2*L1_1 + L1*L2_1 - 2*E*E_1) / (2*Delta) = 0
# d_2 Psi = (L2*L1_2 + L1*L2_2 - 2*E*E_2) / (2*Delta) = 0
# => L2*L1_k + L1*L2_k - 2*E*E_k = 0 for k=1,2

print("\n" + "="*60)
print("Applying critical point conditions: d_k Psi = 0")
print("="*60)

# Solve for E_1 and E_2 in terms of the lambda derivatives
E1_crit = (L2*L1_1 + L1*L2_1) / (2*E)
E2_crit = (L2*L1_2 + L1*L2_2) / (2*E)

print(f"\neta_1 = (lam2*lam1_1 + lam1*lam2_1) / (2*eta)")
print(f"eta_2 = (lam2*lam1_2 + lam1*lam2_2) / (2*eta)")

# Substitute into R_0101
R_crit = R_0101.subs([(E_1, E1_crit), (E_2, E2_crit)])
R_crit = sp.cancel(R_crit)

print("\nR_{0202} at critical point (after substituting d_k(eta)):")
R_crit_num, R_crit_den = sp.fraction(sp.cancel(R_crit))
R_crit_num_expanded = sp.expand(R_crit_num)
R_crit_den_factored = sp.factor(R_crit_den)
nterms = len(sp.Add.make_args(R_crit_num_expanded))
print(f"Numerator has {nterms} terms")
print(f"Denominator = {R_crit_den_factored}")

# ── Second-order conditions ──
# d_1^2 Psi <= 0 at the maximum.
# d_k^2 Psi = [Delta*(L2*L1_kk + L1*L2_kk + 2*L1_k*L2_k - 2*E*E_kk - 2*E_k^2)
#              - (L2*L1_k + L1*L2_k - 2*E*E_k)^2] / (2*Delta^2)
# At the critical point, the second term is zero, so:
# d_k^2 Psi = [L2*L1_kk + L1*L2_kk + 2*L1_k*L2_k - 2*E*E_kk - 2*E_k^2] / (2*Delta)
# This must be <= 0, so (since Delta > 0):
# L2*L1_kk + L1*L2_kk + 2*L1_k*L2_k - 2*E*E_kk - 2*E_k^2 <= 0

# Also the mixed second derivative: d_1 d_2 Psi at critical point
# d_1 d_2 Psi = [L2*L1_12 + L1*L2_12 + L1_1*L2_2 + L1_2*L2_1 - 2*E*E_12 - 2*E_1*E_2] / (2*Delta)
# The full Hessian of Psi must be negative semidefinite.

# Let's substitute the E_k values back to express the Hessian condition in terms of
# lambda derivatives only.

# At critical point, E_k = (L2*L1_k + L1*L2_k)/(2*E), so
# E_k^2 = (L2*L1_k + L1*L2_k)^2 / (4*E^2)

# Define Hessian numerators (must be <= 0):
# H_kk = L2*L1_kk + L1*L2_kk + 2*L1_k*L2_k - 2*E*E_kk - 2*E_k^2
# where E_k = (L2*L1_k + L1*L2_k) / (2*E)

# Substitute E_k:
H_11 = L2*L1_11 + L1*L2_11 + 2*L1_1*L2_1 - 2*E*E_11 - 2*E1_crit**2
H_22 = L2*L1_22 + L1*L2_22 + 2*L1_2*L2_2 - 2*E*E_22 - 2*E2_crit**2
H_12 = L2*L1_12 + L1*L2_12 + L1_1*L2_2 + L1_2*L2_1 - 2*E*E_12 - 2*E1_crit*E2_crit
H_11 = sp.expand(H_11)
H_22 = sp.expand(H_22)
H_12 = sp.expand(H_12)

print("\n" + "="*60)
print("Hessian conditions (each / (2*Delta) <= 0, i.e. each <= 0):")
print("="*60)
print(f"\nH_11 = {H_11}")
print(f"\nH_22 = {H_22}")
print(f"\nH_12 = {H_12}")

# Now try to express R_crit in terms of H_11, H_22, and first-derivative terms
# Let's see if R_crit_num can be written as a combination of H_11, H_22 times
# positive quantities plus manifestly non-positive terms.

# Try solving for second derivatives from Hessian conditions:
# E_11 = (L2*L1_11 + L1*L2_11 + 2*L1_1*L2_1 - H_11 - 2*E1_crit^2) / (2*E)
# etc.
# Actually, let's substitute H_11 as a symbol and see if R_crit factors nicely.

H11, H22, H12_sym = symbols('H_11 H_22 H_12')

# Solve E_11 from H_11:
# H_11 = L2*L1_11 + L1*L2_11 + 2*L1_1*L2_1 - 2*E*E_11 - (L2*L1_1+L1*L2_1)^2/(2*E^2)
# => E_11 = (L2*L1_11 + L1*L2_11 + 2*L1_1*L2_1 - (L2*L1_1+L1*L2_1)^2/(2*E^2) - H_11) / (2*E)
E_11_from_H = (L2*L1_11 + L1*L2_11 + 2*L1_1*L2_1 - (L2*L1_1+L1*L2_1)**2/(2*E**2) - H11) / (2*E)
E_22_from_H = (L2*L1_22 + L1*L2_22 + 2*L1_2*L2_2 - (L2*L1_2+L1*L2_2)**2/(2*E**2) - H22) / (2*E)
E_12_from_H = (L2*L1_12 + L1*L2_12 + L1_1*L2_2 + L1_2*L2_1 - (L2*L1_1+L1*L2_1)*(L2*L1_2+L1*L2_2)/(2*E**2) - H12_sym) / (2*E)

# Substitute everything into R_crit
R_via_H = R_crit.subs([
    (E_11, E_11_from_H),
    (E_22, E_22_from_H),
    (E_12, E_12_from_H)
])
R_via_H = sp.cancel(R_via_H)

R_via_H_num, R_via_H_den = sp.fraction(R_via_H)
R_via_H_num = sp.expand(R_via_H_num)
R_via_H_den = sp.factor(R_via_H_den)

print("\n" + "="*60)
print("R_{0202} expressed via Hessian components H_kk:")
print("="*60)
print(f"Denominator = {R_via_H_den}")

# Collect by H_11 and H_22
R_via_H_collected = collect(R_via_H_num, [H11, H22, H12_sym])
print(f"\nCollected by H_11, H_22, H_12:")
print(R_via_H_collected)

# Check coefficient of H_11
coeff_H11 = R_via_H_num.coeff(H11)
coeff_H22 = R_via_H_num.coeff(H22)
coeff_H12 = R_via_H_num.coeff(H12_sym)
print(f"\nCoefficient of H_11: {sp.factor(coeff_H11)}")
print(f"Coefficient of H_22: {sp.factor(coeff_H22)}")
print(f"Coefficient of H_12: {sp.factor(coeff_H12)}")

# Remainder (terms independent of H_11, H_22, H_12)
remainder = R_via_H_num.subs([(H11, 0), (H22, 0), (H12_sym, 0)])
remainder = sp.expand(remainder)
rterms = sp.Add.make_args(remainder)
print(f"\nRemainder (H-independent) has {len(rterms)} terms")
print(f"Remainder = {remainder}")

# Try to factor the remainder
remainder_factored = sp.factor(remainder)
print(f"\nRemainder factored = {remainder_factored}")

# ── Key diagnostic: set eta=0 to verify diagonal case ──
print("\n" + "="*60)
print("VERIFICATION: eta=0 diagonal case")
print("="*60)
R_diag = R_0101.subs(E, 0)  # This will have issues since we divided by E
# Better to redo from R_0101 directly
R_0101_eta0 = R_0101.subs(E, 0)
R_0101_eta0 = sp.cancel(R_0101_eta0)
print(f"R_{{0202}} with eta=0: {R_0101_eta0}")

# In the diagonal case: K_mix = R_{0202}/(lam1*lam2)
# = -(1/2)[d_2^2 log(lam1) / lam2 + d_1^2 log(lam2) / lam1] + first-derivative terms
# At max of log(lam1*lam2): d_k(lam1)/lam1 + d_k(lam2)/lam2 = 0 for k=1,2
# and d_k^2(lam1)/lam1 + d_k^2(lam2)/lam2 - (d_k lam1)^2/lam1^2 - (d_k lam2)^2/lam2^2 <= 0

# ── Output LaTeX for the key formulas ──
print("\n" + "="*60)
print("LaTeX output")
print("="*60)

print("\n% R_{0202} numerator (full, at critical point of Psi):")
print(f"% {latex(R_crit_num_expanded)}")

print("\n% R_{0202} via Hessian components:")
print(f"% Coeff of H_11: {latex(sp.factor(coeff_H11))}")
print(f"% Coeff of H_22: {latex(sp.factor(coeff_H22))}")
print(f"% Coeff of H_12: {latex(sp.factor(coeff_H12))}")
print(f"% Remainder: {latex(remainder_factored)}")
