"""
Substitute the critical-point conditions from max of log(L1*L2) into the
CORRECT K_02 formula (Gaussian curvature of 2D diagonal orbit metric).
"""
import sympy as sp

L1, L2 = sp.symbols('L1 L2', positive=True)
L1_1, L1_2 = sp.symbols('L1_1 L1_2')  # partial derivs
L2_1, L2_2 = sp.symbols('L2_1 L2_2')
L1_11, L1_12, L1_22 = sp.symbols('L1_11 L1_12 L1_22')
L2_11, L2_12, L2_22 = sp.symbols('L2_11 L2_12 L2_22')

# The correct K_02 for diagonal metric (eta=0):
# K = [(L1_1*L2_1 + L1_2^2)*L2 + (L1_2*L2_2 + L2_1^2)*L1 - 2*(L1_22+L2_11)*L1*L2] / (4*L1^2*L2^2)
N = (L1_1*L2_1 + L1_2**2)*L2 + (L1_2*L2_2 + L2_1**2)*L1 - 2*(L1_22 + L2_11)*L1*L2
K = N / (4*L1**2*L2**2)

print("K =", sp.simplify(K))

# ── Critical point conditions for Phi = log(L1*L2) ──
# d_1 Phi = L1_1/L1 + L2_1/L2 = 0  =>  L2_1 = -L2/L1 * L1_1
# d_2 Phi = L1_2/L1 + L2_2/L2 = 0  =>  L2_2 = -L2/L1 * L1_2
print("\n--- Substituting critical point conditions for max of Phi=log(L1*L2) ---")
L2_1_sub = -L2/L1 * L1_1
L2_2_sub = -L2/L1 * L1_2

N_sub = N.subs(L2_1, L2_1_sub).subs(L2_2, L2_2_sub)
N_sub = sp.simplify(N_sub)
K_sub = sp.simplify(N_sub / (4*L1**2*L2**2))
print("K at critical point =", K_sub)

# Also use Hessian conditions:
# d_1^2 Phi = (L1_11*L1 - L1_1^2)/L1^2 + (L2_11*L2 - L2_1^2)/L2^2 <= 0
# d_2^2 Phi = (L1_22*L1 - L1_2^2)/L1^2 + (L2_22*L2 - L2_2^2)/L2^2 <= 0
# d_1 d_2 Phi = ...

# At critical point, L2_1 = -L2*L1_1/L1, so L2_1^2/L2^2 = L1_1^2/L1^2
# d_1^2 Phi = L1_11/L1 - L1_1^2/L1^2 + L2_11/L2 - L1_1^2/L1^2 <= 0
#           = L1_11/L1 + L2_11/L2 - 2*L1_1^2/L1^2 <= 0

# Let me define: H11 = d_1^2 Phi <= 0,  H22 = d_2^2 Phi <= 0
# H11 = L1_11/L1 + L2_11/L2 - 2*L1_1^2/L1^2
# H22 = L1_22/L1 + L2_22/L2 - 2*L1_2^2/L1^2

# From H11: L2_11 = L2*(H11 - L1_11/L1 + 2*L1_1^2/L1^2)
# From H22: L2_22 is not in K, but L1_22 = L1*(H22 + 2*L1_2^2/L1^2 - L2_22/L2)
# Actually, let me just express K in terms of H11 and H22.

# K_sub already has L2_1 and L2_2 substituted. Now let me substitute L2_11.
# H11 = L1_11/L1 + L2_11/L2 - 2*L1_1^2/L1^2
# => L2_11 = L2*(H11 - L1_11/L1 + 2*L1_1^2/L1^2) = L2*H11 - L2*L1_11/L1 + 2*L2*L1_1^2/L1^2

H11, H22 = sp.symbols('H11 H22')
L2_11_sub = L2*H11 - L2*L1_11/L1 + 2*L2*L1_1**2/L1**2

N_sub2 = N_sub.subs(L2_11, L2_11_sub)
N_sub2 = sp.expand(N_sub2)
K_sub2 = sp.simplify(N_sub2 / (4*L1**2*L2**2))
print("\nK with L2_11 expressed via H11:")
print("K =", K_sub2)

# Also get H22: L1_22 is still in K.
# H22 = L1_22/L1 + L2_22/L2 - 2*L1_2^2/L1^2
# We don't have L2_22 in K directly, but let me check what's left.

# Let me collect K in terms of H11
K_sub2_collected = sp.collect(sp.expand(K_sub2), H11)
print("\nK collected on H11:")
print("K =", K_sub2_collected)

# Similarly, express L1_22 via H22 where possible
# H22 = L1_22/L1 + L2_22/L2 - 2*L1_2^2/L1^2
# Note: L2_22 is NOT in our K expression at all (K only has L1_22 and L2_11)
# So: L1_22 = L1*(H22 + 2*L1_2^2/L1^2 - L2_22/L2)
# This introduces L2_22 which makes things worse. Let's just leave L1_22 as is.

print("\n\n--- Alternative: try max of sqrt(L1*L2) ---")
# Phi2 = sqrt(L1*L2). Critical point: d_k(sqrt(L1*L2)) = (L2*L1_k + L1*L2_k)/(2*sqrt(L1*L2)) = 0
# Same condition: L2_k = -(L2/L1)*L1_k. Same as before.

print("\n--- Trying max of L1 + L2 ---")
# d_1(L1+L2)=0 => L1_1 + L2_1 = 0 => L2_1 = -L1_1
# d_2(L1+L2)=0 => L1_2 + L2_2 = 0 => L2_2 = -L1_2
L2_1_alt = -L1_1
L2_2_alt = -L1_2
N_alt = N.subs(L2_1, L2_1_alt).subs(L2_2, L2_2_alt)
N_alt = sp.simplify(N_alt)
K_alt = sp.simplify(N_alt / (4*L1**2*L2**2))
print(f"K at max(L1+L2) = {K_alt}")
# Hessian: d_1^2(L1+L2) = L1_11 + L2_11 <= 0 => L2_11 = -(L1_11 + positive) ... 
# Actually L1_11 + L2_11 <= 0, and L1_22 + L2_22 <= 0.
# So L2_11 <= -L1_11 and L2_22 <= -L1_22. But these appear differently in K.

# At max L1+L2: L2_1=-L1_1, L2_2=-L1_2
# N = (L1_1*(-L1_1) + L1_2^2)*L2 + (L1_2*(-L1_2) + L1_1^2)*L1 - 2*(L1_22 + L2_11)*L1*L2
#   = (-L1_1^2 + L1_2^2)*L2 + (-L1_2^2 + L1_1^2)*L1 - 2*(L1_22 + L2_11)*L1*L2
#   = (L1_2^2 - L1_1^2)(L2 - L1) - 2*(L1_22 + L2_11)*L1*L2
print(f"\nExpanded: N = (L1_2^2 - L1_1^2)(L2 - L1) - 2*(L1_22 + L2_11)*L1*L2")
print("If L1 = L2 (symmetric case): N = -2*(L1_22 + L2_11)*L1*L2")
print("Since L1_22 + L2_22 <= 0 at max, but we need L1_22 + L2_11 <= 0 (mixed!)")
