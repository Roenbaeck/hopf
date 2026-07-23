#!/usr/bin/env python3
"""
Check whether Lambda(s,g) can be made large enough to force K_min <= 0.

Lambda = int |grad s|^2 / int (Delta_g s)^2

From Bochner: K_min <= 3/(8*Lambda)
So K_min <= 0 requires Lambda -> infinity, i.e., Delta_g s -> 0.

But seam identity: gamma * Delta_g s ≈ tr_g(g - h).
For a non-trivial seam, tr_g(g-h) != 0, so Delta_g s != 0.

Can we make Lambda large by choosing s cleverly?
On a round sphere, eigenfunctions satisfy:
  Delta Y_l = -l(l+1) Y_l
  int |grad Y_l|^2 = l(l+1) int Y_l^2
  int (Delta Y_l)^2 = l^2(l+1)^2 int Y_l^2
So Lambda_l = int |grad Y_l|^2 / int (Delta Y_l)^2 = 1/(l(l+1))

For the product S^2 x S^2 with background metric,
Delta_h = Delta_1 + Delta_2. For s = Y_l1 * Y_l2:
  Delta_h s = -(l1(l1+1) + l2(l2+1)) s
  |grad s|^2 involves products of gradients

Lambda_h(s) = int |grad_h s|^2 / int (Delta_h s)^2
For s = Y_l1(unit1)*Y_l2(unit2):
 = [l1(l1+1) + l2(l2+1)] / [l1(l1+1) + l2(l2+1)]^2
 = 1 / [l1(l1+1) + l2(l2+1)]

So Lambda_h = 1/(l1(l1+1)+l2(l2+1)), which DECREASES for higher harmonics.
For l1=l2=1: Lambda = 1/4 = 0.25
For l1=l2=2: Lambda = 1/12 ≈ 0.083

So K_min_bound = 3/(8*Lambda) = 3(l1(l1+1)+l2(l2+1))/8
For l1=l2=1: K_bound = 3*4/8 = 1.5
For l1=l2=2: K_bound = 3*12/8 = 4.5

This INCREASES for higher harmonics — the bound gets WORSE (less restrictive).

CONCLUSION: The Bochner approach gives K_min <= O(l^2), which is useless
for proving K_min <= 0.

Let me verify this analytically and check what happens for the actual seam metric.
"""

import numpy as np

print("=" * 70)
print("Analytical check: Lambda for eigenfunctions on S^2 x S^2")
print("=" * 70)

print("\nFor s = Y_l1 * Y_l2 on (S^2 x S^2, h_round):")
print(f"{'l1':>3} {'l2':>3} {'Lambda_h':>12} {'K_bound=3/(8L)':>16}")
print("-" * 40)

for l1 in range(1, 6):
    for l2 in range(1, l1+1):
        ev = l1*(l1+1) + l2*(l2+1)
        Lambda = 1.0 / ev
        K_bound = 3.0 / (8 * Lambda)
        print(f"{l1:>3} {l2:>3} {Lambda:>12.4f} {K_bound:>16.4f}")

print()
print("As l increases, Lambda -> 0, so K_bound -> infinity.")
print("The Bochner approach CANNOT prove K_min <= 0.")
print()
print("The reason: the Bochner identity int Ric(grad s, grad s) = int [(Delta s)^2 - |Hess s|^2]")
print("relates Ricci curvature to Laplacian/Hessian norms. For K_min <= 0, we need")
print("Ric(grad s, grad s) to be negative somewhere. But Ric_h = h > 0 on S^2 x S^2,")
print("and the perturbation from the seam is O(gamma), so Ric_g ≈ h + O(gamma) > 0")
print("for small gamma. The Bochner inequality is satisfied with room to spare.")
print()
print("STRUCTURAL CONCLUSION:")
print("The Bochner approach (FEEDBACK Steps 4-5) cannot close the proof.")
print("It gives K_min <= C for some C > 0, not K_min <= 0.")
print("The approach would need an additional mechanism to sharpen the bound to 0.")
