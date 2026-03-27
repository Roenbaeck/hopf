# Seam Obstructions to Positive Sectional Curvature on Product Manifolds

## Project Goal

Prove that no $O(2) \times O(2)$-equivariant order-$\le 2$ seam metric on $S^2 \times S^2$ has strictly positive sectional curvature. This is a partial result toward the Hopf conjecture.

## Files

- `hopf_seam.tex` — Main paper (Springer sn-jnl class)
- `compute_mixed_curvature.py` — Symbolic computation of $R_{0202}$ (needs Python + SymPy)
- `sn-jnl.cls` — Journal class file
- `sn-mathphys-num.bst` — Bibliography style
- `reference_material/jet-framework.tex` — Real seam framework paper (reference)
- `reference_material/complex-jet-framework.tex` — Complex seam framework paper (reference)
- `reference_material/seam_einstein.tex` — Real / complex seam framework applied to the Einstein Tensor (reference)
- `reference_material/quaternion_seam_einstein.tex` — Quaternion seam framwork applied to Maxwell's equations (reference)

## What Is Done

### 1. Classification (§2, Theorem 2.3) — COMPLETE
Eight-generator classification of $O(n_1) \times O(n_2)$-equivariant metric rules on product backgrounds: two factor conformal, two factor gradient, one mixed gradient, two factor Hessians, one mixed Hessian. Proof via invariant theory is standard.

### 2. Conformal product obstruction (§4, Proposition 4.3) — COMPLETE
**Theorem:** No conformal product metric $e^{2u} h_1 + e^{2v} h_2$ on $S^2 \times S^2$ has all mixed sectional curvatures positive.

**Proof:** Maximize $v(\cdot, p_2)$ over factor 1 → Hessian $\nabla_1^2 v \le 0$ at the maximum. Maximize $u(p_1^*, \cdot)$ over factor 2 → Hessian $\nabla_2^2 u \le 0$. At the resulting point, all terms in the mixed curvature formula are $\le 0$.

### 3. Diagonal obstruction (§4, Proposition 4.4) — COMPLETE
**Theorem:** No diagonal product seam metric (all eight generators but cross-coupling $\eta = 0$) has positive mixed curvature.

**Proof:** Define $\Phi = \log(\lambda_1 \lambda_2)$ where $\lambda_k$ are the metric eigenvalues. Maximize $\Phi$ over $S^2 \times S^2$. At the maximum, $\partial_1^2 \log\lambda_2 \le 0$ and $\partial_2^2 \log\lambda_1 \le 0$ (from the Hessian bound on $\Phi$), which are exactly the terms in $K_{\mathrm{mix}} = -\frac{1}{2}[\partial_2^2 \log\lambda_1 + \partial_1^2 \log\lambda_2]$.

### 4. Explicit examples (§5) — COMPLETE
- Height-sum seam $s = \cos\theta_1 + \cos\theta_2$: mixed curvature positive at poles, negative at equators.
- Height-product seam $s = \cos\theta_1 \cos\theta_2$: $\eta \neq 0$, auxiliary function $\Psi$ confirms obstruction numerically.

### 5. Kähler perspective (§6, Proposition 6.2) — SKETCH ONLY
Maximum-point argument on $\psi_{2\bar{2}}$ using $\partial\bar\partial$-lemma. Needs: bound the correction terms from $g^{i\bar{j}}$ in the curvature tensor.

## What Has Gaps

### 6. Full theorem with cross-coupling (§4, Theorem 4.5) — GAP
**The hard part.** When $\eta = \beta_{12} s_1 s_2 + \gamma_{12} s_{12} \neq 0$, the mixed curvature has additional terms $R(\eta)$ from the off-diagonal metric entry. The proof strategy:

- Define $\Psi = \frac{1}{2} \log(\lambda_1 \lambda_2 - \eta^2)$.
- At the maximum of $\Psi$, show $K_g(\sigma) \le 0$.
- The positive part of $R(\eta)$ is $\frac{3}{4}(\partial_k \eta)^2 / (\lambda_1\lambda_2 - \eta^2)$.
- Claim: this vanishes or is controlled at the critical point of $\Psi$.

**What's needed to close the gap:**

1. **Compute $R_{0202}$ explicitly** for the full 4×4 metric with off-diagonal $\eta$. The script `compute_mixed_curvature.py` does this but needs Python/SymPy (not installed in the original workspace). Run this first.

2. **Expand $R(\eta)$ at the critical point of $\Psi$**: At the maximum, $\partial_k \Psi = 0$ gives:
   $$\frac{\lambda_2 \partial_k \lambda_1 + \lambda_1 \partial_k \lambda_2 - 2\eta \partial_k \eta}{2(\lambda_1 \lambda_2 - \eta^2)} = 0$$
   This relates $\partial_k \eta$ to $\partial_k \lambda_j$ and may eliminate the dangerous positive terms.

3. **Sign analysis**: After substituting the critical-point conditions into the full $R_{0202}$, check whether all remaining terms are $\le 0$. If so, theorem is proved. If not, need a different auxiliary function or a different argument.

**Alternative approaches if the $\Psi$-maximum argument fails:**
- Integral method: show $\int K_{\mathrm{mix}} \, dV_g \le 0$ using Stokes, then $K_{\mathrm{mix}}$ can't be everywhere positive.
- Spectral method: expand $s$ in product spherical harmonics, bound the cross-coupling contribution mode-by-mode.
- Topological: use Chern–Gauss–Bonnet on the 2D $(\theta_1, \theta_2)$-surface directly.

## Scope and Caveats

Even with all eight generators covered, this proves the obstruction only for **seam metrics** of order $\le 2$ with $O(2) \times O(2)$-equivariance. The full Hopf conjecture requires universality (that all metrics are seam metrics), which is open. The paper's abstract and Remark 1.3 correctly state this limitation.

## Key References

- Hsiang–Kleiner (1989): $b_2 \le 1$ under positive curvature + $S^1$-action
- Bourguignon–Karcher (1978): auxiliary-function method for curvature sign detection
- Rönnbäck (2025): jet-framework, complex-jet-framework (seam classification papers)

## Suggested Next Steps

1. Install Python + SymPy and run `compute_mixed_curvature.py`
2. Inspect the full $R_{0202}$ formula with $\eta$
3. Substitute the critical-point conditions from $\partial_k \Psi = 0$
4. Determine sign — this resolves whether the $\Psi$ argument works
5. If it works → write up the proof in §4 and close the gap
6. If it doesn't → try the integral or spectral alternatives
7. Clean up the Kähler section (§6)
8. Submission
