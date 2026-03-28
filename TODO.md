# Seam Obstructions to Positive Sectional Curvature on Product Manifolds

## Project Goal

Prove that no $O(2) \times O(2)$-equivariant order-$\le 2$ seam metric on $S^2 \times S^2$ has strictly positive sectional curvature. This is a partial result toward the Hopf conjecture.

## Files

- `hopf_seam.tex` — Main paper (Springer sn-jnl class)
- `compute_mixed_curvature.py` — Symbolic computation of $R_{0202}$ (Python + SymPy)
- `reference_material/jet-framework.txt` — Real seam framework paper (reference)
- `reference_material/complex-jet-framework.txt` — Complex seam framework paper (reference)
- `reference_material/seam_einstein.txt` — Real / complex seam framework applied to the Einstein Tensor (reference)
- `reference_material/quaternion_seam_einstein.txt` — Quaternion seam framework applied to Maxwell's equations (reference)

### Computation scripts (in `.venv` with SymPy 1.14.0, numpy 2.0.2)
- `sympy_diagonal_formula.py` — Derives K₀₂ from 4×4 diagonal metric; confirms paper's old formula was WRONG
- `sympy_compare_2d_4d.py` — Symbolic proof: K₀₂ of 4D metric = Gaussian curvature of 2D orbit metric
- `critical_point_correct.py` — Simplifies K at max of Φ=log(λ₁λ₂): gives -(λ₁,₂₂ + λ₂,₁₁)/(2λ₁λ₂)
- `sign_test_K02.py` — 962 random tests: K always changes sign (0/962 all-positive)
- `test_sequential_minmin.py` — Tests min-min proof approaches (NOT 100%: 219/244 sequential, 226/244 global)
- `test_minmin_fine.py` — Fine grid (N=400) confirming 7/729 failures with K ∈ [0.06, 0.13]
- `test_boundary_integral.py` — Gauss-Bonnet boundary integral test (no definite sign)

## Critical Findings

### Formula error discovered and corrected
The paper's original Proposition 4.4 used:
$$K = -\tfrac{1}{2}[\partial_2^2 \log\lambda_1 + \partial_1^2 \log\lambda_2]$$
This is **wrong**. It is not the Gaussian curvature of a diagonal 2D metric. The correct formula (Brioschi/Gaussian curvature of the orbit metric) is:
$$K = \frac{(\lambda_{1,1}\lambda_{2,1} + \lambda_{1,2}^2)\lambda_2 + (\lambda_{1,2}\lambda_{2,2} + \lambda_{2,1}^2)\lambda_1 - 2(\lambda_{1,22} + \lambda_{2,11})\lambda_1\lambda_2}{4\lambda_1^2\lambda_2^2}$$
Confirmed symbolically (`sympy_diagonal_formula.py`) and numerically.

### Key identity: K₀₂ = Gaussian curvature of orbit metric
The mixed sectional curvature of the 4D metric equals the Gaussian curvature of the 2D orbit metric $g_{\mathrm{orb}} = [[\lambda_1, \eta], [\eta, \lambda_2]]$. Confirmed symbolically (`sympy_compare_2d_4d.py`: `K_2D - K_4D = 0`).

### Direct proof approaches all fail
- **Ψ-argument** (maximize $\Psi = \frac{1}{2}\log(\lambda_1\lambda_2 - \eta^2)$): 441/593 violations
- **Φ-argument** (maximize $\Phi = \log(\lambda_1\lambda_2)$): At max, $K = -(\lambda_{1,22} + \lambda_{2,11})/(2\lambda_1\lambda_2)$. Hessian constrains *same-factor* combinations ($\lambda_{1,11}/\lambda_1 + \lambda_{2,11}/\lambda_2 \le 0$) but K involves *cross* combination ($\lambda_{1,22} + \lambda_{2,11}$) — fundamental mismatch.
- **Sequential min-min**: 219/244 (not 100%)
- **Global min of Φ**: 226/244 (not 100%)
- **Boundary integral (Gauss-Bonnet)**: No definite sign (min=-0.19, max=0.017)
- **Numerical sign test**: 0/962 all-positive confirms K always changes sign, but doesn't yield a proof

### Resolution: Hsiang–Kleiner
For $O(2) \times O(2)$-equivariant seam $s = s(\theta_1, \theta_2)$, the product seam metric is invariant under $U(1) \times U(1)$ rotations in $(\phi_1, \phi_2)$, hence admits an isometric $S^1$-action. By Hsiang–Kleiner (1989), compact oriented 4-manifold with positive sectional curvature + $S^1$-action → $b_2 \le 1$. Since $b_2(S^2 \times S^2) = 2$, positive curvature is impossible.

The paper has been rewritten to use this argument for the equivariant case and to honestly state the relationship with Hsiang–Kleiner.

## Current State of the Paper

### COMPLETE (proofs correct)
1. **Classification (§2, Theorem 2.3):** Eight-generator classification of equivariant metric rules. Standard.
2. **Conformal product obstruction (§4, Prop 4.3):** Direct max-principle proof. Self-contained, does not need Hsiang–Kleiner.
3. **Diagonal obstruction (§4, Prop 4.4):** Now proved via Hsiang–Kleiner (U(1)×U(1) symmetry). Remark on orbit-metric perspective with correct formula and analysis of why max-principle fails.
4. **Full theorem (§4, Thm 4.5):** Proved via Hsiang–Kleiner for equivariant seams.
5. **Examples (§5):** Height-sum (conformal product), height-product (cross-coupling). Updated to remove old Ψ-argument references.
6. **Discussion (§7):** Honestly states result is a special case of Hsiang–Kleiner; identifies three values of seam perspective; frames non-equivariant extension as main open problem.

### NEEDS WORK
7. **Kähler section (§6, Prop 6.2):** Proof sketch only. Max-point argument on ψ₂₂̄ + ∂∂̄-lemma. Needs: bound correction terms from $g^{i\bar{j}}$ in curvature tensor. This is independent of the real proof and could be a genuine new result (Kähler metrics don't need equivariance).
8. **LaTeX compilation:** Paper has been extensively rewritten; needs compilation test.

## Scope and Caveats

The paper proves the obstruction for **$O(2) \times O(2)$-equivariant** seam metrics via Hsiang–Kleiner. The conformal product case has a self-contained elementary proof. The value of the paper is:
- Constructive curvature analysis identifying the Gaussian curvature of the orbit metric as the obstruction mechanism
- Correct explicit formula for the mixed sectional curvature (correcting the literature)
- Identification of the non-equivariant case as the main open problem

The full Hopf conjecture requires: (a) extending to non-equivariant seams (where Hsiang–Kleiner does not apply), and (b) proving universality (that all metrics are seam metrics).

## Key References

- Hsiang–Kleiner (1989): $b_2 \le 1$ under positive curvature + $S^1$-action
- Rönnbäck (2025): jet-framework, complex-jet-framework (seam classification papers)

## Non-Equivariant Investigation (FEEDBACK.md approach)

### Computation scripts
- `test_feedback_steps.py` — Verifies Steps 1–3 on flat $T^4$: first-order cancellation, $Q$ sign-change, correlation with $K/\gamma^2$
- `test_Q_integral.py` — Tests $\int Q\,dV = 0$ identity on flat $T^4$ (30/30 seams, machine precision)
- `test_Q_S2xS2.py` — Full $R_{0202}$ computation on round $S^2 \times S^2$ with $l=1$ product harmonics
- `test_Q_S2xS2_l2.py` — Same with $l=2$ spherical harmonics and random mixed seams
- `test_gamma_scaling.py` — Scaling of $\int K\,dV$ vs $\gamma$ on $S^2 \times S^2$

### Confirmed results

| Finding | Evidence |
|---------|----------|
| **First-order cancellation** ($K = O(\gamma^2)$) | $K/\gamma^2 \to \text{const}$ as $\gamma \to 0$, on both $T^4$ and $S^2 \times S^2$ |
| **$K$ always changes sign on flat $T^4$** | 200/200 random Hessian seams |
| **$\int Q\,dV = 0$ on flat $T^4$** | 30/30 random seams to machine precision; Fourier proof: single-mode $Q \equiv 0$, cross terms vanish by orthogonality |
| **$K$ always changes sign on round $S^2 \times S^2$** | 14/15 pure $l{=}2$, 15/15 random mixed $l{=}1{+}l{=}2$; pure $l{=}1$ has $K_{\mathrm{mix}} \ge 0$ everywhere |
| **$l{=}1$ non-negative mixed curvature**: pure $l{=}1$ seams give $K_{\mathrm{mix}} \ge 0$ | All tested seams (z₁z₂, x₁x₂, n₁·n₂, random), ultra-precise local FD + SymPy exact (K(θ₁,θ₂)=0 for z₁z₂). **Sign convention corrected**: original code computed $-K$, not $K$. |

### Key negative result
**The flat-torus integral identity $\int Q\,dV = 0$ does NOT hold on $S^2 \times S^2$.**
- $\int K\,dV / \gamma^2 \to \text{const} \neq 0$ as $\gamma \to 0$, confirming $\int Q\,dV \neq 0$
- Cause: background curvature of $S^2$ introduces Ricci commutator terms $[\nabla_a, \nabla_b] = R(e_a, e_b)$ within each factor; these don't integrate to zero

### Sign convention correction
**CRITICAL**: The original numerical code (`build_metric_and_curvature`) computes $R_{XYYX}/\text{denom} = -K$, not $K$. Verified by checking within-factor curvature: code gives $-0.94 \approx -1$ for the round $S^2$ (should be $+1$). All previous claims of "$K \le 0$" for $l{=}1$ were actually $K \ge 0$ (non-negative mixed curvature).

### Non-equivariant l=1 seam structure
| Finding | Evidence |
|---------|---------|
| **Conformal Hessian**: $\nabla^2 Y_1^m = -Y_1^m\,g_{S^2}$ for $l{=}1$ harmonics | Verified numerically: $H_{ij}/(-s) = 1.000$ for all within-factor components |
| **SVD reduction**: any $l{=}1$ seam is isometric to $s = \sigma_1 x_1x_2 + \sigma_2 y_1y_2 + \sigma_3 z_1z_2$ | Via $\mathrm{SO}(3) \times \mathrm{SO}(3)$ rotation of each factor |
| **Coupling singular values** $(1, |s|)$: for $s = \hat{n}_1 \cdot \hat{n}_2$ (full rank), the off-diagonal block in orthonormal frames has $\sigma = (1, |s|)$ | Verified exactly at 20+ random points + special cases |
| **$K_{\min} = 0$ everywhere**: for ALL $l{=}1$ seams, $\min_\sigma K_{\mathrm{mix}}(\sigma) = 0$ at every point | Fine angular scan (1440 pts): $K_{\min} \to 4.7{\times}10^{-9}$; Nelder–Mead optimisation: $K_{\min} = 1.8{\times}10^{-9}$; SO(3) consistency verified |
| **K=0 mechanism**: the weaker singular value provides an uncoupled mixed direction | At $s=0$: rank drops, one direction fully decoupled. At $s=\pm 1$: product metric, all $K=0$. At generic $s$: the SVD weak direction gives $K=0$ |
| **$l \ge 2$ seams have $K < 0$**: traceless Hessian breaks conformal structure | 14/15 pure $l{=}2$, 15/15 random mixed $l{=}1{+}l{=}2$: mixed $K$ changes sign |

### Remaining approaches
1. ~~**Bochner approach** (FEEDBACK Steps 4–5)~~: **DOES NOT WORK.** The Bochner inequality gives $K_{\min} \leq 3/(8\Lambda)$ where $\Lambda = \int|\nabla s|^2 / \int(\Delta s)^2$. For eigenfunctions $Y_{l_1} \otimes Y_{l_2}$, $\Lambda = 1/(l_1(l_1{+}1) + l_2(l_2{+}1))$, giving $K_{\min} \leq O(l^2)$. This is an upper bound on $K_{\min}$ but can never prove $K_{\min} \leq 0$. The Ricci curvature of $(S^2 \times S^2, h)$ is positive, and the seam perturbation is $O(\gamma)$, so $\mathrm{Ric}_g \approx h > 0$ for small $\gamma$. The Bochner inequality is satisfied with room to spare; it cannot be saturated to force negativity.
2. **$l{=}1$ definiteness**: Why is $K \le 0$ for $l{=}1$ seams? The Hessian $\nabla^2 Y_1^m = -Y_1^m\,g_{S^2}$ is conformal, which may force definiteness. Could extend to general harmonics with a correction term.
3. **Abandon perturbative, use topological/global argument**.

## Open Problems

1. **Non-equivariant seam obstruction:** When $s = s(\theta_1, \phi_1, \theta_2, \phi_2)$ depends on all four coordinates, the metric need not have any continuous symmetry. Can one still show $K_{\mathrm{mix}}$ changes sign? The first-order cancellation and sign-change have been verified numerically. The Bochner approach (FEEDBACK Steps 4–5) is the most promising analytic avenue.
2. **Complete the Kähler proof (§6):** Make the proof of Prop 6.2 rigorous by bounding the $g^{i\bar{j}}$ correction terms.
3. **Universality:** Is every Riemannian metric on $S^2 \times S^2$ a seam metric? This is a fundamental open question in the seam framework.
