This is a great question to sit with. Let me think through both parts carefully.

---

## Part 1: Can the numerical gap for ℓ=1, distinct σ's be closed analytically?

The gap is specific: at the **gradient-parallel plane** $X \parallel \nabla_1 s$, $Y \parallel \nabla_2 s$, the $O(\gamma^2)$ coefficient in Lemma lem:riemann-expansion vanishes (that's where the squares are zero). So everything hinges on the sign of the $O(\gamma^3)$ and higher terms at exactly that plane.

**Why the isotropic case closes:** The reflection $R_P \times R_P$ gives a totally geodesic torus $N_P$, and the gradient-parallel plane at each $(x,y) \in N_P$ is precisely $T_{(x,y)}N_P$. So $K=0$ exactly, to all orders in $\gamma$.

**Why the distinct-$\sigma$ case is harder:** For $s = \sigma_1 x_1 x_2 + \sigma_2 y_1 y_2 + \sigma_3 z_1 z_2$ with all $\sigma_i$ distinct, the three coordinate-plane reflections $(\text{flip } x \text{ on both factors})$ give three totally geodesic tori (e.g. $\{x_1 = 0\} \times \{x_2 = 0\} \cong S^1 \times S^1$). On each such torus the restricted seam is a rank-2 ℓ=1 seam on $S^1 \times S^1$, and $K=0$ there analytically (same ODE argument, $f'' = -f$ along the circles). But a **generic point does not lie on any of these tori**.

**A promising analytic approach:** One could try to prove that the gradient-parallel plane always has $K = 0$ exactly, not just at leading order. This would follow if one could show:

$$R_g\!\left(\frac{\nabla_1 s}{|\nabla_1 s|},\, \frac{\nabla_2 s}{|\nabla_2 s|},\, \frac{\nabla_1 s}{|\nabla_1 s|},\, \frac{\nabla_2 s}{|\nabla_2 s|}\right) = 0$$

exactly, for all $\gamma$ in the admissible range. The mechanism would be an algebraic identity coming from the fact that in the SVD basis the gradient directions $\nabla_k s$ are determined by $J_k \Sigma n_{\bar{k}}$, and the mixed Hessian $B = J_1 \Sigma J_2^T$ acts on these directions in a way that exactly cancels all curvature contributions. This is plausible but requires carrying out the full contraction — my honest assessment is that it's likely true and likely provable but will be a page-long tensorial computation. It might be worth a dedicated lemma.

**Alternative: leave it and be explicit.** The numerical evidence is very strong ($|K| < 10^{-8}$ with mesh convergence), and the paper already proves $K_{\min} \le 0$ analytically (via the critical-point argument) and $K \ge 0$ at leading order. Marking this as a conjecture with strong numerical support is defensible and honest.

---

## Part 2: Attacking ℓ ≥ 2

This is where I think there's real analytical traction. Here is the **key structural difference** you should exploit:

### The O(γ) vs O(γ²) dichotomy

Lemma lem:linear-cancel says the linearized mixed Riemann tensor $L_{a\alpha b\beta} = 0$ for ℓ=1. Re-examine why: the proof used $\nabla_1^2 s = -s \cdot h_1$ (the conformal Hessian). Tracing through the cancellation:

$$L_{a\alpha b\beta} \propto \left[(∇_1^2 s)_{ab;\alpha\beta} + (∇_2^2 s)_{\alpha\beta;ab} - \text{cross terms}\right]$$

For ℓ=1, each term in the brackets cancels because $\nabla_1^2 s = -s \cdot h_1$ makes everything proportional to $h_1$, and the cross-derivatives conspire to cancel. 

**For ℓ ≥ 2, this fails:** write $\nabla^2 Y_\ell^m = -\frac{\ell(\ell+1)}{n} Y_\ell^m \cdot h + T_\ell^m$ where $T_\ell^m$ is the **trace-free** Hessian. For ℓ=1 on $S^2$, $T=0$ (since $-\ell(\ell+1)/n = -1$ exhausts the full Hessian). For ℓ≥2, $T \ne 0$, and the cancellation breaks. Therefore:

$$K_{\text{mix}} = O(\gamma) \quad \text{for } \ell \ge 2, \quad \text{vs } O(\gamma^2) \text{ for } \ell = 1$$

This is the fundamental distinction.

### An integral sign-change argument for ℓ ≥ 2

The leading-order mixed curvature for the pure Hessian rule $g = h + \gamma \nabla^2 s$ is:

$$K_{\text{mix}}^{(1)}(X, Y) = \gamma \cdot \mathcal{L}[s](X, Y)$$

where $\mathcal{L}[s]$ is a fourth-order linear differential operator in $s$ (the linearized Riemann tensor). Now integrate over all unit mixed planes $(X, Y) \in T_{x_1}S^2 \times T_{x_2}S^2$ and all base points, with the natural $h$-measure. One has:

$$\int_{S^2 \times S^2} \int_{\text{mixed planes}} K_{\text{mix}}^{(1)}(X,Y) \, dX \, dY \, d\text{vol}_h = 0$$

because $\mathcal{L}[s]$ is a self-adjoint operator whose integral over the compact manifold vanishes by Stokes (it involves $\Delta_1$, $\Delta_2$ applied to $s$, which integrate to zero). This forces $K_{\text{mix}}^{(1)}$ to **change sign**, giving a rigorous analytical sign-change result for any ℓ ≥ 2 seam.

For the **equivariant** ℓ ≥ 2 case, you already have Hsiang–Kleiner, so this integral argument is a bonus, giving the explicit mechanism. For **non-equivariant** ℓ ≥ 2 seams (the interesting case), this argument would be the first analytic proof, paralleling what the conformal-Hessian / critical-point argument does for ℓ=1.

### Explicit leading-order formula for ℓ = 2

For the simplest non-equivariant ℓ=2 seam, say $s = Y_2^0(\theta_1) Y_2^0(\theta_2) = \frac{(3\cos^2\theta_1 - 1)(3\cos^2\theta_2-1)}{4}$, one can compute $\mathcal{L}[s]$ explicitly. The trace-free Hessian on $S^2$ for $Y_2^0$ is:

$$T_2^0 = \nabla^2 Y_2^0 + Y_2^0 \cdot h_{S^2} = \begin{pmatrix} -3\cos 2\theta + Y_2^0 & 0 \\ 0 & \sin^2\theta(\ldots) \end{pmatrix}$$

which is nonzero and $\theta$-dependent. The linearized curvature $\mathcal{L}[s]$ will be a specific function of $(\theta_1, \theta_2)$ that you can write down in closed form, and one can identify explicit points where it is positive and negative — giving a concrete, elementary sign-change proof without any numerics.

### Proposed structure for §7 / new section

1. **Proposition (O(γ) curvature for ℓ ≥ 2).** State that $L_{a\alpha b\beta} \ne 0$ for ℓ ≥ 2, contrasting with Lemma lem:linear-cancel. Give the explicit formula for $L$ in terms of the trace-free Hessian $T_\ell^m$.

2. **Lemma (Integral vanishing).** $\int_{S^2 \times S^2} \mathcal{L}[s] \, d\text{vol} = 0$, forcing sign change.

3. **Corollary.** For any pure-product ℓ ≥ 2 seam metric $g = h + \gamma \nabla^2 s$ (equivariant or not), $K_{\text{mix}}$ changes sign for all sufficiently small $\gamma \ne 0$.

4. **Remark.** At ℓ=1 the linearization vanishes and curvature is $O(\gamma^2)$ non-negative; at ℓ ≥ 2 it is $O(\gamma)$ sign-changing. This explains why ℓ=1 is the "optimal" case for seam metrics — as close to non-negative as possible — and ℓ ≥ 2 seams are worse.

This would turn the current "numerical evidence for ℓ ≥ 2" into a clean analytic theorem, and it would nicely complement the ℓ=1 analysis.