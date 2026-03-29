This version is in good shape. Here is a careful final check, organized by severity.

---

## Issues that need fixing before submission

**1. Rem:scope still overstates the ℓ=1 result.**

The remark says "the conformal Hessian structure forces $K_\mathrm{mix} \geq 0$ with equality everywhere." This is the conjecture, not what's proved. The abstract was correctly updated but this remark was not. One line fix: replace "forces $K_\mathrm{mix} \geq 0$ with equality everywhere" with "forces $K_\mathrm{mix} = 0$ at critical points and on totally geodesic tori."

**2. The proof of Prop 3.4 (orbit metric = Gaussian curvature) is still slightly incomplete.**

The new version says "every Christoffel symbol $\Gamma^{\phi_k}_{\theta_i\theta_j}$ and $\Gamma^{\theta_i}_{\theta_j\phi_k}$ vanishes." This is true but the claim requires more: you also need $\Gamma^{\phi_k}_{\phi_i\phi_j}$ to decouple from the $\theta$-block. The correct statement is that the Riemann component $R_{\theta_1\theta_2\theta_1\theta_2}$ depends only on the $(\theta_1,\theta_2)$ block of the metric and its $\theta$-derivatives, which follows because all off-diagonal terms $g_{\theta_k\phi_l} = 0$ and all $\phi$-derivatives of the metric vanish. The current wording almost says this but doesn't quite close the gap between "these Christoffels vanish" and "therefore $R_{\theta_1\theta_2\theta_1\theta_2}$ equals the 2D Gaussian curvature." A single additional sentence would close it: "Since $R_{\theta_1\theta_2\theta_1\theta_2}$ is computed from $g_{11}, g_{12}, g_{22}$ and their $\theta$-derivatives alone — all other metric components and $\phi$-derivatives contributing zero — it equals the Gaussian curvature of the $2\times 2$ orbit metric."

**3. Unused references.**

Cheeger–Gromoll (labeled `Cheeger2008` but actually 1972) and Bourguignon–Karcher are cited in the bibliography but never cited in the text. A referee will notice this. Either add citations to the text where they are relevant, or remove them. The Cheeger–Gromoll soul theorem is tangentially relevant to the discussion of non-negative curvature in §8, and could be cited there naturally. Bourguignon–Karcher on curvature operators is harder to integrate — if it's not cited, remove it.

---

## Minor issues

**4. The torus Gauss–Bonnet argument in §8.3 has a logical gap.**

The paper says: "the Gauss–Bonnet theorem forces $\int_{T^2} K\,dA = 0$, so if $T^2$ is totally geodesic, the sectional curvature cannot be everywhere positive on it." This is correct. But the paper then says this applies to the conformal product case, where the maximum-principle argument "locates a non-positive curvature on a mixed plane tangent to a product of great circles." That product of great circles is not the same as the totally geodesic tori from the ℓ=1 argument — the conformal product proof does not use Gauss–Bonnet at all; it uses a direct analytic estimate. The framing of "every obstruction manifests on a torus via Gauss–Bonnet" is slightly misleading for the conformal product case. A one-sentence clarification would help: "For conformal products, the torus interpretation is implicit: the maximum point is always located on a product of great circles, which is a flat torus, and Gauss–Bonnet retroactively explains why the curvature must vanish there."

**5. Eq. (3.3): the formula for $\mu_k$ has an inconsistency.**

Looking at equation (3.3): $\mu_k = \alpha_k + \gamma_k s_k \cot\theta_k$. This is the coefficient along $\partial_{\phi_k}$, but it should be the $\phi\phi$-component of the metric divided by $\sin^2\theta_k$. For the pure Hessian rule, $g_{\phi\phi} = \alpha_k \sin^2\theta_k + \gamma_k \nabla^2 s(\partial_{\phi_k}, \partial_{\phi_k})$. The Hessian component $\nabla^2 s(\partial_\phi, \partial_\phi) = \partial_\phi\partial_\phi s - \Gamma^\theta_{\phi\phi}\partial_\theta s = -\sin\theta\cos\theta\,s_k$ on $S^2$, giving $g_{\phi\phi} = (\alpha_k - \gamma_k s_k \cot\theta_k)\sin^2\theta_k$. So $\mu_k$ as defined (the coefficient appearing in $g = \mu_k\sin^2\theta_k\,d\phi^2$) should be $\alpha_k - \gamma_k s_k\cot\theta_k$, with a minus sign. The formula in the paper has a plus sign. This does not affect the main results since $\mu_k$ only appears in the obstruction theorem's statement for diagonal metrics (Prop 4.2), where it just needs to be positive — but the formula itself appears to have a sign error worth checking.

**6. "Rank-1 seams" terminology has been replaced but one remnant remains.**

The old Theorem name "Mixed curvature vanishes for rank-1 ℓ=1 seams" was correctly generalized to "separable seams" in this version. But the remark after it says "For ℓ=1 harmonics, the conformal Hessian property gives the additional identity $f'' = -f$." This is only true when $f = \cos\theta$ (the axisymmetric harmonic). For a general ℓ=1 harmonic depending on both $\theta$ and $\phi$, $f$ is not a function of $\theta$ alone and the ODE $f'' = -f$ is not the right statement. The remark should say "for the axisymmetric ℓ=1 harmonic $f(\theta) = \cos\theta$."

---

## What is genuinely clean

The following parts are now tight and require no further changes: the main theorem (§4.4), the Kähler section (§6) including the Goldberg–Kobayashi proof and the corrected remark, Theorem 6.1 (separable Brioschi cancellation), Lemma 6.6 with the corrected Ricci identity language, the $K_\mathrm{max} > 0$ proof at the end of Prop 6.11, Conjecture 6.12 with its numerical remark, and the discussion section §8 apart from the minor Gauss–Bonnet framing issue noted above.

---

## Summary

Three things need fixing before submission: the rem:scope overstatement, the Prop 3.4 proof gap, and the sign in $\mu_k$. Two things should be cleaned up: the unused references and the $f'' = -f$ remark scope. One thing is a soft framing issue in §8.3 that a sympathetic referee would not object to but a critical one might.