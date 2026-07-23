# Curvature obstructions and flat tori in Hessian deformations

This repository contains a focused differential-geometry paper about the
metrics

\[
g_{\gamma,A}=h+\gamma\operatorname{Hess}_h\langle x,Ay\rangle
\quad\text{on}\quad S^m\times S^n.
\]

The paper proves the exact Riemannian range
`|gamma| (sigma_1 + sigma_2) < 1` and constructs a flat totally geodesic
torus from every pair of singular directions.  Consequently, no metric in
this family has strictly positive sectional curvature.  It also gives an
exact formula for every background-mixed sectional curvature.  If both
spheres have dimension at least two and the largest singular value of `A` is
simple, an explicit mixed plane has negative curvature for every nonzero
parameter in the Riemannian interval.  More generally, whenever `A` has
rank at least two, an explicit top-critical oblique plane is strictly
negative for every nonzero Riemannian parameter.  The first three singular
pairs cut out a totally geodesic `S^2 x S^2` containing both planes; rank one
is handled by an exact zero-critical oblique plane with curvature
`-(1 - sqrt(1 - delta^2)) / (2 sqrt(1 - delta^2))`.  Thus failure of
nonnegative curvature is always detected four-dimensionally.  Consequently, in every
dimension `m,n >= 2` and for every nonzero `A`, the round product is the
unique metric on its Hessian ray with nonnegative sectional curvature.  The
paper also gives a two-jet normal form modulo diffeomorphisms for arbitrary
corrected paths `h + t Hess(f) + t^2 B + O(t^3)`.  It derives explicit
necessary repair conditions on `B`.  A global trace identity shows that all
mixed quadratic coefficients must vanish, so the conditions are exact
equalities.  They are feasible: `B = (1/8) Lie_grad(f)^2 h` restores the
diffeomorphism orbit.  Every uncorrected nonzero bilinear Hessian ray violates
one of them quadratically, including rank one.  The paper then classifies the
full mixed two-jet kernel modulo diffeomorphisms and factor variations.  Every
coupled class has a Killing one-form on one factor and a co-closed one-form on
the other.  On `S^2 x S^2`, only the harmonic bidegrees `(1,l)` and `(l,1)`
survive; there are no modes with both degrees at least two.  Every nonzero
element of this entire coupled quotient has a nearby moving plane with
negative curvature at quadratic order.  This includes cross-degree
superpositions, simultaneous mixtures of the two one-sided towers, and
arbitrary elements of the full `(1,1)` block.  Thus every canonical corrected
Hessian kernel path fails at quartic order.  The canonical
diagonal rotational tensor in the finite `(1,1)` block is transverse to gauge
and product variations and makes every
background-mixed curvature nonnegative, but an exact oblique plane has
curvature `-3 s^2 / (2 (1 - s)^2 (1 + s))`.  Hence this first genuinely
coupled repair fails quartically after setting `s = t^2`.  The
paper also gives an explicit dimension-free negative-curvature margin on
normalized parameter annuli: if
`epsilon <= |gamma|(sigma_1 + sigma_2) <= 1 - epsilon`, then some plane has
curvature less than `-epsilon^4 / 128`.  A separate rational example records a
certified plane-wise sign-change threshold at
`0.935934207...`.
At the isotropic `3 x 3` endpoint, where all singular values agree, an exact
quaternion factorization proves the complementary result: every
background-mixed sectional curvature is nonnegative throughout the
Riemannian interval.  Nevertheless, an explicit oblique plane has curvature
`-delta^2 / (2 (1 - 2 delta)^2)` for every nonzero parameter, so the
isotropic ray also leaves the full nonnegative-curvature cone immediately.
These are results for the stated family, not a proof of the Hopf conjecture.

## Reproduce the checks

Create an environment and install the two dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the independent symbolic and numerical verifier:

```sh
make verify PYTHON=.venv/bin/python
```

Build the paper (a TeX distribution with `latexmk` is required):

```sh
make pdf
```

The resulting paper is `output/pdf/hopf_seam.pdf`.

## Repository layout

- `hopf_seam.tex` — paper source
- `verify_hessian_tori.py` — direct symbolic curvature computation and
  deterministic threshold checks
- `verify_mixed_curvature_counterexample.py` — exact-arithmetic verification
  of the general mixed-curvature identity, the spectral-gap theorem, the
  isotropic quaternion factorization, the general top-critical oblique
  obstruction (and its two-jet coefficient), the rank-one zero-critical
  obstruction, the traced two-jet identity, the zero-threshold
  specialization, and the plane-wise
  nonzero threshold
- `verify_killing_kernel.py` — exact-arithmetic verification of the diagonal
  rotational kernel metric, its sharp Riemannian range, a flat diagonal mixed
  plane, and the negative oblique curvature formula
- `verify_coupled_kernel_classification.py` — exact finite-mode verification
  of the Killing-operator factorization, the full tensor-product kernel, and
  the gauge/co-closed quotient normal form
- `verify_coupled_kernel_obstruction.py` — exact verification of the optimized
  two-parameter moving-plane coefficient, a nonseparated degree-two test,
  the spherical spectral-gap inequality, and the signed-SVD obstruction for
  the full `(1,1)` block
- `requirements.txt` — Python dependencies used by the verifier
- `legacy/` — superseded exploratory material retained for provenance; it is
  not evidence for the paper and includes scripts with obsolete sign
  conventions and discarded formulas

The mathematical source of truth is the paper and the focused verifier at
the repository root.
