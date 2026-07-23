# Mathematical assessment after the rework

The repository now contains a defensible, self-contained paper.  Its claim is
deliberately limited to the Hessian family

\[
h+\gamma\operatorname{Hess}_h\langle x,Ay\rangle
\quad\text{on}\quad S^m\times S^n.
\]

The paper proves, rather than numerically infers:

1. the exact positive-definiteness interval in terms of the two largest
   singular values of `A`;
2. the existence of flat totally geodesic tori throughout that interval;
3. the resulting obstruction to strictly positive sectional curvature;
4. the exact subgroup of factor-preserving round-product isometries retained
   by the deformation;
5. an exact formula for every background-mixed sectional curvature;
6. a zero-threshold example with negative curvature at every nonzero
   parameter and a separate ray with a unique certified nonzero threshold;
7. a spectral-gap theorem forcing negative curvature along every nonzero
   ray with `sigma_1 > sigma_2`;
8. a singular-subsphere reduction that embeds matched lower-dimensional
   Hessian models as totally geodesic submanifolds;
9. a uniform top-critical oblique-plane obstruction for every matrix of rank
   at least two, propagated by that reduction to every dimension `m,n >= 2`;
10. an exact zero-critical oblique-plane obstruction for rank-one matrices,
    with a negative quadratic coefficient at the product metric;
11. a two-jet normal form modulo diffeomorphisms for arbitrary paths
    `h + t Hess(f) + t^2 B + O(t^3)`, an integrated trace identity forcing
    every nonnegative mixed coefficient to vanish, and explicit repair
    identities for bilinear potentials;
12. a full classification of the mixed two-jet kernel modulo
    diffeomorphisms and factor variations: each coupled class has a Killing
    leg on one factor and a co-closed leg on the other; on `S^2 x S^2` the
    surviving harmonic bidegrees are exactly `(1,l)` and `(l,1)`;
13. a full nonlinear obstruction for every nonzero element of the coupled
    quadratic quotient, including cross-degree superpositions, simultaneous
    mixtures of both one-sided towers, and arbitrary elements of the
    `9`-dimensional `(1,1)` block;
14. an exact transverse rotational example in the `(1,1)` kernel block whose
    fixed mixed curvatures are nonnegative but which has a negative quartic
    oblique obstruction;
15. four-dimensional detection of every negative-curvature obstruction and
    an explicit dimension-free negative margin on normalized parameter
    annuli.

The previous general rule classification, sequential conformal argument,
Kähler section, arbitrary Hessian-flatness assertion, and unsupported
numerical claims have been removed from the paper.  Earlier exploratory
scripts are retained only under `legacy/` for provenance.

Together, the rank-one and rank-at-least-two curvature results prove that in
every dimension `m,n >= 2` and for every nonzero `A`, the round product is the
unique nonnegatively curved metric on its Riemannian Hessian ray.  The two-jet
theorem additionally
narrows nonlinear continuations sharing that first variation: any such path
must cancel every quadratic mixed-curvature coefficient.  The simultaneous
conditions are feasible: the canonical correction restores the
diffeomorphism orbit, and ordinary product variations also lie in the kernel.
The paper now classifies the entire coupled quadratic kernel and obstructs
every nonzero pure kernel ray, including all cross-degree, two-tower, and
`(1,1)` mixtures.  Its canonical corrected Hessian paths therefore fail at
quartic order.  The remaining nonlinear search space begins at higher jets:
third- and higher-order metric corrections may interact with the quartic
obstruction.  A still broader result would control those higher-jet
compatibility conditions, treat potentials beyond the bilinear degree-one
family, or classify more of the sectional curvature.
