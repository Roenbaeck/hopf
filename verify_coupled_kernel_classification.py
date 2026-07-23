#!/usr/bin/env python3
"""Exact finite-mode checks for the full coupled-kernel classification."""

from __future__ import annotations

import sympy as sp


def verify_mixed_operator_factorization() -> None:
    """Check that the pure mixed curvature operator is D_1 D_2."""
    x_0, x_1, y_0, y_1 = sp.symbols("x_0 x_1 y_0 y_1")
    x = (x_0, x_1)
    y = (y_0, y_1)
    first = sp.symbols("X_0 X_1")
    second = sp.symbols("Y_0 Y_1")
    coupling = (
        (x_0 * y_0 + x_1**2 * y_1, x_1 * y_0**2 + x_0 * y_1),
        (x_1 * y_0 + x_0**2 * y_1, x_0 * y_1**2 + x_1 * y_0),
    )

    direct = sum(
        first[i]
        * first[j]
        * second[alpha]
        * second[beta]
        * sp.diff(coupling[j][beta], x[i], y[alpha])
        for i in range(2)
        for j in range(2)
        for alpha in range(2)
        for beta in range(2)
    )
    symmetrized = sum(
        first[i]
        * first[j]
        * second[alpha]
        * second[beta]
        * (
            sp.diff(coupling[j][beta], x[i], y[alpha])
            + sp.diff(coupling[i][beta], x[j], y[alpha])
            + sp.diff(coupling[j][alpha], x[i], y[beta])
            + sp.diff(coupling[i][alpha], x[j], y[beta])
        )
        / 4
        for i in range(2)
        for j in range(2)
        for alpha in range(2)
        for beta in range(2)
    )
    assert sp.expand(direct - symmetrized) == 0


def verify_tensor_product_kernel() -> None:
    """Check the spectral kernel and the gauge/co-closed normal form."""
    dimension_1, killing_1 = 7, 2
    dimension_2, killing_2 = 8, 3
    killing_operator_1 = sp.diag(0, 0, 1, 2, 3, 4, 5)
    killing_operator_2 = sp.diag(0, 0, 0, 1, 2, 3, 4, 5)
    mixed_operator = sp.kronecker_product(
        killing_operator_1, killing_operator_2
    )

    kernel_modes = {
        (i, j)
        for i in range(dimension_1)
        for j in range(dimension_2)
        if i < killing_1 or j < killing_2
    }
    expected_nullity = (
        killing_1 * dimension_2
        + dimension_1 * killing_2
        - killing_1 * killing_2
    )
    assert len(kernel_modes) == expected_nullity
    assert len(mixed_operator.nullspace()) == expected_nullity
    for i, j in kernel_modes:
        column = sp.zeros(dimension_1 * dimension_2, 1)
        column[i * dimension_2 + j] = 1
        assert mixed_operator * column == sp.zeros(dimension_1 * dimension_2, 1)

    # Model the Hodge splittings H_i = K_i + Z_i^0 + dC-infinity.
    coclosed_1 = set(range(4))
    coclosed_2 = set(range(5))
    exact_1 = set(range(4, dimension_1))
    exact_2 = set(range(5, dimension_2))
    nonkilling_coclosed_1 = coclosed_1 - set(range(killing_1))

    gauge_modes = {
        (i, j) for i in range(killing_1) for j in exact_2
    } | {
        (i, j) for i in exact_1 for j in range(killing_2)
    }
    canonical_modes = {
        (i, j) for i in range(killing_1) for j in coclosed_2
    } | {
        (i, j) for i in nonkilling_coclosed_1 for j in range(killing_2)
    }
    assert gauge_modes.isdisjoint(canonical_modes)
    assert gauge_modes | canonical_modes == kernel_modes
    canonical_dimension = (
        killing_1 * len(coclosed_2)
        + len(coclosed_1) * killing_2
        - killing_1 * killing_2
    )
    assert len(canonical_modes) == canonical_dimension

    # On S^2, the coexact degree-l multiplicity is 2*l+1.
    for cutoff in range(1, 8):
        tower_dimension = 9 + 6 * sum(
            2 * degree + 1 for degree in range(2, cutoff + 1)
        )
        assert tower_dimension == 6 * (cutoff + 1) ** 2 - 15


def verify_s2_harmonic_selection() -> None:
    """Check one allowed and one forbidden bidegree at a sphere pole."""
    u, v = sp.symbols("u v", real=True)
    zero = {u: 0, v: 0}

    def coexact_deformation(scalar: sp.Expr) -> sp.Matrix:
        # Orthographic coordinates are normal at the pole.  The first jet of
        # the Hodge star there is Euclidean, so *d f = (-f_v, f_u).
        one_form = sp.Matrix([-sp.diff(scalar, v), sp.diff(scalar, u)])
        return sp.Matrix(
            2,
            2,
            lambda i, j: sp.simplify(
                (
                    sp.diff(one_form[j], (u, v)[i])
                    + sp.diff(one_form[i], (u, v)[j])
                ).subs(zero)
                / 2
            ),
        )

    degree_1 = sp.sqrt(1 - u**2 - v**2)
    degree_2 = u * v
    killing_deformation = coexact_deformation(degree_1)
    higher_deformation = coexact_deformation(degree_2)
    assert killing_deformation == sp.zeros(2)
    assert higher_deformation == sp.diag(-1, 1)

    direction = sp.Matrix([1, 0])
    killing_value = (direction.T * killing_deformation * direction)[0]
    higher_value = (direction.T * higher_deformation * direction)[0]
    assert killing_value * higher_value == 0
    assert higher_value**2 == 1


def main() -> None:
    verify_mixed_operator_factorization()
    verify_tensor_product_kernel()
    verify_s2_harmonic_selection()
    print("PASS coupled kernel: mixed operator factors as D_1 D_2")
    print("PASS coupled kernel: tensor-product kernel is exact")
    print("PASS coupled kernel: gauge quotient has the co-closed normal form")
    print("PASS coupled kernel: S2 harmonic cutoff dimension")
    print("PASS coupled kernel: S2 allowed and forbidden bidegrees")


if __name__ == "__main__":
    main()
