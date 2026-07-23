#!/usr/bin/env python3
"""Exact checks for coupled Killing-kernel variations on S^2 x S^2."""

from __future__ import annotations

import sympy as sp


def orthographic_killing_metric(parameter: sp.Expr) -> tuple[tuple[sp.Symbol, ...], sp.Matrix]:
    """Return h + parameter*B for the diagonal rotational Killing coupling."""
    coordinates = sp.symbols("u_1 u_2 v_1 v_2", real=True)
    u = sp.Matrix(coordinates[:2])
    v = sp.Matrix(coordinates[2:])
    x = sp.Matrix([sp.sqrt(1 - u.dot(u)), u[0], u[1]])
    y = sp.Matrix([sp.sqrt(1 - v.dot(v)), v[0], v[1]])
    jacobian_x = x.jacobian(u)
    jacobian_y = y.jacobian(v)
    h_x = jacobian_x.T * jacobian_x
    h_y = jacobian_y.T * jacobian_y

    rotational_x = sp.Matrix.hstack(
        *(x.cross(jacobian_x[:, index]) for index in range(2))
    )
    rotational_y = sp.Matrix.hstack(
        *(y.cross(jacobian_y[:, index]) for index in range(2))
    )
    coupling = rotational_x.T * rotational_y
    metric = h_x.row_join(parameter * coupling).col_join(
        (parameter * coupling.T).row_join(h_y)
    )
    return coordinates, metric


def curvature_at_origin(
    coordinates: tuple[sp.Symbol, ...],
    metric: sp.Matrix,
    first: sp.Matrix,
    second: sp.Matrix,
) -> tuple[sp.Expr, sp.Expr]:
    """Return squared area and sectional curvature in the given plane."""
    zero = {coordinate: 0 for coordinate in coordinates}
    dimension = len(coordinates)
    metric_zero = metric.subs(zero)
    inverse = metric_zero.inv()
    first_derivative = [
        [
            [sp.diff(metric[i, j], coordinates[k]).subs(zero) for k in range(dimension)]
            for j in range(dimension)
        ]
        for i in range(dimension)
    ]
    second_derivative = [
        [
            [
                [
                    sp.diff(metric[i, j], coordinates[k], coordinates[l]).subs(zero)
                    for l in range(dimension)
                ]
                for k in range(dimension)
            ]
            for j in range(dimension)
        ]
        for i in range(dimension)
    ]
    christoffel_first = [
        [
            [
                sp.Rational(1, 2)
                * (
                    first_derivative[p][i][j]
                    + first_derivative[p][j][i]
                    - first_derivative[i][j][p]
                )
                for j in range(dimension)
            ]
            for i in range(dimension)
        ]
        for p in range(dimension)
    ]

    def second_contraction(
        vector_1: sp.Matrix,
        vector_2: sp.Matrix,
        vector_3: sp.Matrix,
        vector_4: sp.Matrix,
    ) -> sp.Expr:
        return sum(
            second_derivative[i][j][k][l]
            * vector_1[i]
            * vector_2[j]
            * vector_3[k]
            * vector_4[l]
            for i in range(dimension)
            for j in range(dimension)
            for k in range(dimension)
            for l in range(dimension)
        )

    linear = (
        second_contraction(first, second, second, first)
        - second_contraction(first, first, second, second) / 2
        - second_contraction(second, second, first, first) / 2
    )

    def first_kind_contraction(vector_1: sp.Matrix, vector_2: sp.Matrix) -> sp.Matrix:
        return sp.Matrix(
            [
                sum(
                    christoffel_first[p][i][j] * vector_1[i] * vector_2[j]
                    for i in range(dimension)
                    for j in range(dimension)
                )
                for p in range(dimension)
            ]
        )

    gamma_12 = first_kind_contraction(first, second)
    gamma_11 = first_kind_contraction(first, first)
    gamma_22 = first_kind_contraction(second, second)
    numerator = sp.factor(
        linear
        + (gamma_12.T * inverse * gamma_12)[0]
        - (gamma_11.T * inverse * gamma_22)[0]
    )
    area = sp.factor(
        (first.T * metric_zero * first)[0]
        * (second.T * metric_zero * second)[0]
        - (first.T * metric_zero * second)[0] ** 2
    )
    return area, sp.factor(numerator / area)


def main() -> None:
    parameter = sp.symbols("s", real=True)
    coordinates, metric = orthographic_killing_metric(parameter)

    symmetric_1 = sp.Matrix([1, 0, 1, 0]) / sp.sqrt(2 * (1 + parameter))
    antisymmetric_1 = sp.Matrix([1, 0, -1, 0]) / sp.sqrt(2 * (1 - parameter))
    symmetric_2 = sp.Matrix([0, 1, 0, 1]) / sp.sqrt(2 * (1 + parameter))
    antisymmetric_2 = sp.Matrix([0, 1, 0, -1]) / sp.sqrt(2 * (1 - parameter))
    oblique_first = (symmetric_1 - antisymmetric_1) / sp.sqrt(2)
    oblique_second = (symmetric_2 + antisymmetric_2) / sp.sqrt(2)
    area, curvature = curvature_at_origin(
        coordinates, metric, oblique_first, oblique_second
    )
    expected_oblique = -3 * parameter**2 / (
        2 * (1 - parameter) ** 2 * (1 + parameter)
    )
    assert sp.factor(area - 1) == 0
    assert sp.factor(curvature - expected_oblique) == 0

    mixed_first = sp.Matrix([1, 0, 0, 0])
    mixed_second = sp.Matrix([0, 0, 0, 1])
    mixed_area, mixed_curvature = curvature_at_origin(
        coordinates, metric, mixed_first, mixed_second
    )
    assert sp.factor(mixed_area - 1) == 0
    assert sp.factor(mixed_curvature) == 0

    zero = {coordinate: 0 for coordinate in coordinates}
    metric_zero = metric.subs(zero)
    assert sp.factor(metric_zero.det() - (1 - parameter**2) ** 2) == 0
    print("PASS rotational Killing kernel: exact Riemannian boundary |s| = 1")
    print("PASS rotational Killing kernel: diagonal mixed plane remains flat")
    print("PASS rotational Killing kernel: exact negative oblique curvature")


if __name__ == "__main__":
    main()
