#!/usr/bin/env python3
"""Independent checks for the Hessian-deformation paper.

This script intentionally derives the two-dimensional metric from the
potential and then computes its Levi-Civita connection and curvature.  It
does not encode ``K = 0`` as a special formula.  A small deterministic
numerical test also checks the sharp singular-value threshold in ambient
dimensions larger than the torus calculation.
"""

from __future__ import annotations

import itertools

import numpy as np
import sympy as sp


def assert_zero(expression: sp.Expr, label: str) -> None:
    """Raise a useful error unless SymPy proves that expression is zero."""
    simplified = sp.trigsimp(sp.factor(sp.simplify(expression)))
    if simplified != 0:
        raise AssertionError(f"{label} did not vanish: {simplified}")


def christoffel_symbols(
    metric: sp.Matrix, coordinates: tuple[sp.Symbol, ...]
) -> list[list[list[sp.Expr]]]:
    """Compute Gamma^k_ij from a metric matrix."""
    dimension = len(coordinates)
    inverse = sp.simplify(metric.inv())
    gamma = [
        [[sp.S.Zero for _ in range(dimension)] for _ in range(dimension)]
        for _ in range(dimension)
    ]
    for upper, lower_1, lower_2 in itertools.product(
        range(dimension), repeat=3
    ):
        gamma[upper][lower_1][lower_2] = sp.simplify(
            sp.Rational(1, 2)
            * sum(
                inverse[upper, index]
                * (
                    sp.diff(metric[index, lower_2], coordinates[lower_1])
                    + sp.diff(metric[index, lower_1], coordinates[lower_2])
                    - sp.diff(metric[lower_1, lower_2], coordinates[index])
                )
                for index in range(dimension)
            )
        )
    return gamma


def riemann_tensor(
    gamma: list[list[list[sp.Expr]]],
    coordinates: tuple[sp.Symbol, ...],
) -> list[list[list[list[sp.Expr]]]]:
    """Compute R^l_ijk directly from Christoffel symbols."""
    dimension = len(coordinates)
    curvature = [
        [
            [
                [[sp.S.Zero for _ in range(dimension)] for _ in range(dimension)]
                for _ in range(dimension)
            ]
            for _ in range(dimension)
        ]
        for _ in range(dimension)
    ]
    for upper, lower, direction_1, direction_2 in itertools.product(
        range(dimension), repeat=4
    ):
        value = sp.diff(
            gamma[upper][lower][direction_2], coordinates[direction_1]
        ) - sp.diff(
            gamma[upper][lower][direction_1], coordinates[direction_2]
        )
        value += sum(
            gamma[upper][direction_1][index]
            * gamma[index][lower][direction_2]
            - gamma[upper][direction_2][index]
            * gamma[index][lower][direction_1]
            for index in range(dimension)
        )
        curvature[upper][lower][direction_1][direction_2] = sp.simplify(value)
    return curvature


def symbolic_torus_check() -> None:
    """Derive the induced metric, its separated form, and its curvature."""
    u, v, alpha, beta = sp.symbols("u v alpha beta", real=True)
    gamma, sigma_i, sigma_j = sp.symbols(
        "gamma sigma_i sigma_j", real=True
    )

    phi = sigma_i * sp.cos(u) * sp.cos(v) + sigma_j * sp.sin(u) * sp.sin(v)
    assert_zero(sp.diff(phi, u, 2) + phi, "phi_uu + phi")
    assert_zero(sp.diff(phi, v, 2) + phi, "phi_vv + phi")

    hessian = sp.hessian(phi, (u, v))
    metric_uv = sp.eye(2) + gamma * hessian

    # u=(alpha+beta)/2 and v=(alpha-beta)/2.
    substitution = {u: (alpha + beta) / 2, v: (alpha - beta) / 2}
    jacobian = sp.Matrix(
        [[sp.Rational(1, 2), sp.Rational(1, 2)],
         [sp.Rational(1, 2), -sp.Rational(1, 2)]]
    )
    metric_ab = sp.simplify(
        jacobian.T * metric_uv.subs(substitution) * jacobian
    ).applyfunc(sp.trigsimp)

    a = (sigma_i + sigma_j) / 2
    b = (sigma_i - sigma_j) / 2
    expected = sp.diag(
        sp.Rational(1, 2) - gamma * b * sp.cos(alpha),
        sp.Rational(1, 2) - gamma * a * sp.cos(beta),
    )
    for row, column in itertools.product(range(2), repeat=2):
        assert_zero(
            metric_ab[row, column] - expected[row, column],
            f"separated metric entry ({row}, {column})",
        )

    connection = christoffel_symbols(metric_ab, (alpha, beta))
    curvature = riemann_tensor(connection, (alpha, beta))
    for indices in itertools.product(range(2), repeat=4):
        value = curvature[indices[0]][indices[1]][indices[2]][indices[3]]
        assert_zero(value, f"Riemann component {indices}")

    endpoint = 1 / (sigma_i + sigma_j)
    positive_block = sp.Matrix(
        [
            [1 - endpoint * sigma_i, endpoint * sigma_j],
            [endpoint * sigma_j, 1 - endpoint * sigma_i],
        ]
    )
    negative_block = sp.Matrix(
        [
            [1 - endpoint * sigma_i, -endpoint * sigma_j],
            [-endpoint * sigma_j, 1 - endpoint * sigma_i],
        ]
    )
    positive_endpoint = (
        sp.Matrix([1, -1]).T * positive_block * sp.Matrix([1, -1])
    )[0]
    negative_endpoint = (
        sp.Matrix([1, 1]).T * negative_block * sp.Matrix([1, 1])
    )[0]
    assert_zero(positive_endpoint, "positive endpoint null vector")
    assert_zero(negative_endpoint, "negative endpoint null vector")


def symbolic_ambient_hessian_check() -> None:
    """Check the ambient Hessian formula in exact S2 x S2 coordinates."""
    theta, phi, psi, chi = sp.symbols("theta phi psi chi", real=True)
    coordinates = (theta, phi, psi, chi)
    x = sp.Matrix(
        [
            sp.sin(theta) * sp.cos(phi),
            sp.sin(theta) * sp.sin(phi),
            sp.cos(theta),
        ]
    )
    y = sp.Matrix(
        [
            sp.sin(psi) * sp.cos(chi),
            sp.sin(psi) * sp.sin(chi),
            sp.cos(psi),
        ]
    )
    # This exact nondiagonal matrix exercises every type of mixed term.
    matrix = sp.Matrix([[2, 1, -1], [3, 0, 2], [-2, 1, 4]])
    potential = (x.T * matrix * y)[0]
    background = sp.diag(1, sp.sin(theta) ** 2, 1, sp.sin(psi) ** 2)
    connection = christoffel_symbols(background, coordinates)

    derivatives_x = (sp.diff(x, theta), sp.diff(x, phi))
    derivatives_y = (sp.diff(y, psi), sp.diff(y, chi))
    expected = sp.zeros(4)
    expected[:2, :2] = -potential * background[:2, :2]
    expected[2:, 2:] = -potential * background[2:, 2:]
    for left_index, tangent_x in enumerate(derivatives_x):
        for right_index, tangent_y in enumerate(derivatives_y, start=2):
            mixed = (tangent_x.T * matrix * tangent_y)[0]
            expected[left_index, right_index] = mixed
            expected[right_index, left_index] = mixed

    for row, column in itertools.product(range(4), repeat=2):
        coordinate_hessian = sp.diff(
            potential, coordinates[row], coordinates[column]
        ) - sum(
            connection[index][row][column]
            * sp.diff(potential, coordinates[index])
            for index in range(4)
        )
        assert_zero(
            coordinate_hessian - expected[row, column],
            f"ambient Hessian entry ({row}, {column})",
        )


def orthogonal_complement(vector: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis for the Euclidean orthogonal complement."""
    _, _, vh = np.linalg.svd(vector.reshape(1, -1), full_matrices=True)
    return vh[1:].T


def numerical_threshold_check() -> None:
    """Test the global bound on deterministic random rectangular examples."""
    rng = np.random.default_rng(20260722)
    for shape in ((5, 4), (4, 6), (2, 2)):
        matrix = rng.normal(size=shape)
        left, singular_values, right_transpose = np.linalg.svd(
            matrix, full_matrices=True
        )
        threshold_sum = singular_values[0] + singular_values[1]

        for sign in (-1.0, 1.0):
            gamma = sign * 0.99 / threshold_sum
            for _ in range(100):
                x = rng.normal(size=shape[0])
                x /= np.linalg.norm(x)
                y = rng.normal(size=shape[1])
                y /= np.linalg.norm(y)
                tangent_x = orthogonal_complement(x)
                tangent_y = orthogonal_complement(y)
                coupling = tangent_x.T @ matrix @ tangent_y
                scalar = 1.0 - gamma * (x @ matrix @ y)
                block = np.block(
                    [
                        [
                            scalar * np.eye(shape[0] - 1),
                            gamma * coupling,
                        ],
                        [
                            gamma * coupling.T,
                            scalar * np.eye(shape[1] - 1),
                        ],
                    ]
                )
                if np.linalg.eigvalsh(block)[0] <= 1.0e-10:
                    raise AssertionError("metric failed inside the sharp interval")

        # The singular-frame witness is nonpositive beyond either endpoint.
        u1, u2 = left[:, 0], left[:, 1]
        v1, v2 = right_transpose.T[:, 0], right_transpose.T[:, 1]
        gamma = 1.01 / threshold_sum
        positive_witness = 2 * (
            1 - gamma * (u1 @ matrix @ v1) - gamma * (u2 @ matrix @ v2)
        )
        gamma = -1.01 / threshold_sum
        negative_base_value = (-u1) @ matrix @ v1
        negative_witness = 2 * (
            1 - gamma * negative_base_value + gamma * (u2 @ matrix @ v2)
        )
        if positive_witness >= 0 or negative_witness >= 0:
            raise AssertionError("endpoint singular-frame witness has wrong sign")


def main() -> None:
    symbolic_ambient_hessian_check()
    print("PASS ambient Hessian block formula in exact S2 x S2 coordinates")
    symbolic_torus_check()
    print("PASS symbolic Hessian and separated-metric identities")
    print("PASS direct Levi-Civita/Riemann computation: torus curvature is zero")
    numerical_threshold_check()
    print("PASS deterministic tests of the sharp singular-value threshold")


if __name__ == "__main__":
    main()
