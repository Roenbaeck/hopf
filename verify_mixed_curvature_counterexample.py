#!/usr/bin/env python3
"""Exact checks for the mixed-curvature formula and spectral consequences.

All asserted quantities are SymPy rationals or rational functions.  The
coordinate calculation differentiates the full orthographic-chart metric;
it does not use finite differences or the closed formulas it is checking.
"""

from __future__ import annotations

import sympy as sp


def direct_mixed_curvature(
    matrix: sp.Matrix,
    gamma: sp.Expr,
    x_direction: sp.Matrix,
    y_direction: sp.Matrix,
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Return linear part, R(X,Y,X,Y), area squared, and sectional K."""
    coordinates = sp.symbols("u_1 u_2 v_1 v_2", real=True)
    u = sp.Matrix(coordinates[:2])
    v = sp.Matrix(coordinates[2:])
    zero = {coordinate: 0 for coordinate in coordinates}

    x = sp.Matrix([sp.sqrt(1 - u.dot(u)), u[0], u[1]])
    y = sp.Matrix([sp.sqrt(1 - v.dot(v)), v[0], v[1]])
    jacobian_x = x.jacobian(u)
    jacobian_y = y.jacobian(v)
    h_x = jacobian_x.T * jacobian_x
    h_y = jacobian_y.T * jacobian_y
    coupling = jacobian_x.T * matrix * jacobian_y
    lam = 1 - gamma * (x.T * matrix * y)[0]
    metric = (lam * h_x).row_join(gamma * coupling).col_join(
        (gamma * coupling.T).row_join(lam * h_y)
    )
    metric_zero = metric.subs(zero)

    dimension = 4
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
    inverse = metric_zero.inv()

    def second_contraction(
        first: sp.Matrix, second: sp.Matrix, third: sp.Matrix, fourth: sp.Matrix
    ) -> sp.Expr:
        return sum(
            second_derivative[i][j][k][l]
            * first[i]
            * second[j]
            * third[k]
            * fourth[l]
            for i in range(dimension)
            for j in range(dimension)
            for k in range(dimension)
            for l in range(dimension)
        )

    linear = sp.factor(
        second_contraction(x_direction, y_direction, y_direction, x_direction)
        - sp.Rational(1, 2)
        * second_contraction(x_direction, x_direction, y_direction, y_direction)
        - sp.Rational(1, 2)
        * second_contraction(y_direction, y_direction, x_direction, x_direction)
    )

    def first_kind_contraction(first: sp.Matrix, second: sp.Matrix) -> sp.Matrix:
        return sp.Matrix(
            [
                sum(
                    christoffel_first[p][i][j] * first[i] * second[j]
                    for i in range(dimension)
                    for j in range(dimension)
                )
                for p in range(dimension)
            ]
        )

    gamma_xy = first_kind_contraction(x_direction, y_direction)
    gamma_xx = first_kind_contraction(x_direction, x_direction)
    gamma_yy = first_kind_contraction(y_direction, y_direction)
    numerator = sp.factor(
        linear
        + (gamma_xy.T * inverse * gamma_xy)[0]
        - (gamma_yy.T * inverse * gamma_xx)[0]
    )
    denominator = sp.factor(
        (x_direction.T * metric_zero * x_direction)[0]
        * (y_direction.T * metric_zero * y_direction)[0]
        - (x_direction.T * metric_zero * y_direction)[0] ** 2
    )
    return linear, numerator, denominator, sp.factor(numerator / denominator)


def covector_formula_numerator(
    matrix: sp.Matrix,
    gamma: sp.Expr,
    x_direction: sp.Matrix,
    y_direction: sp.Matrix,
) -> sp.Expr:
    """The coordinate-free theorem specialized at (e_0,e_0)."""
    a = matrix[1:, 0]
    b = matrix[0, 1:].T
    tangent_coupling = matrix[1:, 1:]
    r = x_direction.dot(x_direction)
    w = y_direction.dot(y_direction)
    alpha = a.dot(x_direction)
    beta = b.dot(y_direction)
    eta_xx = (r * a / 2 - alpha * x_direction).col_join(-r * b / 2)
    eta_xy = (-beta * x_direction / 2).col_join(-alpha * y_direction / 2)
    eta_yy = (-w * a / 2).col_join(w * b / 2 - beta * y_direction)
    lam = 1 - gamma * matrix[0, 0]
    metric = (lam * sp.eye(2)).row_join(gamma * tangent_coupling).col_join(
        (gamma * tangent_coupling.T).row_join(lam * sp.eye(2))
    )
    inverse = metric.inv()
    return sp.factor(
        gamma**2
        * ((eta_xy.T * inverse * eta_xy)[0] - (eta_xx.T * inverse * eta_yy)[0])
    )


def verify_two_jet_trace_identity() -> None:
    """Check that the trace of the mixed two-jet functional is a divergence."""
    coordinates = sp.symbols("x_1 x_2 y_1 y_2", real=True)
    first_indices = range(2)
    second_indices = range(2, 4)
    tensor = sp.zeros(4)
    for first in range(4):
        for second in range(first, 4):
            coefficient = first + second + 2
            value = (
                coefficient * coordinates[0] * coordinates[2]
                + (coefficient + 1) * coordinates[1] * coordinates[3]
                + (first + 1) * coordinates[0] ** 2
                + (second + 1) * coordinates[2] ** 2
                + (first + second + 1) * coordinates[1] * coordinates[2]
            )
            tensor[first, second] = value
            tensor[second, first] = value

    mixed_trace = sp.Integer(0)
    for first in first_indices:
        for second in second_indices:
            mixed_trace += (
                sp.diff(
                    tensor[first, second],
                    coordinates[first],
                    coordinates[second],
                )
                - sp.diff(tensor[second, second], coordinates[first], 2) / 2
                - sp.diff(tensor[first, first], coordinates[second], 2) / 2
            )

    trace_first = sum(tensor[index, index] for index in first_indices)
    trace_second = sum(tensor[index, index] for index in second_indices)
    mixed_divergence = sum(
        sp.diff(
            tensor[first, second],
            coordinates[first],
            coordinates[second],
        )
        for first in first_indices
        for second in second_indices
    )
    expected_divergence = (
        mixed_divergence
        - sum(
            sp.diff(trace_second, coordinates[first], 2)
            for first in first_indices
        )
        / 2
        - sum(
            sp.diff(trace_first, coordinates[second], 2)
            for second in second_indices
        )
        / 2
    )
    assert mixed_trace != 0
    assert sp.expand(mixed_trace - expected_divergence) == 0


def verify_zero_threshold(parameter: sp.Symbol) -> None:
    matrix = sp.Matrix([[0, 2, 0], [1, 0, 0], [0, 1, 0]])
    x_direction = sp.Matrix([1, 0, 0, 0])
    y_direction = sp.Matrix([0, 0, 1, 0])
    linear, numerator, denominator, curvature = direct_mixed_curvature(
        matrix, parameter, x_direction, y_direction
    )
    expected = -3 * parameter**4 / (4 * (1 - parameter**2))
    theorem_numerator = covector_formula_numerator(
        matrix, parameter, sp.Matrix([1, 0]), sp.Matrix([1, 0])
    )

    assert matrix.T * matrix == sp.diag(1, 5, 0)
    assert linear == 0
    assert sp.factor(numerator - theorem_numerator) == 0
    assert denominator == 1
    assert sp.factor(curvature - expected) == 0


def verify_generic_spectral_gap(parameter: sp.Symbol) -> None:
    """Check the generic negative-curvature reduction and a direct example."""
    spectral_top, spectral_middle, spectral_bottom = sp.symbols(
        "B sigma_2 mu", positive=True
    )
    transverse = sp.symbols("nu", real=True)

    # In the SVD-adapted tangent basis used in the proof, the only relevant
    # tangent coupling is the row (mu, nu).  This calculation is independent
    # of the orthographic-coordinate curvature calculation below.
    coupling = sp.Matrix([[0, 0], [spectral_bottom, transverse]])
    metric = sp.eye(4)
    metric[:2, 2:] = parameter * coupling
    metric[2:, :2] = parameter * coupling.T
    eta_xx = sp.Matrix(
        [-spectral_middle / 2, 0, -spectral_top / 2, 0]
    )
    eta_xy = sp.Matrix(
        [-spectral_top / 2, 0, -spectral_middle / 2, 0]
    )
    inverse = metric.inv()
    reduced = sp.factor(
        parameter**2
        * (
            (eta_xy.T * inverse * eta_xy)[0]
            - (eta_xx.T * inverse * eta_xx)[0]
        )
    )
    expected_reduced = -(
        parameter**4
        * (spectral_top**2 - spectral_middle**2)
        * spectral_bottom**2
        / (4 * (1 - parameter**2 * (spectral_bottom**2 + transverse**2)))
    )
    assert sp.factor(reduced - expected_reduced) == 0

    # Direct exact-coordinate check for the canonical construction with
    # singular values (3,2,1).  The base point and tangent frames are rotated
    # before direct_mixed_curvature differentiates the full chart metric.
    sigma_1, sigma_2, sigma_3 = sp.Integer(3), sp.Integer(2), sp.Integer(1)
    c = sp.sqrt(11) / 4
    d = sp.sqrt(5) / 4
    big_b = sp.sqrt((sigma_1**2 + sigma_2**2) / 2)
    x = sp.Matrix([c, 0, d])
    x_direction_ambient = sp.Matrix([0, 1, 0])
    x_complement = sp.Matrix([d, 0, -c])
    y = sp.Matrix([0, 1, 0])
    y_direction_ambient = sp.Matrix(
        [sigma_1 * c, 0, sigma_3 * d]
    ) / big_b
    y_complement = sp.Matrix(
        [-sigma_3 * d, 0, sigma_1 * c]
    ) / big_b
    left_frame = x.row_join(x_direction_ambient).row_join(x_complement)
    right_frame = y.row_join(y_direction_ambient).row_join(y_complement)
    matrix = left_frame.T * sp.diag(sigma_1, sigma_2, sigma_3) * right_frame
    x_direction = sp.Matrix([1, 0, 0, 0])
    y_direction = sp.Matrix([0, 0, 1, 0])
    linear, numerator, denominator, curvature = direct_mixed_curvature(
        matrix, parameter, x_direction, y_direction
    )
    theorem_numerator = covector_formula_numerator(
        matrix, parameter, sp.Matrix([1, 0]), sp.Matrix([1, 0])
    )
    kappa_squared = (
        sigma_1**2 - sigma_2**2 + 2 * sigma_3**2
    ) / 2
    expected = -(
        parameter**4
        * (sigma_1**2 - sigma_2**2) ** 2
        * (sigma_1**2 + sigma_2**2 - 2 * sigma_3**2)
        / (
            16
            * (sigma_1**2 + sigma_2**2)
            * (1 - parameter**2 * kappa_squared)
        )
    )

    characteristic = sp.factor((matrix.T * matrix).charpoly().as_expr())
    characteristic_expected = (
        (sp.Symbol("lambda") - 9)
        * (sp.Symbol("lambda") - 4)
        * (sp.Symbol("lambda") - 1)
    )
    assert characteristic == characteristic_expected
    assert linear == 0
    assert denominator == 1
    assert sp.factor(numerator - theorem_numerator) == 0
    assert sp.factor(curvature - expected) == 0


def verify_top_critical_oblique() -> None:
    """Check the general top-critical obstruction and its specializations."""
    top, second, third = sp.symbols("x y z", positive=True)
    d = 1 - top
    lambda_second_plus = d + second
    lambda_second_minus = d - second
    lambda_third_plus = d + third
    lambda_third_minus = d - third
    delta_second = lambda_second_plus * lambda_second_minus
    delta_third = lambda_third_plus * lambda_third_minus
    root_product = (
        sp.sqrt(lambda_second_plus)
        * sp.sqrt(lambda_second_minus)
        * sp.sqrt(lambda_third_plus)
        * sp.sqrt(lambda_third_minus)
    )

    symmetric_second = sp.Matrix([1, 0, 1, 0]) / sp.sqrt(
        2 * lambda_second_plus
    )
    antisymmetric_second = sp.Matrix([1, 0, -1, 0]) / sp.sqrt(
        2 * lambda_second_minus
    )
    symmetric_third = sp.Matrix([0, 1, 0, 1]) / sp.sqrt(
        2 * lambda_third_plus
    )
    antisymmetric_third = sp.Matrix([0, 1, 0, -1]) / sp.sqrt(
        2 * lambda_third_minus
    )
    oblique_first = (symmetric_second - antisymmetric_second) / sp.sqrt(2)
    oblique_second = (symmetric_third + antisymmetric_third) / sp.sqrt(2)

    _, _, area, curvature = direct_mixed_curvature(
        sp.diag(top, second, third),
        sp.Integer(1),
        oblique_first,
        oblique_second,
    )
    numerator = (
        d * (second**2 + third**2)
        + (2 * top - 1) * (d**2 - root_product)
    )
    expected = -numerator / (2 * delta_second * delta_third)
    assert sp.factor(area - 1) == 0
    assert sp.factor(curvature - expected) == 0

    # Recover the repeated-top formula exactly.
    parameter, tau = sp.symbols("delta tau", positive=True)
    repeated_a = 1 - 2 * parameter
    repeated_pi = (1 - parameter) ** 2 - parameter**2 * tau**2
    repeated_root = sp.sqrt(repeated_a) * sp.sqrt(repeated_pi)
    repeated_numerator = sp.factor(
        numerator.subs(
            {
                top: parameter,
                second: parameter,
                third: tau * parameter,
                root_product: repeated_root,
            }
        )
    )
    expected_repeated_numerator = sp.factor(
        repeated_a * repeated_root
        - (1 - parameter) * (repeated_pi - parameter)
    )
    assert sp.factor(repeated_numerator - expected_repeated_numerator) == 0
    isotropic_curvature = sp.factor(
        expected.subs(
            {
                top: parameter,
                second: parameter,
                third: parameter,
            }
        )
    )
    assert sp.factor(
        isotropic_curvature
        + parameter**2 / (2 * (1 - 2 * parameter) ** 2)
    ) == 0

    # The oblique obstruction is quadratic after the Hessian first
    # variation is removed by its infinitesimal diffeomorphism.  Scaling
    # all three singular values by the path parameter isolates that
    # two-jet coefficient.
    scale = sp.symbols("t", positive=True)
    singular_first, singular_second, singular_third = sp.symbols(
        "sigma_1 sigma_2 sigma_3", positive=True
    )
    scaled_curvature = expected.xreplace(
        {
            top: scale * singular_first,
            second: scale * singular_second,
            third: scale * singular_third,
        }
    )
    two_jet_coefficient = sp.simplify(
        sp.limit(scaled_curvature / scale**2, scale, 0)
    )
    assert sp.factor(
        two_jet_coefficient
        + (singular_second**2 + singular_third**2) / 4
    ) == 0

    # Re-center a rank-one map at a zero singular critical pair.  The
    # resulting oblique plane is negative throughout |delta| < 1 and has
    # coefficient -delta^2/4 at the product metric.
    delta = sp.symbols("delta", positive=True)
    rank_one_first = (
        sp.Matrix([1, 0, 1, 0]) / sp.sqrt(2 * (1 + delta))
        - sp.Matrix([1, 0, -1, 0]) / sp.sqrt(2 * (1 - delta))
    ) / sp.sqrt(2)
    rank_one_second = sp.Matrix([0, 1, 0, 0])
    _, _, rank_one_area, rank_one_curvature = direct_mixed_curvature(
        sp.diag(0, delta, 0),
        sp.Integer(1),
        rank_one_first,
        rank_one_second,
    )
    rank_one_root = sp.sqrt(1 - delta) * sp.sqrt(1 + delta)
    rank_one_expected = -(1 - rank_one_root) / (2 * rank_one_root)
    assert sp.factor(rank_one_area - 1) == 0
    assert sp.factor(rank_one_curvature - rank_one_expected) == 0
    assert sp.limit(rank_one_curvature / delta**2, delta, 0) == -sp.Rational(1, 4)


def verify_isotropic_mixed_nonnegativity(parameter: sp.Symbol) -> None:
    """Check the quaternion factorization for the isotropic 3-by-3 case."""
    s, alpha, beta, p, q, r, z, transverse_a, transverse_b = sp.symbols(
        "s alpha beta p q r z a_perp b_perp", real=True
    )
    lam = 1 - parameter * s
    coupling = sp.Matrix([[p, q], [r, z]])
    metric = (lam * sp.eye(2)).row_join(parameter * coupling).col_join(
        (parameter * coupling.T).row_join(lam * sp.eye(2))
    )
    eta_xx = sp.Matrix(
        [-alpha, transverse_a, -beta, -transverse_b]
    ) / 2
    eta_xy = sp.Matrix([-beta, 0, -alpha, 0]) / 2
    eta_yy = sp.Matrix(
        [-alpha, -transverse_a, -beta, transverse_b]
    ) / 2
    numerator = sp.expand(
        transverse_a**2 * lam * (lam**2 - parameter**2 * (p**2 + q**2))
        + transverse_b**2 * lam * (lam**2 - parameter**2 * (p**2 + r**2))
        + 2
        * transverse_a
        * transverse_b
        * parameter
        * (lam**2 * z - parameter**2 * p * (p * z - q * r))
        + lam
        * parameter**2
        * (beta**2 - alpha**2)
        * (q**2 - r**2)
    )

    # This checks the block-inversion numerator before imposing orthogonality.
    adjugate_bracket = sp.factor(
        4
        * (
            (eta_xy.T * metric.adjugate() * eta_xy)[0]
            - (eta_xx.T * metric.adjugate() * eta_yy)[0]
        )
        - numerator
    )
    assert adjugate_bracket == 0

    q_0, q_1, q_2, q_3 = sp.symbols("q_0 q_1 q_2 q_3", real=True)
    rotation = sp.Matrix(
        [
            [
                1 - 2 * (q_2**2 + q_3**2),
                2 * (q_1 * q_2 - q_0 * q_3),
                2 * (q_1 * q_3 + q_0 * q_2),
            ],
            [
                2 * (q_1 * q_2 + q_0 * q_3),
                1 - 2 * (q_1**2 + q_3**2),
                2 * (q_2 * q_3 - q_0 * q_1),
            ],
            [
                2 * (q_1 * q_3 - q_0 * q_2),
                2 * (q_2 * q_3 + q_0 * q_1),
                1 - 2 * (q_1**2 + q_2**2),
            ],
        ]
    )
    substitutions = {
        s: rotation[0, 0],
        beta: rotation[0, 1],
        transverse_b: rotation[0, 2],
        alpha: rotation[1, 0],
        p: rotation[1, 1],
        q: rotation[1, 2],
        transverse_a: rotation[2, 0],
        r: rotation[2, 1],
        z: rotation[2, 2],
    }
    omega, xi, upsilon, zeta = q_0**2, q_1**2, q_2**2, q_3**2
    certificate = (
        xi * zeta * (1 - 2 * parameter * xi) * (1 + 2 * parameter * zeta)
        + omega
        * upsilon
        * (1 - 2 * parameter * omega)
        * (1 + 2 * parameter * upsilon)
        + 8 * parameter**2 * omega * xi * upsilon * zeta
    )
    expected = 8 * (1 - 2 * parameter * rotation[0, 0]) * certificate
    difference = sp.Poly(
        sp.expand(numerator.subs(substitutions) - expected), q_0
    )
    unit_relation = sp.Poly(
        q_0**2 + q_1**2 + q_2**2 + q_3**2 - 1, q_0
    )
    assert sp.factor(difference.rem(unit_relation).as_expr()) == 0

    # The same isotropic ray nevertheless has a strictly negative oblique
    # plane.  These are the exact coordinates of (D_1-A_1)/sqrt(2) and
    # (D_2+A_2)/sqrt(2) in the proof, with D_i and A_i metric-normalized.
    antidiagonal_weight = 1 - 2 * parameter
    oblique_first = sp.Matrix(
        [
            (1 - 1 / sp.sqrt(antidiagonal_weight)) / 2,
            0,
            (1 + 1 / sp.sqrt(antidiagonal_weight)) / 2,
            0,
        ]
    )
    oblique_second = sp.Matrix(
        [
            0,
            (1 + 1 / sp.sqrt(antidiagonal_weight)) / 2,
            0,
            (1 - 1 / sp.sqrt(antidiagonal_weight)) / 2,
        ]
    )
    _, _, area, curvature = direct_mixed_curvature(
        sp.eye(3), parameter, oblique_first, oblique_second
    )
    expected_curvature = -parameter**2 / (2 * antidiagonal_weight**2)
    assert sp.factor(area - 1) == 0
    assert sp.factor(curvature - expected_curvature) == 0


def verify_positive_threshold(parameter: sp.Symbol) -> sp.Expr:
    integer_matrix = sp.Matrix(
        [[1, 2, -32], [-48, -23, 3], [-39, -15, -3]]
    )
    matrix = integer_matrix / 100
    x_direction = sp.Matrix(
        [sp.Rational(63, 100), sp.Rational(78, 100), 0, 0]
    )
    y_direction = sp.Matrix([0, 0, sp.Rational(9, 100), -1])
    linear, numerator, denominator, curvature = direct_mixed_curvature(
        matrix, parameter, x_direction, y_direction
    )
    theorem_numerator = covector_formula_numerator(
        matrix, parameter, x_direction[:2, :], y_direction[2:, :]
    )

    polynomial = (
        1873383134707 * parameter**3
        - 187783301783700 * parameter**2
        - 6938955210000 * parameter
        + 169451579000000
    )
    area_factor = (
        28151192500 - 563023850 * parameter - 7287743 * parameter**2
    )
    determinant_factor = (
        489 * parameter**4
        + 6160 * parameter**3
        - 306400 * parameter**2
        - 160000 * parameter
        + 4000000
    )
    expected = (
        11
        * parameter**2
        * polynomial
        / (4 * area_factor * determinant_factor)
    )

    assert linear == 0
    assert sp.factor(numerator - theorem_numerator) == 0
    assert sp.factor(curvature - expected) == 0
    assert sp.factor(denominator - 9 * area_factor / 250000000000) == 0

    # The tight rational root bracket and uniqueness on the entire
    # Riemannian interval are exact checks.
    lower = sp.Rational(4679671, 5000000)
    upper = sp.Rational(9359343, 10000000)
    assert polynomial.subs(parameter, lower) > 0
    assert polynomial.subs(parameter, upper) < 0
    assert sp.diff(polynomial, parameter, 2).subs(parameter, 2) < 0
    assert sp.diff(polynomial, parameter).subs(parameter, 0) < 0

    twice_frobenius_squared = 2 * sum(entry**2 for entry in matrix)
    assert (
        sp.Rational(47, 50) ** 2 * twice_frobenius_squared
        == sp.Rational(6213917, 6250000)
        < 1
    )
    assert (matrix[:, 0].dot(matrix[:, 0])) == sp.Rational(3826, 10000) > sp.Rational(1, 4)

    roots = sp.nroots(polynomial, n=30, maxsteps=200)
    return next(root for root in roots if lower < sp.re(root) < upper)


def main() -> None:
    parameter = sp.symbols("gamma", real=True)
    verify_two_jet_trace_identity()
    verify_generic_spectral_gap(parameter)
    verify_top_critical_oblique()
    verify_isotropic_mixed_nonnegativity(parameter)
    verify_zero_threshold(parameter)
    root = verify_positive_threshold(parameter)
    print("PASS exact general mixed-curvature identity in tested coordinates")
    print("PASS integrated two-jet trace: pointwise trace is a divergence")
    print("PASS spectral gap: exact negative quartic curvature")
    print("PASS top-critical plane: exact obstruction for every rank >= 2")
    print("PASS two-jet obstruction: quadratic coefficient in every rank")
    print("PASS rank-one zero-critical plane: exact negative curvature")
    print("PASS isotropic 3-by-3 case: mixed nonnegativity and oblique negativity")
    print("PASS zero threshold: K = -3*gamma^4/(4*(1-gamma^2))")
    print(f"PASS unique nonzero threshold: gamma_* = {root}")


if __name__ == "__main__":
    main()
