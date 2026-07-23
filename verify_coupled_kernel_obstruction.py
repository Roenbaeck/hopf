#!/usr/bin/env python3
"""Exact checks for the nonlinear obstruction of the full coupled kernel."""

from __future__ import annotations

import sympy as sp

from verify_killing_kernel import curvature_at_origin


def sphere_average(polynomial: sp.Expr, variables: tuple[sp.Symbol, ...]) -> sp.Expr:
    """Integrate a polynomial over the unit two-sphere with probability area."""
    expanded = sp.Poly(sp.expand(polynomial), *variables)
    result = 0
    for powers, coefficient in expanded.terms():
        if any(power % 2 for power in powers):
            continue
        half_powers = tuple(power // 2 for power in powers)
        numerator = sp.prod(sp.factorial2(2 * power - 1) for power in half_powers)
        denominator = sp.factorial2(2 * sum(half_powers) + 1)
        result += coefficient * numerator / denominator
    return sp.factor(result)


def tangential_gradient(
    polynomial: sp.Expr, variables: tuple[sp.Symbol, ...]
) -> sp.Matrix:
    ambient_gradient = sp.Matrix([sp.diff(polynomial, variable) for variable in variables])
    radial = sp.Matrix(variables)
    return sp.simplify(ambient_gradient - radial * ambient_gradient.dot(radial))


def verify_general_coupled_moving_plane_coefficient() -> None:
    """Check the two-parameter moving-plane formula for a general kernel jet."""
    p, q, r, t, parameter, shift_second, shift_first = sp.symbols(
        "p q r t s a b", real=True
    )
    c_0, c_p, c_q, c_r, c_t, c_pq, c_rt = sp.symbols(
        "c_0 c_p c_q c_r c_t c_pq c_rt", real=True
    )
    d_0, d_p, d_pp = sp.symbols("d_0 d_p d_pp", real=True)
    e_0, e_r, e_rr = sp.symbols("e_0 e_r e_rr", real=True)

    first_sphere = sp.Matrix([sp.sqrt(1 - p**2 - q**2), p, q])
    second_sphere = sp.Matrix([sp.sqrt(1 - r**2 - t**2), r, t])
    first_jacobian = first_sphere.jacobian((p, q))
    second_jacobian = second_sphere.jacobian((r, t))
    first_metric = first_jacobian.T * first_jacobian
    second_metric = second_jacobian.T * second_jacobian

    # Only the displayed jets enter the quadratic coefficient.  The missing
    # p*r term in C_{p r} is the fixed-mixed kernel condition at the origin.
    mixed = sp.Matrix(
        [
            [
                c_0
                + c_p * p
                + c_q * q
                + c_r * r
                + c_t * t
                + c_pq * p * q
                + c_rt * r * t,
                e_0 + e_r * r + sp.Rational(1, 2) * e_rr * r**2,
            ],
            [
                d_0 + d_p * p + sp.Rational(1, 2) * d_pp * p**2,
                0,
            ],
        ]
    )
    metric = first_metric.row_join(parameter * mixed).col_join(
        (parameter * mixed.T).row_join(second_metric)
    )
    first_vector = sp.Matrix([1, 0, 0, parameter * shift_second])
    second_vector = sp.Matrix([0, parameter * shift_first, 1, 0])
    _, curvature = curvature_at_origin(
        (p, q, r, t), metric, first_vector, second_vector
    )
    coefficient = sp.factor(
        sp.diff(curvature, parameter, 2).subs(parameter, 0) / 2
    )

    first_curl = d_p - c_q
    second_curl = e_r - c_t
    first_curl_derivative = d_pp - c_pq
    second_curl_derivative = e_rr - c_rt
    expected = sp.factor(
        shift_second**2
        + shift_first**2
        - shift_second * second_curl_derivative
        - shift_first * first_curl_derivative
        + (first_curl**2 + second_curl**2) / 4
    )
    assert sp.factor(coefficient - expected) == 0
    optimized = sp.factor(
        expected.subs(
            {
                shift_second: second_curl_derivative / 2,
                shift_first: first_curl_derivative / 2,
            },
            simultaneous=True,
        )
    )
    optimized_expected = sp.factor(
        (
            first_curl**2
            + second_curl**2
            - first_curl_derivative**2
            - second_curl_derivative**2
        )
        / 4
    )
    assert sp.factor(optimized - optimized_expected) == 0


def verify_generic_moving_plane_coefficient() -> None:
    """Check the coefficient with all three first-factor Killing legs present."""
    p, q, r, t, parameter, plane_shift = sp.symbols(
        "p q r t s c", real=True
    )
    value_r, value_t = sp.symbols("value_r value_t", real=True)
    eta_rr, eta_rt, eta_tr = sp.symbols("eta_rr eta_rt eta_tr", real=True)
    eta_rrr, eta_rrt = sp.symbols("eta_rrr eta_rrt", real=True)
    normal_r, normal_t, normal_r_2, normal_t_2 = sp.symbols(
        "normal_r normal_t normal_r_2 normal_t_2", real=True
    )
    other_r, other_t, other_rr, other_rt, other_tr, other_tt = sp.symbols(
        "other_r other_t other_rr other_rt other_tr other_tt", real=True
    )

    first_sphere = sp.Matrix([p, sp.sqrt(1 - p**2 - q**2), q])
    second_sphere = sp.Matrix([sp.sqrt(1 - r**2 - t**2), r, t])
    first_jacobian = first_sphere.jacobian((p, q))
    second_jacobian = second_sphere.jacobian((r, t))
    first_metric = first_jacobian.T * first_jacobian
    second_metric = second_jacobian.T * second_jacobian

    axes = tuple(sp.eye(3)[:, index] for index in range(3))
    killing_forms = tuple(
        sp.simplify(first_jacobian.T * axis.cross(first_sphere)) for axis in axes
    )

    selected_form = sp.Matrix(
        [
            value_r
            + eta_rr * r
            + eta_rt * t
            + sp.Rational(1, 2) * eta_rrr * r**2
            + eta_rrt * r * t,
            value_t + eta_tr * r - eta_rr * t,
        ]
    )
    # The coefficient along the normal ambient axis must vanish at the point;
    # its first jet remains arbitrary.  This is exactly x perpendicular to
    # the image of the R^3-valued one-form E.
    normal_axis_form = sp.Matrix(
        [normal_r * r + normal_t * t, normal_r_2 * r + normal_t_2 * t]
    )
    other_tangent_form = sp.Matrix(
        [
            other_r + other_rr * r + other_rt * t,
            other_t + other_tr * r + other_tt * t,
        ]
    )
    coupling = (
        killing_forms[0] * selected_form.T
        + killing_forms[1] * normal_axis_form.T
        + killing_forms[2] * other_tangent_form.T
    )
    metric = first_metric.row_join(parameter * coupling).col_join(
        (parameter * coupling.T).row_join(second_metric)
    )

    first_vector = sp.Matrix([0, 1, 0, parameter * plane_shift])
    second_vector = sp.Matrix([0, 0, 1, 0])
    _, curvature = curvature_at_origin(
        (p, q, r, t), metric, first_vector, second_vector
    )
    coefficient = sp.factor(
        sp.diff(curvature, parameter, 2).subs(parameter, 0) / 2
    )
    curl = eta_tr - eta_rt
    curl_derivative = -eta_rrt
    expected = sp.factor(
        plane_shift**2
        - plane_shift * curl_derivative
        + sp.Rational(1, 4) * curl**2
    )
    assert sp.factor(coefficient - expected) == 0
    optimized = sp.factor(expected.subs(plane_shift, curl_derivative / 2))
    assert optimized == sp.factor((curl**2 - curl_derivative**2) / 4)


def verify_nonseparated_degree_two_mode() -> None:
    """Check a rank-two vector-valued degree-two spherical harmonic."""
    y_0, y_1, y_2 = sp.symbols("y_0 y_1 y_2", real=True)
    vector_harmonic = sp.Matrix(
        [y_0 * y_1, y_0 * y_2, (y_1**2 - y_2**2) / 2]
    )
    north = {y_0: 1, y_1: 0, y_2: 0}
    tangent_1 = sp.diff(vector_harmonic, y_1).subs(north)
    tangent_2 = sp.diff(vector_harmonic, y_2).subs(north)
    value = vector_harmonic.subs(north)
    assert value == sp.zeros(3, 1)
    assert tangent_1 == sp.Matrix([1, 0, 0])
    assert tangent_2 == sp.Matrix([0, 1, 0])
    selected_ambient_direction = tangent_1
    scalar_value = (selected_ambient_direction.T * value)[0]
    scalar_derivative = (selected_ambient_direction.T * tangent_1)[0]
    assert (scalar_value**2 - scalar_derivative**2) / 4 == -sp.Rational(1, 4)


def verify_full_one_one_block() -> None:
    """Check the signed-SVD moving plane for an arbitrary (1,1) block."""
    p, q, r, t, parameter = sp.symbols("p q r t s", real=True)
    tau_1, tau_2, tau_3 = sp.symbols("tau_1 tau_2 tau_3", real=True)
    first_sphere = sp.Matrix([sp.sqrt(1 - p**2 - q**2), p, q])
    second_sphere = sp.Matrix([sp.sqrt(1 - r**2 - t**2), r, t])
    first_jacobian = first_sphere.jacobian((p, q))
    second_jacobian = second_sphere.jacobian((r, t))
    first_metric = first_jacobian.T * first_jacobian
    second_metric = second_jacobian.T * second_jacobian
    first_rotated = sp.Matrix.hstack(
        *(first_sphere.cross(first_jacobian[:, index]) for index in range(2))
    )
    second_rotated = sp.Matrix.hstack(
        *(second_sphere.cross(second_jacobian[:, index]) for index in range(2))
    )

    # In the oriented frames (x,X,x cross X) and (y,Y,y cross Y), a
    # signed SVD can be arranged in this form.  The tau_1 entry supplies
    # X(alpha), while the -tau_3 entry supplies Y(beta).
    coupling_matrix = sp.Matrix(
        [
            [tau_2, 0, 0],
            [0, 0, tau_1],
            [0, -tau_3, 0],
        ]
    )
    coupling = first_rotated.T * coupling_matrix * second_rotated
    metric = first_metric.row_join(parameter * coupling).col_join(
        (parameter * coupling.T).row_join(second_metric)
    )
    first_vector = sp.Matrix([1, 0, 0, -parameter * tau_3])
    second_vector = sp.Matrix([0, parameter * tau_1, 1, 0])
    _, curvature = curvature_at_origin(
        (p, q, r, t), metric, first_vector, second_vector
    )
    coefficient = sp.factor(
        sp.diff(curvature, parameter, 2).subs(parameter, 0) / 2
    )
    assert sp.factor(coefficient + tau_1**2 + tau_3**2) == 0


def verify_cross_degree_frame_average() -> None:
    """Check the positive averaged defect for an exact degree-two/three sum."""
    y_0, y_1, y_2 = sp.symbols("y_0 y_1 y_2", real=True)
    amplitude_2, amplitude_3 = sp.symbols("amplitude_2 amplitude_3", real=True)
    variables = (y_0, y_1, y_2)
    harmonic_2 = y_0 * y_1
    harmonic_3 = y_0 * y_1 * y_2
    mixed_harmonic = amplitude_2 * harmonic_2 + amplitude_3 * harmonic_3
    gradient = tangential_gradient(mixed_harmonic, variables)
    norm = sphere_average(mixed_harmonic**2, variables)
    energy = sphere_average(gradient.dot(gradient), variables)
    assert sp.factor(norm - amplitude_2**2 / 15 - amplitude_3**2 / 105) == 0
    assert (
        sp.factor(energy - 6 * amplitude_2**2 / 15 - 12 * amplitude_3**2 / 105)
        == 0
    )
    averaged_defect = sp.factor((energy / 2 - norm) / 3)
    assert (
        sp.factor(averaged_defect - 2 * amplitude_2**2 / 45 - amplitude_3**2 / 63)
        == 0
    )

    # Degree-one (the matrix block) has zero cross average against the
    # higher modes, so it cannot cancel the strictly positive defect.
    linear_coefficients = sp.symbols("m_0 m_1 m_2", real=True)
    linear_harmonic = sum(
        coefficient * variable
        for coefficient, variable in zip(linear_coefficients, variables)
    )
    linear_gradient = tangential_gradient(linear_harmonic, variables)
    assert (
        sp.factor(sphere_average(linear_harmonic * mixed_harmonic, variables))
        == 0
    )
    assert (
        sp.factor(sphere_average(linear_gradient.dot(gradient), variables))
        == 0
    )
    spectral_cross = sphere_average(
        linear_gradient.dot(gradient) / 2 - linear_harmonic * mixed_harmonic,
        variables,
    )
    assert sp.factor(spectral_cross) == 0


def verify_spectral_gap() -> None:
    degree = sp.symbols("ell", integer=True, positive=True)
    eigenvalue = degree * (degree + 1)
    assert eigenvalue.subs(degree, 2) == 6
    for tested_degree in range(2, 12):
        assert eigenvalue.subs(degree, tested_degree) > 2


def main() -> None:
    verify_general_coupled_moving_plane_coefficient()
    verify_generic_moving_plane_coefficient()
    verify_nonseparated_degree_two_mode()
    verify_full_one_one_block()
    verify_cross_degree_frame_average()
    verify_spectral_gap()
    print("PASS coupled kernel: general two-parameter moving-plane coefficient")
    print("PASS one-sided towers: generic moving-plane coefficient")
    print("PASS one-sided towers: nonseparated degree-two mode")
    print("PASS coupled kernel: arbitrary (1,1) signed-SVD block")
    print("PASS coupled kernel: exact cross-degree average and tower orthogonality")
    print("PASS one-sided towers: spherical spectral gap lambda_l > 2")


if __name__ == "__main__":
    main()
