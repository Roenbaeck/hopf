#!/usr/bin/env python3
"""Exact-derivative experiments for mixed curvature of Hessian sphere products.

This is deliberately independent of the legacy finite-difference code.  It uses
orthographic charts centred at the point under test and automatic
differentiation of the exact metric

    g = (1-gamma*s)(h_1+h_2) + 2*gamma <dx, A dy>.

Only floating-point evaluation remains; all coordinate derivatives entering
the Riemann tensor are obtained by automatic differentiation.
"""

from __future__ import annotations

import argparse

import autograd.numpy as np
import numpy as onp
from autograd import hessian, jacobian


def local_metric(q: np.ndarray, matrix: np.ndarray, gamma: float) -> np.ndarray:
    """Metric in orthographic charts centred at x=e_0 and y=e_0."""
    m = matrix.shape[0] - 1
    n = matrix.shape[1] - 1
    u = q[:m]
    v = q[m:]

    xu0 = np.sqrt(1.0 - np.dot(u, u))
    yv0 = np.sqrt(1.0 - np.dot(v, v))
    x = np.concatenate((np.array([xu0]), u))
    y = np.concatenate((np.array([yv0]), v))

    # Columns are the coordinate tangent vectors dx/du_i and dy/dv_alpha.
    jx = np.concatenate(((-u / xu0)[None, :], np.eye(m)), axis=0)
    jy = np.concatenate(((-v / yv0)[None, :], np.eye(n)), axis=0)
    h1 = np.dot(jx.T, jx)
    h2 = np.dot(jy.T, jy)
    coupling = np.dot(np.dot(jx.T, matrix), jy)
    seam = np.dot(x, np.dot(matrix, y))
    lam = 1.0 - gamma * seam

    top = np.concatenate((lam * h1, gamma * coupling), axis=1)
    bottom = np.concatenate((gamma * coupling.T, lam * h2), axis=1)
    return np.concatenate((top, bottom), axis=0)


def christoffel_upper(q: np.ndarray, matrix: np.ndarray, gamma: float) -> np.ndarray:
    metric_function = lambda z: local_metric(z, matrix, gamma)
    metric = metric_function(q)
    derivative = jacobian(metric_function)(q)  # dg[i,j,k] = partial_k g_ij
    first = 0.5 * (
        np.transpose(derivative, (0, 2, 1))
        + derivative
        - np.transpose(derivative, (2, 1, 0))
    )
    # The expression above has indices first[k,i,j].
    return np.einsum("km,mij->kij", np.linalg.inv(metric), first)


def riemann_lower_at_centre(matrix: onp.ndarray, gamma: float) -> onp.ndarray:
    """Return R_ijkl at the chart centre using exact coordinate derivatives."""
    matrix_ad = np.array(matrix)
    dimension = matrix.shape[0] + matrix.shape[1] - 2
    centre = np.zeros(dimension)
    metric_function = lambda z: local_metric(z, matrix_ad, gamma)
    metric = metric_function(centre)
    derivative = jacobian(metric_function)(centre)
    second = jacobian(jacobian(metric_function))(centre)

    # Gamma_{pij} (first kind).
    first_kind = 0.5 * (
        np.transpose(derivative, (0, 2, 1))
        + derivative
        - np.transpose(derivative, (2, 1, 0))
    )

    # This convention gives R_0101=+1 on the round unit sphere.
    linear = 0.5 * (
        np.transpose(second, (0, 2, 3, 1))
        + np.transpose(second, (2, 0, 1, 3))
        - np.transpose(second, (0, 2, 1, 3))
        - np.transpose(second, (2, 0, 3, 1))
    )
    inverse = np.linalg.inv(metric)
    quadratic = np.einsum("pq,pjk,qil->ijkl", inverse, first_kind, first_kind)
    quadratic -= np.einsum("pq,pjl,qik->ijkl", inverse, first_kind, first_kind)
    return onp.asarray(linear + quadratic, dtype=float)


def mixed_sectional_curvature(
    matrix: onp.ndarray, gamma: float, x_direction: onp.ndarray, y_direction: onp.ndarray
) -> tuple[float, float, float]:
    m = matrix.shape[0] - 1
    n = matrix.shape[1] - 1
    tangent = m + n
    x_vector = onp.concatenate((x_direction, onp.zeros(n)))
    y_vector = onp.concatenate((onp.zeros(m), y_direction))
    matrix_ad = np.array(matrix)
    centre = np.zeros(tangent)
    x_ad = np.array(x_vector)
    y_ad = np.array(y_vector)
    metric_function = lambda z: local_metric(z, matrix_ad, gamma)
    metric_ad = metric_function(centre)
    derivative = jacobian(metric_function)(centre)

    # Contract the exact lower-index curvature formula before taking second
    # derivatives.  This is substantially faster than constructing all d^2
    # metric and Riemann components.
    fxy = lambda z: np.dot(x_ad, np.dot(metric_function(z), y_ad))
    fxx = lambda z: np.dot(x_ad, np.dot(metric_function(z), x_ad))
    fyy = lambda z: np.dot(y_ad, np.dot(metric_function(z), y_ad))
    linear = np.dot(x_ad, np.dot(hessian(fxy)(centre), y_ad))
    linear -= 0.5 * np.dot(y_ad, np.dot(hessian(fxx)(centre), y_ad))
    linear -= 0.5 * np.dot(x_ad, np.dot(hessian(fyy)(centre), x_ad))

    first_kind = 0.5 * (
        np.transpose(derivative, (0, 2, 1))
        + derivative
        - np.transpose(derivative, (2, 1, 0))
    )
    gamma_xy = np.einsum("pij,i,j->p", first_kind, x_ad, y_ad)
    gamma_xx = np.einsum("pij,i,j->p", first_kind, x_ad, x_ad)
    gamma_yy = np.einsum("pij,i,j->p", first_kind, y_ad, y_ad)
    inverse = np.linalg.inv(metric_ad)
    quadratic = np.dot(gamma_xy, np.dot(inverse, gamma_xy))
    quadratic -= np.dot(gamma_yy, np.dot(inverse, gamma_xx))
    numerator = linear + quadratic

    metric = onp.asarray(metric_ad)
    gxx = x_vector @ metric @ x_vector
    gyy = y_vector @ metric @ y_vector
    gxy = x_vector @ metric @ y_vector
    denominator = gxx * gyy - gxy * gxy
    return float(numerator / denominator), float(numerator), float(denominator)


def normalized_matrix(rng: onp.random.Generator, m: int, n: int) -> onp.ndarray:
    matrix = rng.normal(size=(m + 1, n + 1))
    singular_values = onp.linalg.svd(matrix, compute_uv=False)
    scale = singular_values[0] + (singular_values[1] if len(singular_values) > 1 else 0.0)
    return matrix / scale


def random_unit(rng: onp.random.Generator, dimension: int) -> onp.ndarray:
    vector = rng.normal(size=dimension)
    return vector / onp.linalg.norm(vector)


def sanity_checks() -> None:
    # gamma=0 is the round product: a same-factor plane has K=1 and a
    # background-mixed plane has K=0.
    matrix = onp.array(
        [[0.2, -0.1, 0.3], [0.4, 0.7, -0.2], [0.1, 0.5, -0.6]]
    )
    curvature = riemann_lower_at_centre(matrix, 0.0)
    assert abs(curvature[0, 1, 0, 1] - 1.0) < 1e-10
    assert abs(curvature[0, 2, 0, 2]) < 1e-10

    # The sign of gamma is redundant: x -> -x sends g_gamma,A to g_-gamma,A.
    rng = onp.random.default_rng(11)
    direction_x = random_unit(rng, 2)
    direction_y = random_unit(rng, 2)
    positive = mixed_sectional_curvature(matrix, 0.2, direction_x, direction_y)[0]
    full = riemann_lower_at_centre(matrix, 0.2)
    xv = onp.concatenate((direction_x, onp.zeros(2)))
    yv = onp.concatenate((onp.zeros(2), direction_y))
    full_numerator = onp.einsum("ijkl,i,j,k,l", full, xv, yv, xv, yv)
    centre_metric = onp.asarray(local_metric(np.zeros(4), np.array(matrix), 0.2))
    full_denominator = (xv @ centre_metric @ xv) * (yv @ centre_metric @ yv)
    full_denominator -= (xv @ centre_metric @ yv) ** 2
    assert abs(positive - full_numerator / full_denominator) < 1e-9
    reflected = matrix.copy()
    reflected[0, :] *= -1.0
    reflected[1:, :] *= -1.0
    # At the new basepoint -x, the adapted tangent frame is also reflected.
    negative = mixed_sectional_curvature(reflected, -0.2, direction_x, direction_y)[0]
    assert abs(positive - negative) < 1e-9


def search(seed: int, trials: int, m: int, n: int) -> None:
    rng = onp.random.default_rng(seed)
    best = (onp.inf, None)
    for trial in range(trials):
        matrix = normalized_matrix(rng, m, n)
        gamma = rng.uniform(0.02, 0.999)
        x_direction = random_unit(rng, m)
        y_direction = random_unit(rng, n)
        value, numerator, denominator = mixed_sectional_curvature(
            matrix, gamma, x_direction, y_direction
        )
        if value < best[0]:
            best = (
                value,
                (matrix, gamma, x_direction, y_direction, numerator, denominator),
            )
            print(
                f"trial={trial:5d} K={value:+.12e} gamma={gamma:.8f} "
                f"numerator={numerator:+.12e} denominator={denominator:.12e}",
                flush=True,
            )
        if value < -1e-9:
            print("COUNTEREXAMPLE")
            onp.set_printoptions(precision=17, suppress=False)
            print("A =", repr(matrix))
            print("gamma =", repr(gamma))
            print("X =", repr(x_direction))
            print("Y =", repr(y_direction))
            return
    print(f"No negative value in {trials} trials; best K={best[0]:+.12e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args()
    sanity_checks()
    print("PASS exact-derivative sanity checks")
    search(args.seed, args.trials, args.m, args.n)


if __name__ == "__main__":
    main()
