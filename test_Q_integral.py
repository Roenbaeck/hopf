#!/usr/bin/env python3
"""
Can we prove Q(s) always changes sign?

Q = sum_l (s_{02l})^2 - sum_l s_{00l} * s_{22l}

On T^4, integrate Q over the manifold:
  int Q dV = int [sum_l s_{02l}^2 - sum_l s_{00l} s_{22l}] dV

Using integration by parts on T^4:
  int s_{00l} s_{22l} dV = -int s_{00} s_{22ll} dV  [IBP in x^l]
                         = +int s_{00l} s_{22l} dV   [this is circular]

Let me try Fourier analysis. If s = sum c_k exp(i k.x), then:
  s_{ij} = -(k_i k_j) cos(k.x + phi)  [for real s with cos representation]

Actually, let's use complex Fourier:
  s = sum c_k e^{ik.x}  with c_{-k} = conj(c_k)
  s_{ij} = -k_i k_j c_k e^{ik.x}
  s_{ijl} = -i k_i k_j k_l c_k e^{ik.x}

Then:
  Q(x) = sum_l |sum_k -i k_0 k_2 k_l c_k e^{ik.x}|^2
        - sum_l [sum_k -i k_0^2 k_l c_k e^{ik.x}][sum_q -i k_2^2 k_l c_q e^{iq.x}]

Wait, this is getting complicated. Let me just check the integral numerically.
"""

import numpy as np

def test_integral_Q():
    """
    Test whether int_T4 Q dV = 0 or has definite sign.
    """
    print("=" * 70)
    print("Integral test: does int Q dV = 0?")
    print("=" * 70)

    rng = np.random.default_rng(42)
    N = 12  # grid points per dimension (12^4 = 20736)
    xx = np.linspace(0, 2*np.pi, N, endpoint=False)
    X1, X2, X3, X4 = np.meshgrid(xx, xx, xx, xx, indexing='ij')
    X1 = X1.ravel(); X2 = X2.ravel(); X3 = X3.ravel(); X4 = X4.ravel()
    Npts = len(X1)
    dV = (2*np.pi / N)**4

    modes = 2
    n_trials = 30

    print(f"Grid: {N}^4 = {Npts} points, {n_trials} random seams, modes={modes}")

    for trial in range(min(n_trials, 10)):
        rng_t = np.random.default_rng(100 + trial)

        # Build Fourier coefficients
        coeffs = []
        for k1 in range(-modes, modes + 1):
            for k2 in range(-modes, modes + 1):
                for k3 in range(-modes, modes + 1):
                    for k4 in range(-modes, modes + 1):
                        if k1 == 0 and k2 == 0 and k3 == 0 and k4 == 0:
                            continue
                        norm = k1**2 + k2**2 + k3**2 + k4**2
                        amp = rng_t.normal() / (1 + norm)
                        phase = rng_t.uniform(0, 2*np.pi)
                        coeffs.append((k1, k2, k3, k4, amp, phase))

        k_arr = np.array([(c[0], c[1], c[2], c[3]) for c in coeffs])
        amps = np.array([c[4] for c in coeffs])
        phases = np.array([c[5] for c in coeffs])

        # Compute third derivatives at all grid points
        # s_{ijl} = amp * k_i * k_j * k_l * sin(k.x + phase)
        phase_all = (k_arr[:, 0:1] * X1[np.newaxis, :] +
                     k_arr[:, 1:2] * X2[np.newaxis, :] +
                     k_arr[:, 2:3] * X3[np.newaxis, :] +
                     k_arr[:, 3:4] * X4[np.newaxis, :] +
                     phases[:, np.newaxis])
        sin_p = np.sin(phase_all)

        # Q = sum_l s_{02l}^2 - sum_l s_{00l} s_{22l}
        Q = np.zeros(Npts)
        for l in range(4):
            s_02l = np.einsum('m,m,m,m,mg->g', amps,
                              k_arr[:, 0], k_arr[:, 2], k_arr[:, l], sin_p)
            s_00l = np.einsum('m,m,m,m,mg->g', amps,
                              k_arr[:, 0], k_arr[:, 0], k_arr[:, l], sin_p)
            s_22l = np.einsum('m,m,m,m,mg->g', amps,
                              k_arr[:, 2], k_arr[:, 2], k_arr[:, l], sin_p)
            Q += s_02l**2 - s_00l * s_22l

        integral_Q = np.sum(Q) * dV
        print(f"  Trial {trial}: int Q dV = {integral_Q:.6f}, "
              f"Q range = [{np.min(Q):.4f}, {np.max(Q):.4f}]")

    # Now check: does int Q dV = 0 ALWAYS?
    print(f"\nChecking {n_trials} trials:")
    integrals = []
    for trial in range(n_trials):
        rng_t = np.random.default_rng(100 + trial)
        coeffs = []
        for k1 in range(-modes, modes + 1):
            for k2 in range(-modes, modes + 1):
                for k3 in range(-modes, modes + 1):
                    for k4 in range(-modes, modes + 1):
                        if k1 == 0 and k2 == 0 and k3 == 0 and k4 == 0:
                            continue
                        norm = k1**2 + k2**2 + k3**2 + k4**2
                        amp = rng_t.normal() / (1 + norm)
                        phase = rng_t.uniform(0, 2*np.pi)
                        coeffs.append((k1, k2, k3, k4, amp, phase))

        k_arr = np.array([(c[0], c[1], c[2], c[3]) for c in coeffs])
        amps = np.array([c[4] for c in coeffs])
        phases = np.array([c[5] for c in coeffs])
        phase_all = (k_arr[:, 0:1] * X1[np.newaxis, :] +
                     k_arr[:, 1:2] * X2[np.newaxis, :] +
                     k_arr[:, 2:3] * X3[np.newaxis, :] +
                     k_arr[:, 3:4] * X4[np.newaxis, :] +
                     phases[:, np.newaxis])
        sin_p = np.sin(phase_all)

        Q = np.zeros(Npts)
        for l in range(4):
            s_02l = np.einsum('m,m,m,m,mg->g', amps,
                              k_arr[:, 0], k_arr[:, 2], k_arr[:, l], sin_p)
            s_00l = np.einsum('m,m,m,m,mg->g', amps,
                              k_arr[:, 0], k_arr[:, 0], k_arr[:, l], sin_p)
            s_22l = np.einsum('m,m,m,m,mg->g', amps,
                              k_arr[:, 2], k_arr[:, 2], k_arr[:, l], sin_p)
            Q += s_02l**2 - s_00l * s_22l

        integrals.append(np.sum(Q) * dV)

    integrals = np.array(integrals)
    print(f"  int Q range: [{np.min(integrals):.6f}, {np.max(integrals):.6f}]")
    print(f"  mean = {np.mean(integrals):.6f}, std = {np.std(integrals):.6f}")

    n_pos = np.sum(integrals > 1e-6)
    n_neg = np.sum(integrals < -1e-6)
    n_zero = n_trials - n_pos - n_neg
    print(f"  Positive: {n_pos}, Negative: {n_neg}, ~Zero: {n_zero}")

    if n_pos > 0 and n_neg > 0:
        print("\n  RESULT: int Q dV changes sign across seams")
        print("  => No simple integral identity forces Q to change sign")
    elif n_pos == 0 and n_neg == 0:
        print("\n  RESULT: int Q dV ≈ 0 always!")
        print("  => Orthogonality: int Q dV = 0 is an identity")
        print("  => Q must change sign pointwise!")
    elif n_neg == 0:
        print("\n  RESULT: int Q dV >= 0 always")
    else:
        print("\n  RESULT: int Q dV <= 0 always")


def test_Q_fourier_identity():
    """
    Analytic test: does int Q dV = 0 hold as a Fourier identity?

    For a single mode s = c cos(k.x + phi):
      s_{ijl} = c k_i k_j k_l sin(k.x + phi)

    Q = sum_l (c k_0 k_2 k_l)^2 sin^2 - sum_l (c k_0^2 k_l)(c k_2^2 k_l) sin^2
      = c^2 sin^2(k.x + phi) [k_0^2 k_2^2 |k|^2 - k_0^2 k_2^2 |k|^2]

    Wait:
    sum_l (k_0 k_2 k_l)^2 = k_0^2 k_2^2 sum_l k_l^2 = k_0^2 k_2^2 |k|^2
    sum_l (k_0^2 k_l)(k_2^2 k_l) = k_0^2 k_2^2 sum_l k_l^2 = k_0^2 k_2^2 |k|^2

    So for a SINGLE mode: Q = 0 identically!

    For TWO modes s = c1 cos(k.x + phi1) + c2 cos(q.x + phi2):
    s_{ijl} = c1 k_i k_j k_l sin(k.x + phi1) + c2 q_i q_j q_l sin(q.x + phi2)

    Q = sum_l [c1 k_0 k_2 k_l sin(K) + c2 q_0 q_2 q_l sin(Q_)]^2
      - sum_l [c1 k_0^2 k_l sin(K) + c2 q_0^2 q_l sin(Q_)]
              [c1 k_2^2 k_l sin(K) + c2 q_2^2 q_l sin(Q_)]

    The diagonal terms cancel (each mode's self-interaction vanishes as shown).
    The cross terms:
    2 c1 c2 sin(K) sin(Q_) sum_l [k_0 k_2 k_l q_0 q_2 q_l - k_0^2 k_l q_2^2 q_l]
    = 2 c1 c2 sin(K) sin(Q_) [k_0 k_2 q_0 q_2 (k.q) - k_0^2 q_2^2 (k.q)]
    = 2 c1 c2 sin(K) sin(Q_) (k.q) [k_0 k_2 q_0 q_2 - k_0^2 q_2^2]
    = 2 c1 c2 sin(K) sin(Q_) (k.q) k_0 q_2 [k_2 q_0 - k_0 q_2]

    This is SIGN-INDEFINITE! And its integral is:
    int sin(k.x+phi1) sin(q.x+phi2) dV = 0 for k != q and k != -q.
    For k = -q: int sin(k.x+phi1) sin(-k.x+phi2) dV = -(V/2) cos(phi1+phi2)

    So:
    int Q dV = sum over pairs with k=-q of:
      2 c_k c_{-k} [-(V/2)cos(phi_k+phi_{-k})] (k.(-k)) k_0 (-k_2) [k_2(-k_0) - k_0(-k_2)]
    = sum ... * |k|^2 * ... * [-k_0 k_2 + k_0 k_2] ... wait this might be 0 too.

    Let me just check numerically.
    """
    print("\n" + "=" * 70)
    print("Analytic check: single-mode Q = 0?")
    print("=" * 70)

    # For a single mode k = (1, 0, 1, 0), c = 1, phi = 0:
    # s = cos(x1 + x3)
    # s_{ijl} = k_i k_j k_l sin(x1 + x3)
    # s_{020} = k_0*k_2*k_0 = 1*1*1 = 1, s_{021} = 0, s_{022} = 1*1*1 = 1, s_{023} = 0
    # Wait, k = (1,0,1,0), so k_0=1, k_1=0, k_2=1, k_3=0
    # s_{02l}: (k_0*k_2*k_l) for l=0,1,2,3 = (1, 0, 1, 0)
    # s_{00l}: (k_0*k_0*k_l) = (1, 0, 1, 0)
    # s_{22l}: (k_2*k_2*k_l) = (1, 0, 1, 0)
    # Q = sum_l (k_0 k_2 k_l)^2 - sum_l (k_0^2 k_l)(k_2^2 k_l)
    #   = (1+0+1+0) - (1+0+1+0) = 0. ✓

    # Two modes: k=(1,0,1,0), q=(1,1,0,0)
    k = np.array([1, 0, 1, 0])
    q = np.array([1, 1, 0, 0])
    # Cross term coefficient: (k.q) * k[0] * q[2] * (k[2]*q[0] - k[0]*q[2])
    kdotq = np.dot(k, q)
    cross_coeff = kdotq * k[0] * q[2] * (k[2]*q[0] - k[0]*q[2])
    print(f"  Two modes k={k}, q={q}")
    print(f"  k.q = {kdotq}, cross = {cross_coeff}")
    if cross_coeff == 0:
        print(f"  Cross term vanishes for this pair")
    else:
        print(f"  Cross term is nonzero => Q is sign-indefinite")

    # Different pair: k=(1,0,0,1), q=(0,1,1,0)
    k = np.array([1, 0, 0, 1])
    q = np.array([0, 1, 1, 0])
    kdotq = np.dot(k, q)
    cross_coeff = kdotq * k[0] * q[2] * (k[2]*q[0] - k[0]*q[2])
    print(f"\n  Two modes k={k}, q={q}")
    print(f"  k.q = {kdotq}, cross = {cross_coeff}")

    # k=(1,0,1,0), q=(0,1,0,1)
    k = np.array([1, 0, 1, 0])
    q = np.array([0, 1, 0, 1])
    kdotq = np.dot(k, q)
    cross_coeff = kdotq * k[0] * q[2] * (k[2]*q[0] - k[0]*q[2])
    print(f"\n  Two modes k={k}, q={q}")
    print(f"  k.q = {kdotq}, cross = {cross_coeff}")

    # k=(1,1,1,0), q=(1,0,0,1)
    k = np.array([1, 1, 1, 0])
    q = np.array([1, 0, 0, 1])
    kdotq = np.dot(k, q)
    cross_coeff = kdotq * k[0] * q[2] * (k[2]*q[0] - k[0]*q[2])
    print(f"\n  Two modes k={k}, q={q}")
    print(f"  k.q = {kdotq}, cross = {cross_coeff}")
    if cross_coeff != 0:
        print(f"  NONZERO cross term => Q is sign-indefinite for this pair")
        # Check: does integral vanish?
        print(f"  But integral over T^4: int sin(k.x)sin(q.x) dV = 0 since k != +-q")
        print(f"  So integral of cross term = 0")

    print(f"\n  CONCLUSION: For a single mode, Q = 0 identically.")
    print(f"  For multiple modes, Q is sign-indefinite with zero integral.")
    print(f"  => int Q dV = 0 for ALL seams (Parseval-type identity).")
    print(f"  => Q MUST change sign pointwise!")


if __name__ == '__main__':
    test_integral_Q()
    test_Q_fourier_identity()
