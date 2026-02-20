"""
    Tests for AILibs.dsp.KalmanFilter

    Steady-state Kalman filter for discrete linear dynamical systems:
        x(n+1) = A x(n) + B u(n)
        y(n)   = H x(n) + noise
"""
import pytest
import numpy

import AILibs


@pytest.mark.dsp
class TestKalmanFilter:

    # ------------------------------------------------------------------
    # 1. Noiseless fully-observed system — filter output must match the
    #    true state almost exactly.
    # ------------------------------------------------------------------
    def test_noiseless_fully_observed(self, rng):
        n_states = 3
        n_inputs = 2
        n_steps  = 500

        # stable dynamics (scale down to keep eigenvalues inside unit circle)
        A = rng.standard_normal((n_states, n_states)) * 0.3
        B = rng.standard_normal((n_states, n_inputs)) * 0.5
        H = numpy.eye(n_states)                          # fully observed

        q_noise = numpy.eye(n_states) * 1e-6
        r_noise = numpy.eye(n_states) * 1e-6

        kf = AILibs.dsp.KalmanFilter(A, B, H, q_noise, r_noise)

        x = rng.standard_normal((n_states, 1)) * 0.1

        errors = []
        for _ in range(n_steps):
            u = rng.standard_normal((n_inputs, 1)) * 0.1
            y_obs = H @ x                               # perfect observation

            x_est = kf.step(y_obs, u)
            errors.append(numpy.linalg.norm(x_est - x))

            x = A @ x + B @ u                           # true dynamics

        errors = numpy.array(errors)

        # after initial transient the estimate must converge
        steady_errors = errors[n_steps // 2:]
        print(f"mean steady-state error: {steady_errors.mean():.6e}")
        assert steady_errors.mean() < 1e-3


    # ------------------------------------------------------------------
    # 2. Noisy measurements — filter should still track the true state
    #    better than raw observations.
    # ------------------------------------------------------------------
    def test_noisy_measurements(self, rng):
        n_states  = 4
        n_inputs  = 1
        n_outputs = 4
        n_steps   = 1000

        A = rng.standard_normal((n_states, n_states)) * 0.4
        B = rng.standard_normal((n_states, n_inputs)) * 0.5
        H = numpy.eye(n_outputs, n_states)

        sensor_std = 0.5
        q_noise = numpy.eye(n_states) * 1e-4
        r_noise = numpy.eye(n_outputs) * sensor_std**2

        kf = AILibs.dsp.KalmanFilter(A, B, H, q_noise, r_noise)

        x = numpy.zeros((n_states, 1))

        filter_errors = []
        obs_errors    = []

        for _ in range(n_steps):
            u = rng.standard_normal((n_inputs, 1)) * 0.2
            noise = rng.standard_normal((n_outputs, 1)) * sensor_std
            y_obs = H @ x + noise

            x_est = kf.step(y_obs, u)

            filter_errors.append(numpy.linalg.norm(x_est - x))
            obs_errors.append(numpy.linalg.norm(y_obs - H @ x))

            x = A @ x + B @ u

        # discard transient
        k = n_steps // 4
        mean_filter_err = numpy.mean(filter_errors[k:])
        mean_obs_err    = numpy.mean(obs_errors[k:])

        print(f"mean observation error: {mean_obs_err:.4f}")
        print(f"mean filter error:      {mean_filter_err:.4f}")

        assert mean_filter_err < mean_obs_err, (
            "filter should reduce estimation error compared to raw observations"
        )


    # ------------------------------------------------------------------
    # 3. Partial observation — only some states are observed,
    #    filter should still estimate the full state.
    # ------------------------------------------------------------------
    def test_partial_observation(self, rng):
        n_states  = 4
        n_inputs  = 1
        n_outputs = 2
        n_steps   = 1000

        A = rng.standard_normal((n_states, n_states)) * 0.3
        B = rng.standard_normal((n_states, n_inputs)) * 0.3

        # observe only first two states
        H = numpy.zeros((n_outputs, n_states))
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        sensor_std = 0.1
        q_noise = numpy.eye(n_states) * 1e-4
        r_noise = numpy.eye(n_outputs) * sensor_std**2

        kf = AILibs.dsp.KalmanFilter(A, B, H, q_noise, r_noise)

        x = numpy.zeros((n_states, 1))

        errors = []
        for _ in range(n_steps):
            u = rng.standard_normal((n_inputs, 1)) * 0.1
            y_obs = H @ x + rng.standard_normal((n_outputs, 1)) * sensor_std

            x_est = kf.step(y_obs, u)
            errors.append(numpy.linalg.norm(x_est - x))

            x = A @ x + B @ u

        steady_errors = numpy.array(errors[n_steps // 2:])
        print(f"mean partial-obs error: {steady_errors.mean():.4f}")

        # filter should still give reasonable estimates
        assert steady_errors.mean() < 1.0


    # ------------------------------------------------------------------
    # 4. Reset — after reset the filter state must be zeroed out.
    # ------------------------------------------------------------------
    def test_reset(self, rng):
        n_states = 3
        n_inputs = 2

        A = numpy.eye(n_states) * 0.9
        B = numpy.zeros((n_states, n_inputs))
        H = numpy.eye(n_states)

        q_noise = numpy.eye(n_states) * 1e-3
        r_noise = numpy.eye(n_states) * 1e-3

        kf = AILibs.dsp.KalmanFilter(A, B, H, q_noise, r_noise)

        # run a few steps so internal state is non-zero
        for _ in range(10):
            y = rng.standard_normal((n_states, 1))
            u = numpy.zeros((n_inputs, 1))
            kf.step(y, u)

        assert numpy.linalg.norm(kf.x_hat) > 0, "state should be non-zero before reset"

        kf.reset()
        assert numpy.allclose(kf.x_hat, 0.0), "state should be zero after reset"


    # ------------------------------------------------------------------
    # 5. Shape consistency — output shape must match the state dimension.
    # ------------------------------------------------------------------
    def test_output_shapes(self, rng):
        n_states  = 6
        n_inputs  = 3
        n_outputs = 4

        A = rng.standard_normal((n_states, n_states)) * 0.3
        B = rng.standard_normal((n_states, n_inputs))
        H = rng.standard_normal((n_outputs, n_states))

        q_noise = numpy.eye(n_states)
        r_noise = numpy.eye(n_outputs)

        kf = AILibs.dsp.KalmanFilter(A, B, H, q_noise, r_noise)

        y_obs = rng.standard_normal((n_outputs, 1))
        u     = rng.standard_normal((n_inputs, 1))

        x_est = kf.step(y_obs, u)

        assert x_est.shape == (n_states, 1), (
            f"expected shape ({n_states}, 1), got {x_est.shape}"
        )
        assert kf.k.shape == (n_states, n_outputs), (
            f"Kalman gain shape should be ({n_states}, {n_outputs}), "
            f"got {kf.k.shape}"
        )


    # ------------------------------------------------------------------
    # 6. Zero-input stable system — state should decay to zero.
    # ------------------------------------------------------------------
    def test_zero_input_convergence(self, rng):
        n_states = 3
        n_inputs = 1
        n_steps  = 500

        # stable diagonal system (eigenvalues < 1)
        A = numpy.diag([0.8, 0.7, 0.6])
        B = numpy.zeros((n_states, n_inputs))
        H = numpy.eye(n_states)

        q_noise = numpy.eye(n_states) * 1e-4
        r_noise = numpy.eye(n_states) * 1e-2

        kf = AILibs.dsp.KalmanFilter(A, B, H, q_noise, r_noise)

        # start from a known non-zero state
        x = numpy.array([[5.0], [3.0], [-2.0]])

        for _ in range(n_steps):
            u = numpy.zeros((n_inputs, 1))
            y_obs = H @ x + rng.standard_normal((n_states, 1)) * 0.1

            x_est = kf.step(y_obs, u)
            x = A @ x  # true state decays

        # both true state and estimate should be near zero
        assert numpy.linalg.norm(x) < 1e-3, "true state should have decayed"
        assert numpy.linalg.norm(x_est) < 1e-1, "estimate should track the decayed state"


    # ------------------------------------------------------------------
    # 7. Kalman gain sanity — with very low measurement noise the gain
    #    should be close to H⁻¹ (trust measurements fully).
    # ------------------------------------------------------------------
    def test_kalman_gain_low_measurement_noise(self):
        n_states = 3

        A = numpy.eye(n_states) * 0.9
        B = numpy.zeros((n_states, 1))
        H = numpy.eye(n_states)

        q_noise = numpy.eye(n_states) * 1.0
        r_noise = numpy.eye(n_states) * 1e-10   # almost no sensor noise

        kf = AILibs.dsp.KalmanFilter(A, B, H, q_noise, r_noise)

        # K ≈ I  (fully trust the measurement)
        print(f"Kalman gain:\n{kf.k}")
        assert numpy.allclose(kf.k, numpy.eye(n_states), atol=1e-3), (
            "with negligible sensor noise the Kalman gain should be ≈ I"
        )


    # ------------------------------------------------------------------
    # 8. Kalman gain sanity — with very high measurement noise the gain
    #    should be close to zero (ignore measurements).
    # ------------------------------------------------------------------
    def test_kalman_gain_high_measurement_noise(self):
        n_states = 3

        A = numpy.eye(n_states) * 0.5
        B = numpy.zeros((n_states, 1))
        H = numpy.eye(n_states)

        q_noise = numpy.eye(n_states) * 1e-10
        r_noise = numpy.eye(n_states) * 1e6    # huge sensor noise

        kf = AILibs.dsp.KalmanFilter(A, B, H, q_noise, r_noise)

        # K ≈ 0  (ignore the measurements)
        print(f"Kalman gain:\n{kf.k}")
        assert numpy.allclose(kf.k, 0.0, atol=1e-3), (
            "with huge sensor noise the Kalman gain should be ≈ 0"
        )
