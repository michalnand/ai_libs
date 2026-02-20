# DSP Tests

Tests for `AILibs.dsp` — **KalmanFilter** (steady-state Kalman filter for discrete linear dynamical systems).

The filter implements the standard predict–update loop for systems of the form:

$$x(n+1) = A\,x(n) + B\,u(n)$$
$$y(n) = H\,x(n) + \text{noise}$$

All datasets are synthetic with a seeded RNG (`seed=42`) for full reproducibility.

```bash
# run all dsp tests
pytest tests/dsp/ -v -s

# by marker
pytest -m dsp

# single class
pytest tests/dsp/test_kalman_filter.py::TestKalmanFilter
```

---

## TestKalmanFilter

Each test constructs a linear dynamical system with known matrices (A, B, H),
runs the Kalman filter over a sequence of observations, and checks estimation
accuracy, gain properties, or API behaviour.

| # | Test | Scenario | What it checks |
|---|------|----------|----------------|
| 1 | `test_noiseless_fully_observed` | 3 states, 2 inputs, H = I, near-zero process & sensor noise, 500 steps | Perfect-observation baseline: steady-state estimation error < 1e-3 |
| 2 | `test_noisy_measurements` | 4 states, 1 input, fully observed, sensor σ = 0.5, 1000 steps | Filter reduces error vs raw observations (mean filter error < mean observation error) |
| 3 | `test_partial_observation` | 4 states, 2 observed, sensor σ = 0.1, 1000 steps | Partial observability: filter still estimates full state (steady error < 1.0) |
| 4 | `test_reset` | 3 states, diagonal A = 0.9 I, 10 warm-up steps then `kf.reset()` | `reset()` zeroes internal state (`x_hat ≈ 0` after call) |
| 5 | `test_output_shapes` | 6 states, 3 inputs, 4 outputs, random matrices | Shape consistency: `x_est` is (n_states, 1), Kalman gain `K` is (n_states, n_outputs) |
| 6 | `test_zero_input_convergence` | Stable diagonal A = diag(0.8, 0.7, 0.6), zero input, 500 steps | State decays to zero; filter tracks the decay (‖x_est‖ < 0.1) |
| 7 | `test_kalman_gain_low_measurement_noise` | A = 0.9 I, R = 1e-10 I (negligible sensor noise) | Kalman gain K ≈ I — filter fully trusts measurements |
| 8 | `test_kalman_gain_high_measurement_noise` | A = 0.5 I, R = 1e6 I (huge sensor noise) | Kalman gain K ≈ 0 — filter ignores measurements |
