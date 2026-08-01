"""
Fixtures for linear_regression tests.
"""
import pytest
import numpy


@pytest.fixture
def linear_system(rng):
    """
    Random linear system  y = x @ A.
    Returns (x, y, A_target).
    """
    n_inputs = 5
    n_outputs = 3
    n_samples = 200

    x = rng.standard_normal((n_samples, n_inputs))
    A = rng.standard_normal((n_inputs, n_outputs))
    y = x @ A

    return x, y, A


@pytest.fixture
def state_space_system(rng):
    """
    Random state-space system  x_{n+1} = A x_n + B u_n  (no noise).
    Returns (x_seq, u_seq, A_true, B_true).
    """
    n_states = 3
    n_inputs = 2
    n_steps = 500

    A_true = rng.standard_normal((n_states, n_states)) * 0.5
    B_true = rng.standard_normal((n_states, n_inputs)) * 0.5

    x = numpy.zeros((n_steps, n_states))
    u = rng.standard_normal((n_steps, n_inputs))
    x[0] = rng.standard_normal(n_states)

    for k in range(n_steps - 1):
        x[k + 1] = A_true @ x[k] + B_true @ u[k]

    return x, u, A_true, B_true
