import numpy

from tests.conftest import rng


def test_data_linear_dynamics(n_samples, n_states, rng, dt, x_initial = None):
    dx_result = []

    if x_initial is None:
        x = rng.standard_normal((n_states, 1))
    else:        
        x = numpy.array(x_initial).reshape((n_states, 1))   

    A = rng.standard_normal((n_states, n_states))

    for _ in range(n_samples):
        dx = A@x
        x+= dx*dt   

        dx_result.append(dx[:, 0].copy())

    return numpy.array(dx_result)


def test_data_lorenz_attractor(n_samples, dt, x_initial=None, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Generate time series data from the Lorenz attractor (nonlinear dynamical system).

    Parameters:
        n_samples   : number of time steps to simulate.
        dt          : integration time step.
        x_initial   : initial state [x, y, z]. If None, a random initial condition is used.
        sigma       : Lorenz parameter sigma (Prandtl number), default 10.0.
        rho         : Lorenz parameter rho (Rayleigh number), default 28.0.
        beta        : Lorenz parameter beta, default 8/3.

    Returns:
        numpy array of shape (n_samples, 3) containing the [x, y, z] trajectory.
    """
   
    state = numpy.array(x_initial, dtype=numpy.float64)

    trajectory = []

    for _ in range(n_samples):
        x, y, z = state

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        state = state + numpy.array([dx, dy, dz]) * dt

        trajectory.append(state.copy())

    return numpy.array(trajectory)

