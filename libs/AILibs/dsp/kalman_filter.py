import numpy
import scipy
from scipy.stats import chi2

'''
    statedy state kalman filter for discrete dynamical system model : 
    x(n+1) = Ax(n) + Bu(n)
    y(n)   = Hx(n)

    x       : system state, shape (n_states, 1)
    u       : control input, shape (n_inputs, 1)
    y       : system output (observed), shape (n_outputs, 1)

    A       : system dynamics matrix, shape (n_states, n_states)
    B       : system input matrix, shape (n_states, n_inputs)
    H       : output matrix, shape (n_outputs, n_states), for fully observed system diagonal matrix (n_states, n_states)

    q_noise : model noise covariance matrix, shape (n_states, n_states)
    r_noise : measurement (sensor) noise covariance, shape (n_outputs, n_outputs)
'''
class KalmanFilter:

    def __init__(self, a, b, h, q_noise, r_noise):
        self.a = a
        self.b = b
        self.h = h

        self.k = self._solve_kalman_gain(self.a, self.h, q_noise, r_noise)
        
        self.reset()

    '''
        reset filter state to zero
    '''
    def reset(self):
        self.x_hat = numpy.zeros((self.a.shape[0], 1))


    '''
        filter output step
        y_obs : observed system output
        u     : system control input
    '''
    def step(self, y_obs, u):
        # correction (filtered estimate at time n)
        x_filtered = self.x_hat + self.k @ (y_obs - self.h @ self.x_hat)

        # prediction (for next step)
        self.x_hat = self.a @ x_filtered + self.b @ u
        
        return x_filtered

    # compute kalman gain matrix K 
    def _solve_kalman_gain(self, a, h, q, r):
        # solve the discrete-time Algebraic Riccati Equation (DARE)
        p = scipy.linalg.solve_discrete_are(a.T, h.T, q, r)

        # compute the steady-state Kalman gain
        k = p @ h.T @ numpy.linalg.inv(h @ p @ h.T + r)

        return k
    
