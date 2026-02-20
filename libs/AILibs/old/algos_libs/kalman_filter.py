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
        # steady state kalman filter
        self.x_hat = self.a@self.x_hat + self.b@u + self.k@(y_obs - self.h@self.x_hat)

        return self.x_hat

    # compute kalman gain matrix K 
    def _solve_kalman_gain(self, a, h, q, r):
        # solve the discrete-time Algebraic Riccati Equation (DARE)
        p = scipy.linalg.solve_discrete_are(a.T, h.T, q, r)

        # compute the steady-state Kalman gain
        k = p @ h.T @ numpy.linalg.inv(h @ p @ h.T + r)

        return k
    





'''
    kalman filter extension for anomaly detection
    method step returns estimated state x_hat, and anomaly score
    anomaly score is in ranke 0 .. 1

    aditional parameters : 
    alpha : confidence interval
    k_lp  : anomaly score is filtered, this is low pass filter coeff
'''
class KalmanFilterAnomaly(KalmanFilter):
    def __init__(self, a, b, h, q_noise, r_noise, alpha = 0.99, k_lp = 0.99):
        KalmanFilter.__init__(self,  a, b, h, q_noise, r_noise)

        # variables for anomaly detection
        self.r_inv = numpy.linalg.inv(r_noise) 

        dof = self.r_inv.shape[0]
        self.threshold = chi2.ppf(alpha, dof)

        self.k_lp  = k_lp
        self.a_lp  = 0.0

    def step(self, y_obs, u):
        x_hat = super().step(y_obs, u)
        
        # measure anomalies
        a = self._residual_score(y_obs, x_hat)

        # low pass filter for anomaly score
        self.a_lp = self.k_lp*self.a_lp + (1.0 - self.k_lp)*(a > self.threshold)

        return self.x_hat, self.a_lp
    

    def _residual_score(self, y_obs, x_hat):
        r = y_obs - self.h @ x_hat

        distance = r.T @ self.r_inv @ r
        return float(distance)