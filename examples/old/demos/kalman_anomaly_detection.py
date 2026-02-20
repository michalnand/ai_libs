import numpy
import matplotlib.pyplot as plt

import scipy
from scipy.stats import chi2


'''
    2nd order motor model
'''
class MotorModel:
    def __init__(self, r_noise, dt  = 0.001):

        # example of motor parameters

        R = 0.2       # Ohm
        L = 0.0005    # H
        J = 0.002     # kg·m^2
        b = 0.0004    # N·m·s/rad
        Kt = 0.1      # Nm/A
        Ke = 0.01     # Vs/rad

        self.set(R, L, J, b, Kt, Ke, r_noise, dt)

    def set(self, R, L, J, b, Kt, Ke, r_noise, dt):

        # Continuous-time matrices
        A_c = numpy.array([
            [-b / J,     Kt / J],
            [-Ke / L,   -R / L]
        ])

        B_c = numpy.array([
            [0],
            [1 / L]
        ])

        # Discretization
        I = numpy.eye(2)
        A_d = I + dt * A_c
        B_d = dt * B_c

        self.A = A_d
        self.B = B_d

        # matrix for disturbance
        E_c = numpy.array([[-1 / J], [0]])
        E_d = dt * E_c

        self.E = E_d

        self.r = r_noise

        self.x = numpy.zeros((2, 1))


    def step(self, u, d = None):

        self.x = self.A@self.x + self.B@u

        if d is not None:
            self.x+= self.E@d

        noise = self.r@numpy.random.randn(self.x.shape[0], self.x.shape[1])

        return self.x, self.x + noise

''''
    statedy state kalman filter for dynamical system model : 
    x(n+1) = Ax(n) + Bu(n)

    q_noise = model noise covariance
    r_noise = measurement (sensor) noise covariance
'''
class KalmanFilter:

    def __init__(self, a, b, q_noise, r_noise):

        self.a = a
        self.b = b
        
        self.h = numpy.eye(self.a.shape[0])

        self.k = self._solve_kalman_gain(self.a, self.b, self.h, q_noise, r_noise)

        self.x_hat = numpy.zeros((self.a.shape[0], 1))

      
    def step(self, y_obs, u):
        # steady state kalman filter equation
        self.x_hat = self.a@self.x_hat + self.b@u + self.k@(y_obs - self.h@self.x_hat)

        return self.x_hat
    

    '''
        compute kalman gain matrix F for observer : 
        x_hat(n+1) = Ax_hat(n) + Bu(n) + K(x(n) - x_hat(n))
    ''' 
    def _solve_kalman_gain(self, a, b, h, q, r):
        # Solve the discrete-time Algebraic Riccati Equation (DARE)
        p = scipy.linalg.solve_discrete_are(a.T, h.T, q, r)

        # Compute the steady-state Kalman gain
        k = p @ h.T @ numpy.linalg.inv(h @ p @ h.T + r)

        return k



class KalmanFilterAnomaly(KalmanFilter):
    def __init__(self, a, b, q_noise, r_noise, alpha = 0.99, k_lp = 0.99):
        KalmanFilter.__init__(self,  a, b, q_noise, r_noise)


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
        k = 0.99
        self.a_lp = k*self.a_lp + (1.0 - k)*(a > self.threshold)

        return self.x_hat, self.a_lp
    

    def _residual_score(self, y_obs, x_hat):
        r = y_obs - self.h @ x_hat

        distance = r.T @ self.r_inv @ r
        return float(distance)


if __name__ == "__main__":

    dt = 0.001

    # sensor noise
    r_noise = numpy.diag([5.0, 0.5])

    # model noise
    q = 10e-4
    q_noise = numpy.diag([q, q]) 



    real_system  = MotorModel(r_noise, dt)

    kalman_filter = KalmanFilterAnomaly(real_system.A, real_system.B, q_noise, r_noise)

    n_samples = 20000

    t_result  = []
    u_result  = []
    d_result  = []
    xr_result = []
    yr_result = []
    xm_result = []
    a_result  = []

    for n in range(n_samples):

        # square wave control input
        if (n//2000)%2 != 0:
            u_in = 1.0
        else:
            u_in = 0.0

        # add disturbance
        if n > n_samples/2:
            d_in = 0.3
        else:
            d_in = 0.0


        u_tmp = numpy.array([[u_in]])
        d_tmp = numpy.array([[d_in]])

        xr, yr = real_system.step(u_tmp, d_tmp)

        xm, a = kalman_filter.step(yr, u_tmp)


        t_result.append(n*dt)
        u_result.append(u_in)
        d_result.append(d_in)
        xr_result.append(xr[:, 0])
        yr_result.append(yr[:, 0])
        xm_result.append(xm[:, 0])
        a_result.append(a)

    t_result    = numpy.array(t_result)
    u_result    = numpy.array(u_result)
    d_result    = numpy.array(d_result)
    xr_result   = numpy.array(xr_result)
    yr_result   = numpy.array(yr_result)
    xm_result   = numpy.array(xm_result)
    a_result    = numpy.array(a_result)

    fig, ax = plt.subplots(4)

    ax[0].plot(t_result, u_result, c='blue', label="control")
    ax[0].plot(t_result, d_result, c='red', label="disturbance")
    ax[0].set_ylabel("input")
    ax[0].legend()

    
    ax[1].plot(t_result, yr_result[:, 0], c='green',  alpha=0.5, label="observed")
    ax[1].plot(t_result, xr_result[:, 0], c='blue', label="reference", lw=4)
    ax[1].plot(t_result, xm_result[:, 0], c='red', label="predicted")
    ax[1].set_ylabel("velocity [rad/s]")
    ax[1].legend()

    ax[2].plot(t_result, yr_result[:, 1], c='green', alpha=0.5, label="observed")
    ax[2].plot(t_result, xr_result[:, 1], c='blue', label="reference", lw=4)
    ax[2].plot(t_result, xm_result[:, 1], c='red', label="predicted")
    ax[2].set_ylabel("current [A]")
    ax[2].legend()


    ax[3].plot(t_result, a_result, c='blue')
    ax[3].set_ylabel("anomaly score")
    ax[3].set_xlabel("time [s]")
    

    plt.legend()
    plt.show()