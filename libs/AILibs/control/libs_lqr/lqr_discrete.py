import numpy
import scipy


'''
solve LQR controller for discrete discrete system
x(n+1) = Ax(n) + Bu(n)

Q, R are weight matrices in quadratic loss

A matrix, shape (n_states, n_states)
B matrix, shape (n_states, n_inputs)

Q matrix, shape (n_states, n_states)
R matrix, shape (n_inputs, n_inputs)

control law : 
e_sum(n)= xr(n) - x(n)
u(n)    = K*e(n)

'''  
class LQRDiscrete:

    def __init__(self, a, b, q, r, antiwindup = 10**10):
        self.k = self.solve(a, b, q, r)

        self.antiwindup = antiwindup



    '''
    inputs:
        xr : required state, shape (n_states, 1)
        x  : system state, shape (n_states, 1)

    returns:
        u : input into plant, shape (n_inputs, 1)
    '''
    def forward(self, xr, x):
        #error action
        error = xr - x

        #LQR controll law
        u = self.k@error

        u = numpy.clip(u, -self.antiwindup, self.antiwindup)

        return u


    '''
        solve the discrete time lqr controller for
        x(n+1) = A x(n) + B u(n)
        cost = sum x[n].T*Q*x[n] + u[n].T*R*u[n]
    '''
    def solve(self, a, b, q, r):
        # discrete-time algebraic Riccati equation solution
        p = scipy.linalg.solve_discrete_are(a, b, q, r)

        # compute the LQR gain
        k = numpy.linalg.inv(r + b.T @ p @ b) @ (b.T @ p @ a)

        return k