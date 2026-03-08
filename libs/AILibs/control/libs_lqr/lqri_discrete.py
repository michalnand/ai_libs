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
e_sum(n)= e_sum(n-1) + xr(n) - x(n)
u(n)    = -K*x(n) + Ki*e_sum(n)

'''  
class LQRIDiscrete: 

    def __init__(self, a, b, q, r, antiwindup = 10**10, aw_enabled = True):
        self.k, self.ki = self.solve(a, b, q, r)

        self.antiwindup = antiwindup
        self.aw_enabled = aw_enabled


    '''
    inputs:
        xr : required state, shape (n_states, 1)
        x  : system state, shape (n_states, 1)
        integral_action : storage for integral action, shape (n_inputs, 1)

    returns:
        u : input into plant, shape (n_inputs, 1)
        integral_action_new : new IA, shape (n_inputs, 1)
    '''
    def forward(self, xr, x, integral_action):
        # integral action
        error = xr - x

        integral_action_new = integral_action + self.ki@error

        #LQR controll law
        u_new = -self.k@x + integral_action
        
        #conditional antiwindup
        u = numpy.clip(u_new, -self.antiwindup, self.antiwindup)

        if self.aw_enabled:
            integral_action_new = integral_action_new - (u_new - u)
        

        return u, integral_action_new


    '''
    solve the discrete time lqr controller for
    x(n+1) = A x(n) + B u(n)
    cost = sum x[n].T*Q*x[n] + u[n].T*R*u[n]
    '''
    def solve(self, a, b, q, r):
        n = a.shape[0]  # system order
        m = b.shape[1]  # system inputs

        # augmented system
        a_aug = numpy.block([
            [a, numpy.zeros((n, n))],
            [numpy.eye(n), numpy.eye(n)]
        ])
        
        b_aug = numpy.vstack([b, numpy.zeros((n, m))])
        
        # augmented cost
        q_aug = numpy.block([
            [numpy.zeros((n, n)), numpy.zeros((n, n))],   
            [numpy.zeros((n, n)), q]
        ])

        p = scipy.linalg.solve_discrete_are(a_aug, b_aug, q_aug, r)

        k_aug = numpy.linalg.inv(r) @ (b_aug.T @ p)

        #truncated small elements
        k_aug[numpy.abs(k_aug) < 10**-10] = 0
        
        
        k  = k_aug[:, :n]
        ki = k_aug[:, n:]

        return k, ki
