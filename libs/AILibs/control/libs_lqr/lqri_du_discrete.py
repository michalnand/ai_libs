import numpy
import scipy


'''
solve LQR controller with delta-u (du) formulation for discrete system
x(n+1) = A*x(n) + B*u(n)

this matches the C++ LQRI class from lqri.h

instead of computing u directly, the controller predicts du (delta u),
and then integrates : u(n) = u(n-1) + du(n)
with conditional antiwindup - clipping both du and u

augmented state : z = [x, u]
augmented input : v = du

augmented system :
z(n+1) = A_aug * z(n) + B_aug * v(n)

where:
A_aug = [[A, B],     B_aug = [[0],
         [0, I]]              [I]]

control law :
    error   = xr - x
    du      = K * error - Ku * u
    du      = clip(du, -du_max, du_max)
    u_new   = u + du
    u       = clip(u_new, -u_max, u_max)

Q matrix penalises state error, shape (n_states, n_states)
R matrix penalises du (control effort change), shape (n_inputs, n_inputs)
S matrix penalises u magnitude, shape (n_inputs, n_inputs), optional

A matrix, shape (n_states, n_states)
B matrix, shape (n_states, n_inputs)
'''
class LQRIDUDiscrete:

    def __init__(self, a, b, q, r, s = 0.0, u_max=10**10):
        self.Kx, self.Ku = self.solve(a, b, q, r, s)

        self.u_max  = u_max

    '''
    inputs:
        xr : required state, shape (n_states, 1)
        x  : system state,   shape (n_states, 1)
        u  : current control, shape (n_inputs, 1) - accumulated from previous steps

    returns:
        u_new : new control input into plant, shape (n_inputs, 1)
    '''
    def forward(self, xr, x, u):
        # error
        error = xr - x

        # LQR control law : du = K*error - Ku*u
        du = self.Kx @ error - self.Ku @ u

        # integrate     
        u_new = u + du

        # antiwindup : clip u
        u_new = numpy.clip(u_new, -self.u_max, self.u_max)

        return u_new



        

    '''
        solve the discrete time lqr controller for
        x(n+1) = A x(n) + B u(n)
        cost = sum x[n].T*Q*x[n] + u[n].T*R*u[n]
    '''
    def solve(self, a, b, q, r, s=0.0):
        n = a.shape[0]  # system order (n_states)
        m = b.shape[1]  # system inputs (n_inputs)

        # augmented system matrices
        # state z = [x, u], input v = du
        a_aug = numpy.block([
            [a,                   b              ],
            [numpy.zeros((m, n)), numpy.eye(m)   ]  
        ])

        b_aug = numpy.vstack([numpy.zeros((n, m)), numpy.eye(m)])


        # augmented cost matrix
        # penalise state error (Q on x part) and u magnitude (S on u part)
        q_aug = numpy.block([
            [q,                    numpy.zeros((n, m))],
            [numpy.zeros((m, n)),  s*numpy.eye(m)]
        ])
    

        # discrete-time algebraic Riccati equation solution
        p = scipy.linalg.solve_discrete_are(a_aug, b_aug, q_aug, r)

        # compute the LQR gain  
        K = numpy.linalg.inv(r + b_aug.T @ p @ b_aug) @ (b_aug.T @ p @ a_aug)

        Kx  = K[:, :n]
        Ku  = K[:, n:]

        return Kx, Ku
    