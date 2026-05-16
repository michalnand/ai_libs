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
class LQRIDiscrete:

    def __init__(self, A, B, Q, R, q_scale = 1, antiwindup = 10**10):
        self.Kx, self.Ki = self.solve_lqr(A, B, Q, R, q_scale)
        self.antiwindup = antiwindup

    

    '''
    inputs:
        xr : required state, shape (n_states, 1)
        x  : system state, shape (n_states, 1)

        z  : u integral, shape (n_inputs, 1)

    returns:
        u : input into plant, shape (n_inputs, 1)
        z_new : input into plant, shape (n_inputs, 1)
    '''
    def forward(self, xr, x, z):
        #error action
        error = xr - x

        # integrate in u-space for simpler antiwindup
        z_new = z + self.Ki@error

        #LQR controll law
        u = -self.Kx@x + z_new

        u_res = numpy.clip(u, -self.antiwindup, self.antiwindup)

        # antiwindup
        z_new = z_new - (u - u_res)

        return u_res, z_new


    '''
    solve the discrete time lqr controller for
    x(n+1) = A x(n) + B u(n)
    cost = sum x[n].T*Q*x[n] + u[n].T*R*u[n]
    '''
    def solve_lqr(self, a, b, q, r, q_scale = 1):
        n = a.shape[0]  #system order
        m = b.shape[1]  #inputs count

        #matrix augmentation with integral action
        a_aug = numpy.zeros((n+n, n+n))
        b_aug = numpy.zeros((n+n, m))
        q_aug = numpy.zeros((n+n, n+n))


        a_aug[0:n, 0:n] = a 

        #add integrator into augmented a matrix
        for i in range(n):
            a_aug[i + n, i]     = 1.0
            a_aug[i + n, i + n] = 1.0


        b_aug[0:n,0:m]  = b

        #project Q matrix to output, and fill augmented q matrix
        q_aug[0:n, 0:n] = q
        q_aug[n:, n:]   = q_scale*q

        # discrete-time algebraic Riccati equation solution
        p = scipy.linalg.solve_discrete_are(a_aug, b_aug, q_aug, r)

        # compute the LQR gain
        #ki_tmp =  numpy.linalg.inv(r)@(b_aug.T@p)
        K = numpy.linalg.inv(r + b_aug.T @ p @ b_aug) @ (b_aug.T @ p @ a_aug)

        #truncated small elements
        K[numpy.abs(K) < 10**-10] = 0

        #split ki for k and integral action part ki
        Kx   = K[:, 0:a.shape[0]]
        Ki   = K[:, a.shape[0]:]

        return Kx, Ki
    
  