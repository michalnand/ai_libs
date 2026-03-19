import numpy
import scipy

class KoopmanLQR:
    def __init__(self, W, B, Q, R, lift_order):
        
        self.num_states = W.shape[0]//lift_order
        self.lift_order = lift_order

        b_lifted = numpy.zeros((self.num_states*self.lift_order, B.shape[1]))
        b_lifted[0:B.shape[0], :] = B

        q_lifted = numpy.zeros((self.num_states*self.lift_order, self.num_states*self.lift_order))
        q_lifted[0:Q.shape[0], 0:Q.shape[0]] = Q

        #r_lifted = numpy.zeros((self.num_states*self.lift_order, self.num_states*self.lift_order))
        #r_lifted[0:R.shape[0], 0:R.shape[0]] = R

        print(W.shape, b_lifted.shape, q_lifted.shape, R.shape)

        self.k = self.solve(W, b_lifted, Q, R)

        print(self.k)


    def reset(self, z_initial = None):
        if z_initial is None:
            self.z = numpy.zeros((self.num_states*self.lift_order, 1))
        else:
            self.z = z_initial.copy()

    def forward(self, x):

        self.z = numpy.vstack((x, self.z[:-x.shape[0], :]))

        u = -self.k @ self.z    

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