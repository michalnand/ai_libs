import numpy
import scipy


'''
solve LQG controller for discrete system
x(n+1) = Ax(n) + Bu(n)
y(n) = Cx(n)

Q, R are weight matrices in quadratic loss

A matrix, shape (n_states, n_states)
B matrix, shape (n_states, n_inputs)
C matrix, shape (n_outputs, n_inputs)

Q matrix, shape (n_states, n_states)
R matrix, shape (n_inputs, n_inputs)
''' 
class LQGDiscrete:

    def __init__(self, a, b, c, q, r, noise_q, noise_r, antiwindup = 10**10, di_max=10**10):
        self.k, self.ki = self.solve_lqr(a, b, c, q, r)
        self.f = self.solve_kalman_gain(a, c, noise_q, noise_r)

        self.a = a
        self.b = b
        self.c = c

        self.antiwindup = antiwindup
        self.di_max     = di_max

    '''
    inputs:
        yr : required output, shape (n_outputs, 1)
        y  : system output, shape (n_outputs, 1)
        integral_action : storage for integral action, shape (n_inputs, 1)
        x_hat : storage for estimated full state, shape (n_states, 1)

    returns:
        u : input into plant, shape (n_inputs, 1)
        integral_action_new : new IA, shape (n_inputs, 1)
        x_hat_new : storage for estimated full state, shape (n_states, 1)
    '''
    def forward(self, yr, y, integral_action, x_hat):
      
        # integral action   
        error = yr - y

        dintegral_action    = self.ki@error

        #kick clipping
        dintegral_action    = numpy.clip(dintegral_action, -self.di_max , self.di_max)
        integral_action_new = integral_action + dintegral_action


        # LQR controll law 
        u_new = -self.k@x_hat + integral_action_new

        # antiwindup with conditional integration
        u = numpy.clip(u_new, -self.antiwindup, self.antiwindup)
        integral_action_new = integral_action_new - (u_new - u)
    
        # kalman observer
        # only y is known, and using knowledge of dynamics, 
        # the full state x_hat can be reconstructed
        prediction_error = y - self.c@x_hat
        x_hat_new = self.a@x_hat + self.b@u + self.f@prediction_error

        return u, integral_action_new, x_hat_new


    '''
    solve the discrete time lqr controller for
    x(n+1) = A x(n) + B u(n)
    cost = sum x[n].T*Q*x[n] + u[n].T*R*u[n]
    '''
    def solve_lqr(self, a, b, c, q, r):
        print(q)
        print(r)
        
        n = a.shape[0]  #system order
        m = b.shape[1]  #inputs count
        k = c.shape[0]  #outputs count

        #matrix augmentation with integral action
        a_aug = numpy.zeros((n+k, n+k))
        b_aug = numpy.zeros((n+k, m))
        q_aug = numpy.zeros((n+k, n+k))

        
        a_aug[0:n, 0:n] = a 

       #add integrator into augmented a matrix
        for i in range(k):
            a_aug[i + n, i]     = 1.0
            a_aug[i + n, i + n] = 1.0

        b_aug[0:n,0:m]  = b

        #project Q matrix to output, and fill augmented q matrix
        tmp = (c@q).sum(axis=1)
        for i in range(k):
            q_aug[n+i][n+i] = tmp[i]

        
        # discrete-time algebraic Riccati equation solution
        p = scipy.linalg.solve_discrete_are(a_aug, b_aug, q_aug, r)

        # compute the LQR gain
        #ki_tmp =  numpy.linalg.inv(r)@(b_aug.T@p)
        ki_tmp = numpy.linalg.inv(r + b_aug.T @ p @ b_aug) @ (b_aug.T @ p @ a_aug)

        #truncated small elements
        ki_tmp[numpy.abs(ki_tmp) < 10**-10] = 0

        #split ki for k and integral action part ki
        k   = ki_tmp[:, 0:a.shape[0]]
        ki  = ki_tmp[:, a.shape[0]:]

        return k, ki
    
    
    '''
    compute kalman gain matrix F for observer : 
    x_hat(n+1) = Ax_hat(n) + Bu(n) + F(y(n) - Cx_hat(n))
    ''' 
    def solve_kalman_gain(self, a, c, q, r):
        p = scipy.linalg.solve_discrete_are(a.T, c.T, q, r) 
        f = p@c.T@scipy.linalg.inv(c@p@c.T + r)

        return f