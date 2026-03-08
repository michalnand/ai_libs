import numpy


class AnalyticalMPC:
    

    def __init__(self, mat_a, mat_b, mat_q, mat_r, prediction_horizon = 16, control_horizon = 4, u_max = 10**10):
        self.n = mat_a.shape[0]
        self.m = mat_b.shape[1]  
        self.u_max = u_max

        self.h = prediction_horizon

        Psi, Theta, Q_tilde, R_tilde = self._prepare_matrices(mat_a, mat_b, mat_q, mat_r, prediction_horizon, control_horizon)

        self.k, self.f = self._solve_gains(self.n, self.m, Psi, Theta, Q_tilde, R_tilde, prediction_horizon)




    def _prepare_matrices(self, A, B, Q, R, h, k):
        n = A.shape[0]
        m = B.shape[1]
        
        # construct Psi = [A; A^2; ...; A^h]  (hn x n)
        Psi_blocks = []
        A_pow = numpy.eye(n)
        for k in range(1, h+1):
            A_pow = A_pow @ A   # A^k
            Psi_blocks.append(A_pow.copy())
        Psi = numpy.vstack(Psi_blocks)   # shape (h*n, n)

        # construct Theta, block lower-triangular (hn x hm)
        Theta = numpy.zeros((h*n, h*m))
        for row in range(h):    
            for col in range(row+1):  # col <= row
                # block at (row, col) is A^(row-col) * B
                power = row - col
                A_pow = numpy.linalg.matrix_power(A, power) if power > 0 else numpy.eye(n)
                Theta[row*n:(row+1)*n, col*m:(col+1)*m] = A_pow @ B
        

        # block diagonal Q_tilde and R_tilde
        Q_tilde = numpy.kron(numpy.eye(h), Q)   # (h*n, h*n)
        R_tilde = numpy.kron(numpy.eye(h), R)   # (h*m, h*m)

        return Psi, Theta, Q_tilde, R_tilde


    def _solve_gains(self, n, m, Psi, Theta, Qtil, Rtil, h):
        """ 
        Precompute:
            H = Theta^T Qtil Theta + Rtil
            Omega = H^{-1} Theta^T Qtil   (solve linear system)
            Phi = Omega @ Psi
        Then compute selector S to extract first block, and compute:
            K = S @ Phi    (m x n)
            F = S @ Omega  (m x hn)
        """

        # H
        H = Theta.T @ Qtil @ Theta + Rtil  # (hm x hm)

        # Numerical solve for Omega: H * Omega = Theta^T Qtil  -> Omega = H^{-1} Theta^T Qtil
        M = Theta.T @ Qtil  # (hm x hn)
        # use np.linalg.solve for columns; solve H X = M  => X = solve(H, M)
        # Note: np.linalg.solve expects shape (hm, hm) and (hm, k). Works for k = hn.
        Omega = numpy.linalg.solve(H, M)  # shape (hm, hn)

        # Phi = Omega @ Psi
        Phi = Omega @ Psi              # shape (hm, n)

        # selector S (extract first m rows of U): S is m x (h*m)
        S = numpy.zeros((m, h*m))
        S[:, :m] = numpy.eye(m)

        # reduced gains
        K = S @ Phi            # shape (m, n)
        F = S @ Omega          # shape (m, h*n)

        return K, F

    def forward(self, xr, x):
        Xr = []
        for n in range(self.h):
            Xr.append(xr)
        Xr = numpy.vstack(Xr)

        return self.forward_traj(Xr, x)
    
    def forward_traj(self, Xr, x):
        """
        Compute the first control action u0 = -K x + F Xr.

        Inputs:
            x: (n,) or (n,1)
            Xr: stacked reference (h*n,) or (h*n,1)

        Returns:
            u0: (m,) numpy array
        """
        x  = numpy.asarray(x) #.reshape(self.n, )
        Xr = numpy.asarray(Xr) #.reshape(self.h*self.n, )

        # compute u0
        u = - self.k @ x + self.f @ Xr

        u  = numpy.clip(u, -self.u_max, self.u_max)

        return u   