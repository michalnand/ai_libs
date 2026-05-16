import numpy

class MPCAnalytical:
    """
    A: (n_x, n_x)
    B: (n_x, n_u)
    Q: (n_x, n_x) (state cost)
    R: (n_u, n_u) (input cost)
    prediction_horizon  : Hp (how many future states)
    control_horizon     : Hc (how many future inputs we optimize; typically <= Hp)
    """
    def __init__(self, A, B, Q, R, prediction_horizon=16, control_horizon=4, u_max=1e10, du_max=1e10):

        self.A      = A
        self.B      = B
        self.nx     = A.shape[0]
        self.nu     = B.shape[1]
        self.Hp     = prediction_horizon
        self.Hc     = control_horizon
        self.u_max  = u_max
        self.du_max = du_max  # max change in input (for rate limits)

        # 1, build Phi and Theta
        self.Phi, self.Omega, self.Theta = self._build_prediction_matrices(A, B, self.Hp, self.Hc)

        # 2, build augmented tilde Q and tilde R, block-diagonal
        self.Q_aug = numpy.kron(numpy.eye(self.Hp), Q)
        self.R_aug = numpy.kron(numpy.eye(self.Hc), R)

        # Precompute solver matrices: G and Sigma
        G = self.Theta.T @ self.Q_aug @ self.Theta + self.R_aug
        
        # use solve later for stability; but precompute factorization if desired
        # here we compute Sigma by solving G Sigma^T = Theta^T Q_aug  (do via solve)
        # Sigma has shape (n_u*Hc, n_x*Hp)
        # Solve H @ Sigma = Theta.T @ Q_aug
        # Sigma = numpy.linalg.solve(H, Theta.T @ Q_aug)
        self.Sigma  = numpy.linalg.solve(G, self.Theta.T @ self.Q_aug)
        self.Sigma0 = self.Sigma[:self.nu, :]

    def _build_prediction_matrices(self, A, B, Hp, Hc):
        nx = A.shape[0]
        nu = B.shape[1]
        # precompute A powers: A^0 ... A^Hp
        A_pows = [numpy.eye(nx)]
        for i in range(1, Hp + 1):
            A_pows.append(A_pows[-1] @ A)

        # Phi: (nx*Hp, nx) stacked [A; A^2; ...; A^Hp]
        Phi = numpy.zeros((nx * Hp, nx))
        for i in range(Hp):
            Phi[i * nx:(i + 1) * nx, :] = A_pows[i + 1]  # A^(i+1)


        # Construct result_omega matrix
        Omega = numpy.zeros((nx * Hp, nu))        
        for n in range(Hp):
            tmp = numpy.zeros_like(B)
            for i in range(n + 1):  # Cumulative effect up to n+1
                tmp += A_pows[i] @ B
            ofs = n * nx    
            Omega[ofs:ofs + nx, :] = tmp

        # Theta: (nx*Hp, nu*Hc) where block (i,j) is A^(i-j) B for i>=j, else 0
        Theta = numpy.zeros((nx * Hp, nu * Hc))
        for i in range(Hp):
            for j in range(Hc):
                if i >= j:
                    # A^{i-j} B
                    Theta[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu] = A_pows[i - j] @ B
                else:
                    # remains zero
                    pass

        return Phi, Omega, Theta

    '''
    def forward_traj(self, Xr, x, u_prev):
        # residual
        e = Xr - self.Phi @ x - self.Omega@u_prev

        du = self.Sigma0@e

        du = numpy.clip(du, -self.du_max, self.du_max)
        u  = numpy.clip(u_prev + du, -self.u_max, self.u_max)

        return u    
    '''

    def forward_traj(self, Xr, x, u_prev):
        # residual
        e = Xr - self.Phi @ x   

        u = self.Sigma0@e

        u  = numpy.clip(u, -self.u_max, self.u_max)

        return u     
    

    def forward(self, xr, x, u_prev):

        Xr = []
        for n in range(self.Hp):
            Xr.append(xr)
        Xr = numpy.concatenate(Xr)
        return self.forward_traj(Xr, x, u_prev)