import numpy

class MPCFGM:
    """
    A: (n_x, n_x)
    B: (n_x, n_u)
    Q: (n_x, n_x) (state cost)
    R: (n_u, n_u) (input cost)
    prediction_horizon  : Hp (how many future states)
    control_horizon     : Hc (how many future inputs we optimize; typically <= Hp)
    """
    def __init__(self, A, B, Q, R, prediction_horizon=16, control_horizon=4, u_max=1e10):

        self.A      = A
        self.B      = B
        self.nx     = A.shape[0]
        self.nu     = B.shape[1]
        self.Hp     = prediction_horizon
        self.Hc     = control_horizon
        self.u_max  = u_max

        # 1, build Phi and Theta
        self.Phi, self.Theta = self._build_prediction_matrices(A, B, self.Hp, self.Hc)

        # 2, build augmented tilde Q and tilde R, block-diagonal
        self.Q_aug = numpy.kron(numpy.eye(self.Hp), Q)
        self.R_aug = numpy.kron(numpy.eye(self.Hc), R)

        # 3, quadratic cost matrices
        self.H = self.Theta.T @ self.Q_aug @ self.Theta + self.R_aug

        

        # 4, Precompute Lipschitz constant for gradient: L = 2 λ_max(H)
        self.L = 2 * self._estimate_lmax(self.H)
        self.alpha = 1.0 / self.L    # Nesterov step size

        print(self.H)
        print(self.L)



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

        return Phi, Theta
    
    
    def _estimate_lmax(self, M, iters=32):
        n = M.shape[0]
        x = numpy.random.randn(n)
        x /= numpy.linalg.norm(x)

        for _ in range(iters):
            x = M @ x
            nrm = numpy.linalg.norm(x)
            if nrm < 1e-12:
                break
            x /= nrm    

        return float(x @ (M @ x))
    
    def forward_traj(self, Xr, x, iters=8):
        # e = Xr - Phi x   → (nx*Hp,1)
        e = Xr - self.Phi @ x

        # h = Theta^T (Q_aug e)   → (nU,1)
        h = self.Theta.T @ (self.Q_aug @ e)

        nU = self.nu * self.Hc

        # column vectors
        U = numpy.zeros((nU, 1))
        Y = numpy.zeros((nU, 1))
        t = 1.0

        # Fast Gradient Method (Nesterov)
        for _ in range(iters):

            # gradient = 2 (H Y - h)   → always (nU,1)
            g = 2.0 * (self.H @ Y - h)

            # gradient step + projection
            U_new = Y - self.alpha * g
            U_new = numpy.clip(U_new, -self.u_max, self.u_max)

            # Nesterov momentum update
            t_new = 0.5 * (1 + numpy.sqrt(1 + 4*t*t))
            beta  = (t - 1) / t_new
            Y = U_new + beta * (U_new - U)

            U = U_new
            t = t_new

        # first control block → (nu,1)
        u = U[:self.nu]
        return u


    '''
    def forward_traj(self, Xr, x, iters=8):

        # e = reference error in predicted state space
        e = Xr - self.Phi @ x

        # h = Theta Q_aug e   (linear term of QP cost)
        h = self.Theta.T @ (self.Q_aug @ e)

        nU = self.nu * self.Hc

        # initialisatiom
        U = numpy.zeros(nU)
        Y = numpy.zeros(nU)
        t = 1.0 

        # fast gradient method
        for _ in range(iters):

            # gradient = 2 (H Y - h)
            g = 2 * (self.H @ Y - h)

            # gradient step + box projection
            U_new = Y - self.alpha * g
            U_new = numpy.clip(U_new, -self.u_max, self.u_max)

            # Nesterov update
            t_new = 0.5 * (1 + numpy.sqrt(1 + 4*t*t))
            beta  = (t - 1) / t_new
            Y = U_new + beta * (U_new - U)

            U = U_new
            t = t_new
        
        # only first control action
        u = U[:self.nu, 0]
        u = numpy.expand_dims(u, 1)
        
        return u
    '''
   