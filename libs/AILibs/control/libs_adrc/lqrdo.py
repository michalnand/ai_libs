import numpy
import scipy.linalg

class LQRDO:
    def __init__(self, A, B, Q, R, Qo, Ro):
        """
        Parallel Direct-LQR with Steady-State Kalman Disturbance Observer
        A : Transition matrix, (n x n)
        B : Input matrix, (n x m)
        Q : quadratic cost weights for state, (n x n)
        R : quadratic cost weights for control effort, (m x m)
        Qo : kalman process noise, how much we trust to model, (n + m) x (n + m)
        Ro : measurement noise, how much we trust to sensor, (n x n)
        """
        self.A = numpy.array(A, dtype=float)
        self.B = numpy.array(B, dtype=float)

        self.n = self.A.shape[0]  # Number of states
        self.m = self.B.shape[1]  # Number of inputs/disturbances

        # 1. Build Augmented Matrices for the Observer
        top_A = numpy.hstack((self.A, self.B))
        bot_A = numpy.hstack((numpy.zeros((self.m, self.n)), numpy.eye(self.m)))
        self.A_aug = numpy.vstack((top_A, bot_A))
        
        self.B_aug = numpy.vstack((self.B, numpy.zeros((self.m, self.m))))
        self.C_aug = numpy.hstack((numpy.eye(self.n), numpy.zeros((self.n, self.m))))

        # 2. Synthesize LQR (using ideal A, B)
        self.K = self._solve_lqr(self.A, self.B, Q, R)

        # 3. Synthesize Kalman Filter (using augmented A, C)
        self.L = self._solve_kalman(self.A_aug, self.C_aug, Qo, Ro)
        
        # 4. Memory for the next loop
        self.reset()

    def reset(self):
        """Clear memory (e.g., if the robot is picked up or motors disabled)"""
        self.X_hat = numpy.zeros((self.n + self.m, 1))
        self.u_prev = numpy.zeros((self.m, 1))

    def forward(self, xr, x):
        """
        Calculates control effort using parallel LQR and DOB.
        """
        x  = numpy.array(x, dtype=float)
        xr = numpy.array(xr, dtype=float)

        #print(x.shape, xr.shape)

        # ========================================================
        # BLOCK 1: Direct LQR Control (Raw State Feedback)
        # ========================================================
        u_lqr = -self.K @ (x - xr)

        #print(u_lqr.shape)


        # ========================================================
        # BLOCK 2: Parallel Kalman Disturbance Observer
        # ========================================================
        # Predict where the robot should be based on the last total u sent
        X_bar = self.A_aug @ self.X_hat + self.B_aug @ self.u_prev

        # Correct prediction using the raw measured state
        x_predicted = self.C_aug @ X_bar
        self.X_hat = X_bar + self.L @ (x - x_predicted)

        # Extract disturbance and flip sign for cancellation
        d_hat = self.X_hat[self.n:]
        u_d = -d_hat

        # ========================================================
        # BLOCK 3: Total Control Effort
        # ========================================================
        u_total = u_lqr + u_d

        # Save total effort for the observer's prediction in the next loop
        self.u_prev = u_total

        return u_total

    # --- Synthesis Helpers ---
    def _solve_lqr(self, a, b, q, r):
        p = scipy.linalg.solve_discrete_are(a, b, q, r)
        k = numpy.linalg.inv(r + b.T @ p @ b) @ (b.T @ p @ a)
        return k
 
    def _solve_kalman(self, a, c, q, r):
        p = scipy.linalg.solve_discrete_are(a.T, c.T, q, r) 
        f = p @ c.T @ scipy.linalg.inv(c @ p @ c.T + r)
        return f