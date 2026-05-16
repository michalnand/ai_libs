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



class LQIDiscrete:
    """
    Discrete-time LQI controller.

    Plant:
        x[k+1] = A x[k] + B u[k]

    Integral state:
        z[k+1] = z[k] + (r[k] - S x[k])

    Control law:
        u[k] = -Kx x[k] - Ki z[k]
    """

    def __init__(self, A, B, Qx, Qi, R, tracking_indices=None, antiwindup=None):

        self.A = np.asarray(A)
        self.B = np.asarray(B)

        n = A.shape[0]

        # which states are tracked/integrated
        if tracking_indices is None:
            tracking_indices = [0]

        p = len(tracking_indices)

        # state selection matrix
        S = np.zeros((p, n))

        for i, idx in enumerate(tracking_indices):
            S[i, idx] = 1.0

        self.S = S

        # augmented system
        A_aug = np.block([
            [A,               np.zeros((n, p))],
            [-S,              np.eye(p)]
        ])

        B_aug = np.vstack([
            B,
            np.zeros((p, B.shape[1]))
        ])

        # augmented cost
        Q_aug = scipy.linalg.block_diag(Qx, Qi)

        # solve DARE
        P = scipy.linalg.solve_discrete_are(
            A_aug,
            B_aug,
            Q_aug,
            R
        )

        # correct discrete LQR gain
        K_aug = np.linalg.inv(
            B_aug.T @ P @ B_aug + R
        ) @ (B_aug.T @ P @ A_aug)

        self.Kx = K_aug[:, :n]
        self.Ki = K_aug[:, n:]

        self.antiwindup = antiwindup

    def forward(self, xr, x, z):

        xr = np.asarray(xr).reshape(-1, 1)
        x = np.asarray(x).reshape(-1, 1)
        z = np.asarray(z).reshape(-1, 1)

        # tracked state error
        e = xr - self.S @ x

        # update integrator
        z_new = z + e

        # control law
        u_unsat = -self.Kx @ x - self.Ki @ z_new

        u = u_unsat.copy()

        # optional saturation + antiwindup
        if self.antiwindup is not None:

            u = np.clip(
                u,
                -self.antiwindup,
                self.antiwindup
            )

            # back-calculation antiwindup
            z_new = z_new + (u - u_unsat)

        return u, z_new