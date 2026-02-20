import numpy


'''
    2nd order motor model
'''

'''
class MotorModel:
    def __init__(self, r_noise, dt  = 0.001):

        # example of motor parameters

        self.R = 0.2       # Ohm
        self.L = 0.0005    # H
        self.J = 0.002     # kg·m^2
        self.b = 0.0004    # N·m·s/rad
        self.Kt = 0.1      # Nm/A
        self.Ke = 0.01     # Vs/rad

        self.dt = dt

        self.set(self.R, self.L, self.J, self.b, self.Kt, self.Ke, r_noise, self.dt)

    def set(self, R, L, J, b, Kt, Ke, r_noise, dt):

        # Continuous-time matrices
        A_c = numpy.array([
            [-b / J,     Kt / J],
            [-Ke / L,   -R / L]
        ])

        B_c = numpy.array([
            [0],
            [1 / L]
        ])

        # Discretization
        I = numpy.eye(2)
        A_d = I + dt * A_c
        B_d = dt * B_c

        self.A = A_d
        self.B = B_d
        self.C = numpy.eye(A_d.shape[0])

        self.r = r_noise

        self.x = numpy.zeros((2, 1))

    def reset(self):
        self.x[:] = 0.0


    def step(self, u, brake_torque = None, delta_J=None):
        # handle changing inertia
        if delta_J is not None:
            new_J = self.J + delta_J
            self._update_matrices(new_J)
        else:
            new_J = float(self.J)


        # motor step
        self.x = self.A@self.x + self.B@u

        # apply external braking torque
        if brake_torque is not None and brake_torque > 0:
            omega = self.x[0, 0]  # angular velocity
            if abs(omega) > 1e-6:
                sign = -numpy.sign(omega)
                brake_accel = (sign * brake_torque) / new_J
                self.x[0, 0] += self.dt * brake_accel  

        # noise for observation
        noise = self.r@numpy.random.randn(self.x.shape[0], self.x.shape[1])

        return self.x, self.x + noise


    def _update_matrices(self, J):
        A_c = numpy.array([
            [-self.b / J,     self.Kt / J],
            [-self.Ke / self.L,   -self.R / self.L]
        ])

        B_c = numpy.array([
            [0],
            [1 / self.L]
        ])

        I = numpy.eye(2)
        self.A = I + self.dt * A_c
        self.B = self.dt * B_c

        self.J_current = J
'''




import numpy


class MotorModel:
    def __init__(self, r_noise, dt=0.001):
        self.R = 0.2       # Ohm
        self.L = 0.0005    # H
        self.J = 0.002     # kg·m²
        self.b = 0.0004    # N·m·s/rad
        self.Kt = 0.1      # Nm/A
        self.Ke = 0.01     # Vs/rad

        self.dt = dt
        self.set(self.R, self.L, self.J, self.b, self.Kt, self.Ke, r_noise, self.dt)

    def set(self, R, L, J, b, Kt, Ke, r_noise, dt):
        # Continuous-time matrices (3 states: angle, velocity, current)
        A_c = numpy.array([
            [0,        1,          0],
            [0,   -b / J,     Kt / J],
            [0, -Ke / L,    -R / L]
        ])

        B_c = numpy.array([
            [0],
            [0],
            [1 / L]
        ])

        # Discretize using forward Euler (simple + fast for small dt)
        I = numpy.eye(3)
        A_d = I + dt * A_c
        B_d = dt * B_c

        self.A = A_d
        self.B = B_d
        self.C = numpy.eye(3)

        self.r = r_noise
        self.J_current = J
        self.x = numpy.zeros((3, 1))  # [angle, angular velocity, current]

    def reset(self):
        self.x[:] = 0.0

    def step(self, u, brake_torque=None, delta_J=None):
        # Handle changing inertia
        if delta_J is not None:
            new_J = self.J + delta_J
            self._update_matrices(new_J)
        else:
            new_J = float(self.J_current)

        # Step the model
        self.x = self.A @ self.x + self.B @ u

        # Apply braking torque
        if brake_torque is not None and brake_torque > 0:
            omega = self.x[1, 0]  # angular velocity
            if abs(omega) > 1e-6:
                sign = -numpy.sign(omega)
                brake_accel = (sign * brake_torque) / new_J
                self.x[1, 0] += self.dt * brake_accel  # Δω = α * dt

        # Add noise to all states
        noise = self.r @ numpy.random.randn(self.x.shape[0], self.x.shape[1])
        return self.x, self.x + noise

    def _update_matrices(self, J):
        A_c = numpy.array([
            [0,        1,          0],
            [0,   -self.b / J,     self.Kt / J],
            [0, -self.Ke / self.L, -self.R / self.L]
        ])

        B_c = numpy.array([
            [0],
            [0],
            [1 / self.L]
        ])

        I = numpy.eye(3)
        self.A = I + self.dt * A_c
        self.B = self.dt * B_c
        self.J_current = J