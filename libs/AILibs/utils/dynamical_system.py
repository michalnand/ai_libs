import numpy
from .ode_solver import *

'''
continuous linear dynamical system

dx = Ax + Bu
y  = Cx

for solving runge kutta (RK4) is used

A matrix, shape (n_states, n_states)
B matrix, shape (n_states, n_inputs)
C matrix, shape (n_outputs, n_states)

x, system state, shape (n_states, 1)
u, controll input, shape (n_inputs, 1)
y, plant output, shape (n_outputs, 1)

default C is diagonal matrix, providing full state x observation
'''
class DynamicalSystem:

    def __init__(self, a, b, c, dt):

        self.a = a.copy()
        self.b = b.copy()

        if c is not None:
            self.c = c.copy()
        else:
            self.c = numpy.eye(self.a.shape[0])

        self.dt = dt

        self.x = numpy.zeros((self.a.shape[0], 1))
        


    def __repr__(self):

        result = ""
        
        result+= "mat_a = \n"
        for j in range(self.a.shape[0]):
            for i in range(self.a.shape[1]):
                result+= str(round(self.a[j][i], 5)) + " "
            result+= "\n"
        result+= "\n\n"


        result+= "mat_b = \n"
        for j in range(self.b.shape[0]):
            for i in range(self.b.shape[1]):
                result+= str(round(self.b[j][i], 5)) + " "
            result+= "\n"
        result+= "\n\n"

        result+= "mat_c = \n"
        for j in range(self.c.shape[0]):
            for i in range(self.c.shape[1]):
                result+= str(round(self.c[j][i], 5)) + " "
            result+= "\n"
        result+= "\n\n"

        return result

    #set initial state, used with forward_state
    def reset(self, x_initial = None):

        if x_initial is not None:
            self.x = x_initial.copy()
        else:
            self.x = numpy.zeros((self.a.shape[0], 1))

        self.y = numpy.zeros((self.c.shape[0], 1))

    
    #state-less forward func
    def forward(self, x, u, rk_solver = True):
        if rk_solver:
            x_new, y = ODESolverRK4(self._step, x, u, self.dt)
        else:
            x_new, y = ODESolverEuler(self._step, x, u, self.dt)

        return x_new, y
    
    #state forward
    def forward_state(self, u):
        self.x, self.y = self.forward(self.x, u)
        return self.x, self.y
    
    #step callback for solver
    def _step(self, x, u):
        dx = self.a@x + self.b@u
        y  = self.c@x

        return dx, y
    

    # finds minimum and maximum system time constant
    def find_min_max_tau(self):        
        poles = numpy.linalg.eigvals(self.a)

        tau_min = 10**10
        tau_max = 0

        for p in poles:
            tmp = abs(p.real)
            if tmp > 10**-10:
                tau = 1.0/tmp
                
                if tau > tau_max:
                    tau_max = tau

                if tau < tau_min:
                    tau_min = tau

        return tau_min, tau_max