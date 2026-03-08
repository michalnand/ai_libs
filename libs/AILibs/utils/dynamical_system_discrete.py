import numpy

'''
discrete linear dynamical system

x(n+1) = Ax(n) + Bu(n)
y(n)   = Cx(n)


A matrix, shape (n_states, n_states)
B matrix, shape (n_states, n_inputs)
C matrix, shape (n_outputs, n_states)

x, system state, shape (n_states, 1)
u, controll input, shape (n_inputs, 1)
y, plant output, shape (n_outputs, 1)

default C is diagonal matrix, providing full state x observation
'''
class DynamicalSystemDiscrete:

    def __init__(self, a, b, c):
        self.a = a.copy()
        self.b = b.copy()

        if c is not None:
            self.c = c.copy()
        else:
            self.c = numpy.eye(self.a.shape[0])


    #set initial state, used with forward_state
    def reset(self, x_initial = None):

        if x_initial is not None:
            self.x = x_initial.copy()
        else:
            self.x = numpy.zeros((self.a.shape[0], 1))

    
    #state-less forward func
    def forward(self, x, u):
        x_new = self.a@x + self.b@u
        y     = self.c@x
        
        return x_new, y
    
    #state forward
    def forward_state(self, u):
        self.x, y = self.forward(self.x, u)
        return self.x, y
    
   