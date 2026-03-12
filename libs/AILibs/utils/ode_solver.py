'''
    Euler solver, 1st order
'''
def ODESolverEuler(forward_func, x, u, dt):
    dx, y  = forward_func(x, u)
    return x + dx*dt, y

'''
    Runge-Kuta solver, 4th order
'''
def ODESolverRK4(forward_func, x, u, dt):
    k1, y1  = forward_func(x, u)    
    k1      = k1*dt  

    k2, y2  = forward_func(x + 0.5*k1, u)
    k2      = k2*dt

    k3, y3  = forward_func(x + 0.5*k2, u)
    k3      = k3*dt

    k4, y4  = forward_func(x + k3, u)
    k4      = k4*dt

    x_new   = x + (1.0/6.0)*(1.0*k1 + 2.0*k2 + 2.0*k3 + 1.0*k4)
    y       = (1.0/6.0)*(1.0*y1 + 2.0*y2 + 2.0*y3 + 1.0*y4)
    
    return x_new, y

