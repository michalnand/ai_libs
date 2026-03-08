import numpy

import AILibs
import motor_model

def compute_cost(x):
    return numpy.sum(numpy.square(x))

if __name__ == "__main__":
    dt = 0.001

    # 1st order DC motor, some random params

    b  = 0.0001  # Viscous friction
    Kt = 0.05    # Torque constant
    Ke = 0.05    # Back EMF constant
    R  = 2.0     # Armature resistance
    J  = 0.0001  # Rotor inertia

    # create dynamical system
    ds = motor_model.MotorModel(b, Kt, Ke, R, J, dt)

    # print matrices
    print(str(ds))


    # controller design

    # discretise continuous system
    a_disc, b_disc, _ = AILibs.c2d(ds.a, ds.b, None, dt)

    q   = numpy.diag([1.0])
    r   = numpy.diag([100000.0])
    qi  = 1.0

    controller = AILibs.LQRIDiscrete(a_disc, b_disc, q, r, qi)

    print("k  = ", controller.k)
    print("ki = ", controller.ki)

    # simulation steps
    n_steps = 1000

    t_result = []
    u_result = []
    xr_result= []
    x_result = []

    
    x = numpy.zeros((ds.a.shape[0], 1))
    integral_action = numpy.zeros((ds.b.shape[0], 1))

    rpm_req = 5000


    # main simulation
    for n in range(n_steps):
        if n > n_steps*0.1:
            xr = numpy.ones(x.shape)
        else:
            xr = numpy.zeros(x.shape)

        xr = xr*rpm_req/(2.0*numpy.pi*60)       

        # PID takes scalar inputs   
        u, integral_action = controller.forward(xr, x, integral_action)


        # process simulation step
        x, _ = ds.forward(x, u)

        # convert to RPM    
        xr_rpm= xr[:, 0]*(60.0*2.0*numpy.pi)  
        x_rpm = x[:, 0]*(60.0*2.0*numpy.pi)  
        

        # log results
        t_result.append(n*dt)
        u_result.append(u[:, 0])
        xr_result.append(xr_rpm)
        x_result.append(x_rpm)


    AILibs.plot_cl_response(t_result, u_result, xr_result, x_result, "plots/lqr_result.png", ["voltage"], ["rpm"])
    

    u_cost = compute_cost(numpy.array(u_result))
    x_cost = compute_cost(numpy.array(xr_result) - numpy.array(x))
    
    print(round(u_cost, 3), round(x_cost, 3))

