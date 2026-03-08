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

    #kp = 0.1
    #ki = 0.01
    #kd = 0.0

    kp = 0.1
    ki = 0.0015
    kd = 0.0
    controller = AILibs.PID(kp, ki, kd)

    # simulation steps
    n_steps = 1000

    t_result = []
    u_result = []
    xr_result= []
    x_result = []

    
    x = numpy.zeros((ds.a.shape[0], 1))
    u = 0.0

    rpm_req = 5000


    # main simulation
    for n in range(n_steps):
        if n > n_steps*0.1:
            xr = numpy.ones(x.shape)
        else:
            xr = numpy.zeros(x.shape)

        xr = xr*rpm_req/(2.0*numpy.pi*60)       

        # PID takes scalar inputs
        u = controller.forward(xr[0, 0], x[0, 0], u)


        # process simulation step
        u_in = numpy.array([[u]])
        x, _ = ds.forward(x, u_in)

        # convert to RPM    
        xr_rpm= xr[:, 0]*(60.0*2.0*numpy.pi)  
        x_rpm = x[:, 0]*(60.0*2.0*numpy.pi)  
        

        # log results
        t_result.append(n*dt)
        u_result.append(u_in[:, 0])
        xr_result.append(xr_rpm)
        x_result.append(x_rpm)

    
    AILibs.plot_cl_response(t_result, u_result, xr_result, x_result, "plots/pid_result.png", ["voltage"], ["rpm"])
    
    u_cost = compute_cost(numpy.array(u_result))
    x_cost = compute_cost(numpy.array(xr_result) - numpy.array(x))
    
    print(round(u_cost, 3), round(x_cost, 3))

