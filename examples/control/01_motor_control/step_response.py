import numpy

import AILibs
import motor_model


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

    # simulation steps
    n_steps = 1000

    t_result = []
    u_result = []
    x_result = []

    
    x = numpy.zeros((ds.a.shape[0], 1))

    # main simulation
    for n in range(n_steps):
        if n > n_steps*0.1:
            u = 1.0
        else:
            u = 0.0

        # input into ds is matrix
        u_in = numpy.array([[u]])

        # process simulation step
        x, _ = ds.forward(x, u_in)

        # convert to RPM
        x_rpm = x[:, 0]*(60.0*2.0*numpy.pi)  

        # log results
        t_result.append(n*dt)
        u_result.append(u_in[:, 0])
        x_result.append(x_rpm)

    
    AILibs.plot_response(t_result, u_result, x_result, "plots/motor_response.png", ["voltage"], ["rpm"])
    

