import numpy

import AILibs
import servo_model


if __name__ == "__main__":
    dt = 0.01

    # 1st order DC motor, some random params

    tau = 0.5
    k   = 0.3

    # create dynamical system
    ds = servo_model.ServoModel(tau, k, dt)


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

        # convert to degrees and RPM
        x_res = x*2.0*numpy.pi
        x_res[0, 0]*= 360.0
        x_res[1, 0]*= 60.0

        # log results
        t_result.append(n*dt)
        u_result.append(u_in[:, 0])
        x_result.append(x_res[:, 0])    

    
    AILibs.plot_response(t_result, u_result, x_result, "plots/servo_response.png", ["voltage [V]"], ["angle [degrees]", "rpm"])
    

