import numpy

import AILibs
import position_model


if __name__ == "__main__":
    dt = 0.01

    # 1st order DC motor, some random params

    tau = 0.5
    k   = 0.3

    # create dynamical system
    ds = position_model.PositionModel(tau, k, dt)


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
        u_in = numpy.array([[u], [-0.4*u]]) 

        # process simulation step
        x, _ = ds.forward(x, u_in)

        # convert to degrees and RPM

        # log results
        t_result.append(n*dt)
        u_result.append(u_in[:, 0])
        x_result.append(x[:, 0])    

    
    AILibs.plot_response(t_result, u_result, x_result, "plots/response.png", ["input X", "input Y"], ["position X", "position Y", "velocity X", "velocity Y"])
    

