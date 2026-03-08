import numpy

import AILibs
import position_model
import visualisation



def get_required(n, n_steps, prediction_horizon):
    Xr = []
    for h in range(prediction_horizon):

        turn_state = ((n + h)//(n_steps//4))%4

        if turn_state == 0:
            x_pos_req = 1.0
            y_pos_req = 0.0
        elif turn_state == 1:
            x_pos_req = 1.0
            y_pos_req = 1.0
        elif turn_state == 2:
            x_pos_req = 0.0
            y_pos_req = 1.0
        else:
            x_pos_req = 0.0
            y_pos_req = 0.0

        xr = numpy.zeros((ds.a.shape[0], 1))

        xr[0, 0] = x_pos_req
        xr[1, 0] = y_pos_req

        Xr.append(xr)

    Xr = numpy.vstack(Xr)

    return Xr


if __name__ == "__main__":
    dt = 0.01
    u_limit = 10

    # create continuous dynamical system
    tau = 0.5
    k   = 0.3
    ds  = position_model.PositionModel(tau, k, dt)

    # controller design

    # discretise continuous system
    a_disc, b_disc, _ = AILibs.c2d(ds.a, ds.b, None, dt)

    q = numpy.diag([10000, 10000, 0.0, 0.0])
    r = numpy.diag([1.0, 1.0])  

    # compute controller
    prediction_horizon = 64
    control_horizon    = 4
    controller = AILibs.MPCFGM(a_disc, b_disc, q, r, prediction_horizon, control_horizon, u_limit)


    # simulation steps
    n_steps = 1000

    # log results
    t_result = []
    u_result = []
    xr_result= []
    x_result = []

    x  = numpy.zeros((ds.a.shape[0], 1))
    u  = numpy.zeros((ds.b.shape[1], 1))

    # main simulation, steps for logs
    for n in range(n_steps):
        # Xr.shape is (4*prediction_horizon, 1)
        Xr = get_required(n, n_steps, prediction_horizon)
        
        # MPC step, u.shape is (2, 1)
        u = controller.forward_traj(Xr, x)

        # process simulation step, ODE solver single step
        # receive current state x, current control u
        # and return new state x
        x, _ = ds.forward(x, u)

        # log results
        t_result.append(n*dt)
        u_result.append(u[:, 0])
        xr_result.append(Xr[0:4, 0])
        x_result.append(x[:, 0])

    # list to numpy arrays
    t_result = numpy.array(t_result)
    u_result = numpy.array(u_result)
    xr_result = numpy.array(xr_result)
    x_result = numpy.array(x_result)

    # plot results
    AILibs.plot_cl_response(t_result, u_result, xr_result, x_result, "plots/mpc_fgm_result.png",  ["input X", "input Y"],  ["position X", "position Y", "velocity X", "velocity Y"])
    AILibs.plot_cl_response(t_result, u_result[:, [1]], xr_result[:, [1, 3]], x_result[:, [1, 3]], "plots/mpc_fgm_result_x.png",  [ "input X"],  [ "position X",  "velocity X"])


    ds.reset()


    visualisation = visualisation.TrajectoryRenderer()

    
    x_pos_req = 0.0
    y_pos_req = 0.0

    n = 0

    u[:] = 0

    
    # main simulation demo
    while True:

        Xr = get_required(n, n_steps, prediction_horizon)
        
        # PID takes scalar inputs   
        u = controller.forward_traj(Xr, x)

        # process simulation step
        x, _ = ds.forward(x, u) 

        n+= 1

        if n%2 == 0:    
            positions = []
            positions.append(Xr[0:2, 0])
            positions.append(x[0:2, 0])
            res = visualisation.step(positions, [[1.0, 1.0, 1.0], [0.8, 0.0, 0.2]], ["reference", "MPC"])

            if res != 0:
                break
    