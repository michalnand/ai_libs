import numpy

import AILibs
import position_model
import visualisation



def get_required_old(n, n_steps, prediction_horizon, n_states):
    Xr = []
    for h in range(prediction_horizon):

        turn_state = ((n + h)//(n_steps//8))%8

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
        

        xr = numpy.zeros((n_states, 1))

        xr[0, 0] = x_pos_req
        xr[1, 0] = y_pos_req

        Xr.append(xr)

    Xr = numpy.vstack(Xr)

    return Xr




def get_required(n, n_steps, prediction_horizon, n_states):
    Xr = []
    for h in range(prediction_horizon):

        turn_state = ((n + h)//(n_steps//4))

        x_pos_req = ((1664525*turn_state + 1013904223)%100)/100.0
        y_pos_req = ((22695477*turn_state + 1)%100)/100.0


        xr = numpy.zeros((n_states, 1))

        xr[0, 0] = x_pos_req
        xr[1, 0] = y_pos_req

        Xr.append(xr)

    Xr = numpy.vstack(Xr)

    return Xr


if __name__ == "__main__":
    dt = 0.01
    u_limit = 10

    # 1st order DC motor, some random params

    tau = 0.5
    k   = 0.3

    # create continuous dynamical system
    ds_lqr = position_model.PositionModel(tau, k, dt)
    ds_mpc = position_model.PositionModel(tau, k, dt)

  
    # controller design

    # discretise continuous system
    a_disc, b_disc, _ = AILibs.c2d(ds_lqr.a, ds_lqr.b, None, dt)

    # synthetise LQR optimal control
    q = numpy.diag([1.0, 1.0, 0.0, 0.0])
    r = numpy.diag([1.0, 1.0])  

    lqr_controller = AILibs.LQRIDiscrete(a_disc, b_disc, q, r, u_limit)


    # synthetise model predictive control
    q = numpy.diag([10000, 10000, 0.0, 0.0])
    r = numpy.diag([1.0, 1.0])  

    prediction_horizon = 64
    control_horizon    = 4
    mpc_controller = AILibs.MPCAnalytical(a_disc, b_disc, q, r, prediction_horizon, control_horizon, u_limit)


    visualisation = visualisation.TrajectoryRenderer(history_length=100)

    
    n_steps   = 1000
   
    x_lqr           = numpy.zeros((ds_lqr.a.shape[0], 1))
    integral_action = numpy.zeros((ds_lqr.b.shape[1], 1))

    x_mpc  = numpy.zeros((ds_mpc.a.shape[0], 1))

    n = 0

    
    # main simulation demo
    while True:

        Xr = get_required(n, n_steps, prediction_horizon, ds_lqr.a.shape[0])
        
        # MPC output
        u_lqr, integral_action = lqr_controller.forward(Xr[0:4], x_lqr, integral_action)

        # process simulation step
        x_lqr, _ = ds_lqr.forward(x_lqr, u_lqr) 

        # MPC output
        u_mpc = mpc_controller.forward_traj(Xr, x_mpc)

        # process simulation step
        x_mpc, _ = ds_mpc.forward(x_mpc, u_mpc) 

        n+= 1

        if n%2 == 0:    
            positions = []
            positions.append(Xr[0:2, 0])
            positions.append(x_lqr[0:2, 0])
            positions.append(x_mpc[0:2, 0])

            res = visualisation.step(positions, [[1.0, 1.0, 1.0], [0.2, 0.0, 0.8], [0.8, 0.0, 0.2]], ["reference", "LQR", "MPC"])

            if res != 0:
                break
    