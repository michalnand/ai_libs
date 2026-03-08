import numpy

import AILibs
import position_model
import visualisation

if __name__ == "__main__":
    dt = 0.01

    # 1st order DC motor, some random params

    tau = 0.5
    k   = 0.3

    # create dynamical system
    ds = position_model.PositionModel(tau, k, dt)

    # print matrices
    print(str(ds))  



    # controller design

    # discretise continuous system
    a_disc, b_disc, _ = AILibs.c2d(ds.a, ds.b, None, dt)

    print(a_disc)
    print(b_disc)

    q = numpy.diag([1.0, 1.0, 0.0, 0.0])
    r = numpy.diag([1.0, 1.0])  

    controller = AILibs.LQRIDiscrete(a_disc, b_disc, q, r, 10.0)

    print("k  = ", controller.k)
    print("ki = ", controller.ki)

    # simulation steps
    n_steps = 1000

    t_result = []
    u_result = []
    xr_result= []
    x_result = []

    x  = numpy.zeros((ds.a.shape[0], 1))

    integral_action = numpy.zeros((ds.b.shape[1], 1))

    pos_req = 1.0

    x_pos_req = 0.0
    y_pos_req = 0.0

    # main simulation, steps for logs
    for n in range(n_steps):
        
        turn_state = (n//(n_steps//4))%4
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
        
        # PID takes scalar inputs   
        u, integral_action = controller.forward(xr, x, integral_action)

        # process simulation step
        x, _ = ds.forward(x, u)

      

        # log results
        t_result.append(n*dt)
        u_result.append(u[:, 0])
        xr_result.append(xr[:, 0])
        x_result.append(x[:, 0])

    t_result = numpy.array(t_result)
    u_result = numpy.array(u_result)
    xr_result = numpy.array(xr_result)
    x_result = numpy.array(x_result)

    AILibs.plot_cl_response(t_result, u_result, xr_result, x_result, "plots/lqr_result.png",  ["input X", "input Y"],  ["position X", "position Y", "velocity X", "velocity Y"])
    AILibs.plot_cl_response(t_result, u_result[:, [1]], xr_result[:, [1, 3]], x_result[:, [1, 3]], "plots/lqr_result_x.png",  [ "input X"],  [ "position X",  "velocity X"])


    ds.reset()



    visualisation = visualisation.TrajectoryRenderer()

    
    x_pos_req = 0.0
    y_pos_req = 0.0

    n = 0

    # main simulation demo
    while True:
        
        turn_state = (n//(n_steps//4))%4
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
        
        # PID takes scalar inputs   
        u, integral_action = controller.forward(xr, x, integral_action)

        # process simulation step
        x, _ = ds.forward(x, u)

        n+= 1

        if n%2 == 0:
            positions = []
            positions.append(xr[0:2, 0])
            positions.append(x[0:2, 0])
            res = visualisation.step(positions, [[1.0, 1.0, 1.0], [0.2, 0.0, 0.8]], ["reference", "LQR"])

            if res != 0:
                break