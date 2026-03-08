import numpy

import AILibs
import servo_model

def compute_cost(x):
    return numpy.sum(numpy.square(x))


if __name__ == "__main__":
    dt = 0.01

    # 1st order DC motor, some random params

    tau = 0.5
    k   = 0.3

    # create dynamical system
    ds = servo_model.ServoModel(tau, k, dt)

    # print matrices    
    print(str(ds))



    # controller design

    # discretise continuous system
    a_disc, b_disc, _ = AILibs.c2d(ds.a, ds.b, None, dt)

    print(a_disc)
    print(b_disc)

    q = numpy.diag([1.0, 0.0])
    r = numpy.diag([1.0])

    controller = AILibs.LQRIDiscrete(a_disc, b_disc, q, r, 1.0)

    print("k  = ", controller.k)
    print("ki = ", controller.ki)

    # simulation steps
    n_steps = 1000

    t_result = []
    u_result = []
    xr_result= []
    x_result = []

    
    x = numpy.zeros((ds.a.shape[0], 1))
    integral_action = numpy.zeros((ds.b.shape[1], 1))

    angle_req = 1000


    # main simulation
    for n in range(n_steps):
        if n > n_steps*0.7:
            xr = numpy.zeros(x.shape)
        elif n > n_steps*0.1:
            xr = numpy.ones(x.shape)
        else:
            xr = numpy.zeros(x.shape)

        xr[1, 0] = 0
        xr = xr*angle_req/(2.0*numpy.pi*360)       

        # PID takes scalar inputs   
        u, integral_action = controller.forward(xr, x, integral_action)

        # process simulation step
        x, _ = ds.forward(x, u)

        # convert to degrees and RPM
        xr_res = xr*2.0*numpy.pi
        xr_res[0, 0]*= 360.0
        xr_res[1, 0]*= 60.0

        x_res = x*2.0*numpy.pi
        x_res[0, 0]*= 360.0
        x_res[1, 0]*= 60.0


        

        # log results
        t_result.append(n*dt)
        u_result.append(u[:, 0])
        xr_result.append(xr_res)
        x_result.append(x_res)


    AILibs.plot_cl_response(t_result, u_result, xr_result, x_result, "plots/lqr_result.png",  ["input"], ["angle [degrees]", "rpm"])
    

    u_cost = compute_cost(numpy.array(u_result))
    x_cost = compute_cost(numpy.array(xr_result) - numpy.array(x))
    print(round(u_cost, 3), round(x_cost, 3))

