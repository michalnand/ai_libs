import numpy
import matplotlib.pyplot as plt

import AILibs

from motor_model import *
from motor_demo  import * 

def evaluate(motor_model, kalman_filter, n_samples):

    t_result  = []
    u_result  = []
    d_result  = []
    xr_result = []
    yr_result = []
    xm_result = []
    a_result  = []


    motor_model.reset()
    kalman_filter.reset()

    for n in range(n_samples):

        # square wave control input
        if (n//2000)%2 != 0:
            u_in = 0.25
        else:
            u_in = 0.0

        # add disturbance
        if n > n_samples/2:
            brake_force = 0.05
        else:
            brake_force = 0.0   


        if n > n_samples/2 and False:
            inertia_change = 0.002
        else:
            inertia_change = 0.0


        u_tmp = numpy.array([[u_in]])

        xr, yr = motor_model.step(u_tmp, brake_force, inertia_change)

        xm, a = kalman_filter.step(yr, u_tmp)


        t_result.append(n*dt)
        u_result.append(u_in)
        d_result.append(brake_force)
        xr_result.append(xr[:, 0])
        yr_result.append(yr[:, 0])
        xm_result.append(xm[:, 0])
        a_result.append(a)

    t_result    = numpy.array(t_result)
    u_result    = numpy.array(u_result)
    d_result    = numpy.array(d_result)
    xr_result   = numpy.array(xr_result)
    yr_result   = numpy.array(yr_result)
    xm_result   = numpy.array(xm_result)
    a_result    = numpy.array(a_result)

    fig, ax = plt.subplots(5)

    ax[0].plot(t_result, u_result, c='blue', label="control")
    ax[0].plot(t_result, d_result, c='red', label="disturbance")
    ax[0].set_ylabel("input")
    ax[0].legend()


    ax[1].plot(t_result, yr_result[:, 0], c='green',  alpha=0.5, label="observed")
    ax[1].plot(t_result, xr_result[:, 0], c='blue', label="reference", lw=4)
    ax[1].plot(t_result, xm_result[:, 0], c='red', label="predicted")
    ax[1].set_ylabel("angle [rad]")
    ax[1].legend()
    
    ax[2].plot(t_result, yr_result[:, 1], c='green',  alpha=0.5, label="observed")
    ax[2].plot(t_result, xr_result[:, 1], c='blue', label="reference", lw=4)
    ax[2].plot(t_result, xm_result[:, 1], c='red', label="predicted")
    ax[2].set_ylabel("velocity [rad/s]")
    ax[2].legend()

    ax[3].plot(t_result, yr_result[:, 2], c='green', alpha=0.5, label="observed")
    ax[3].plot(t_result, xr_result[:, 2], c='blue', label="reference", lw=4)
    ax[3].plot(t_result, xm_result[:, 2], c='red', label="predicted")
    ax[3].set_ylabel("current [A]")
    ax[3].legend()


    ax[4].plot(t_result, a_result, c='blue')
    ax[4].set_ylabel("anomaly score")
    ax[4].set_xlabel("time [s]")
    

    plt.legend()
    plt.show()

if __name__ == "__main__":
    dt = 0.001

    # sensor noise
    r_noise = numpy.diag([0.2, 1.0, 0.1])

    # model noise
    q = 10e-6
    q_noise = numpy.diag([q, q, q]) 


    real_system  = MotorModel(r_noise, dt)

    kalman_filter = AILibs.KalmanFilterAnomaly(real_system.A, real_system.B, real_system.C, q_noise, r_noise)

    n_samples = 20000
    evaluate(real_system, kalman_filter, n_samples)

    
    demo = MotorDemo(real_system, kalman_filter)

    while True:
        res = demo.step()
        if res != True:
            break
    