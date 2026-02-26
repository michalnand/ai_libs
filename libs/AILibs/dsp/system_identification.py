import numpy

def system_identification(input_u, output_x, optimizer, dictionary = None):
    """
        dynamical system identification,
        linear system model     : x(n+1) = Ax(n) + Bu(n)
        nonlinear system model  : x(n+1) = Wf(x(n), u(n))

        input_u.shape  = (num_samples, num_inputs)
        output_x.shape = (num_samples, num_states)

        optimizer   : function, or functor, input is (x, y) and fits model W : y = xW
        dictionary  : non linear state augmentation, lifting
    """

    num_inputs = input_u.shape[1]
    num_states = output_x.shape[1]  

    z = numpy.hstack((input_u, output_x))
    
    # nonlinear state augmentation, optional
    if dictionary is not None:
        z_tmp = dictionary(z)
    else:
        z_tmp = z


    # fit model W : x_next = z W
    x_next = output_x[1:]
    z = z_tmp[:-1]

    # obtain single matrix W
    W = optimizer(z, x_next)

    # transpose W to have states and inputs in rows, and states in columns
    W = W.T

    # decompose to A, B
    A = W[0:num_states, 0:num_states]
    B = W[num_states:, 0:num_states]

    return A, B, W