import numpy 

# discretise model using bilinear transform
def c2d(a, b, c, dt):
    i = numpy.eye(a.shape[0])
    
    tmp_a = numpy.linalg.inv(i - (0.5*dt)*a)
    tmp_b = i + (0.5*dt)*a

    a_disc  = tmp_a@tmp_b
    b_disc  = (tmp_a*dt)@b

    if c is not None:
        c_disc  = c
    else:
        c_disc = numpy.eye(a.shape[0])

    return a_disc, b_disc, c_disc