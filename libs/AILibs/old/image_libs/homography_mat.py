import numpy

def homography_translation(tx, ty):
    return numpy.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=numpy.float32)


def homography_rotation(theta):
    cos_t, sin_t = numpy.cos(theta), numpy.sin(theta)

    return numpy.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ], dtype=numpy.float32)


def homography_scaling(sx, sy):
    return numpy.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ], dtype=numpy.float32)


def homography_shear(shx, shy):
    return numpy.array([
        [1, shx, 0],
        [shy, 1, 0],
        [0,  0,  1]
    ], dtype=numpy.float32)

def homography_perspective(px, py):
    return numpy.array([
        [1, 0, 0],
        [0, 1, 0],
        [px, py, 1]
    ], dtype=numpy.float32)


#def homography_random(translation_range = [-1.0, 1.0], rotation_range = [-3.141/4.0, 3.141/4.0], scale_range = [0.5, 1.5], shear_range = [-0.25, 0.25], perspective_range = [-0.0002, 0.0002]):


def homography_random(translation_range = [-1.0, 1.0], rotation_range = [-3.141/4.0, 3.141/4.0], scale_range = [0.5, 1.5], shear_range = [-0.1, 0.1], perspective_range = [-0.0002, 0.0002]):
    tx      = numpy.random.uniform(translation_range[0], translation_range[1])
    ty      = numpy.random.uniform(translation_range[0], translation_range[1])
    angle   = numpy.random.uniform(rotation_range[0], rotation_range[1])

    sx      = numpy.random.uniform(scale_range[0], scale_range[1])
    sy      = numpy.random.uniform(scale_range[0], scale_range[1])

    shx      = numpy.random.uniform(shear_range[0], shear_range[1])
    shy      = numpy.random.uniform(shear_range[0], shear_range[1])

    px      = numpy.random.uniform(perspective_range[0], perspective_range[1])
    py      = numpy.random.uniform(perspective_range[0], perspective_range[1])

    transforms = [
        homography_translation(tx, ty),
        homography_rotation(angle),
        homography_scaling(sx, sy),
        homography_shear(shx, shy),
        homography_perspective(px, py)
    ]

    # randomise order
    numpy.random.shuffle(transforms)
    
    # combine transforms
    h = numpy.eye(3)
    for t in transforms:
        h = h @ t

    return h