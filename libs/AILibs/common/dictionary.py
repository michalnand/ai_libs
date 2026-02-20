import numpy

def dictionary_constant(x):
    """
    returns constant term to the features, value = 1
    """
    return numpy.ones((x.shape[0], 1), dtype=numpy.float32)
    
def dictionary_polynomial(x, order):
    """
    returns polynomial terms to the features up to the specified order
    """
    poly_terms = [x ** p for p in range(2, order + 1)]
    return numpy.concatenate(poly_terms, axis=1)


def dictionary_cross_products(x):
    """
    Generate unique pairwise feature products (i < j).

    Parameters:
    - x: np.ndarray of shape (batch_size, num_features)

    Returns:
    - np.ndarray of shape (batch_size, num_unique_pairs)
    """
    batch_size, num_features = x.shape
    indices = [(i, j) for i in range(num_features) for j in range(i + 1, num_features)]
    products = [x[:, i] * x[:, j] for i, j in indices]
    return numpy.stack(products, axis=1)


def dictionary_sin_cos(x, n_harmonics):
    """
    returns sine and cosine harmonics of input features up to n_harmonics
    """
    sin_cos = []
    for k in range(1, n_harmonics + 1):
        sin_cos.append(numpy.sin(k * x))
        sin_cos.append(numpy.cos(k * x)) 

    return numpy.concatenate(sin_cos, axis=1)

def dictionary_sin_cos_cross(x):
    """
    returns sin_cos cross terms : x_a * sin(x_b) and x_a * cos(x_b)
    for all unique pairs (a != b), common in sin_cos dynamical systems

    Parameters:
    - x: np.ndarray of shape (batch_size, num_features)

    Returns:
    - np.ndarray of shape (batch_size, num_features * (num_features - 1) * 2)
      columns ordered as:
      x0*sin(x1), x0*cos(x1), x0*sin(x2), x0*cos(x2), ..., x1*sin(x0), x1*cos(x0), ...
    """

    num_features = x.shape[1]
    terms = []
    for a in range(num_features):
        for b in range(num_features):
            if a != b:
                terms.append(x[:, a:a+1] * numpy.sin(x[:, b:b+1]))
                terms.append(x[:, a:a+1] * numpy.cos(x[:, b:b+1]))

    return numpy.concatenate(terms, axis=1)

