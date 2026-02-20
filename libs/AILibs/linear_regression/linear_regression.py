import numpy


def lr_fit(X, Y):
    """
        basic linear regression,
        for data X, Y, fits a linear regression model Y = XA
        
        :param X: numpy array, shape (n_samples, n_inputs)
        :param Y: numpy array, shape (n_samples, n_ouputs)
    """
    
    A = numpy.linalg.lstsq(X, Y, rcond=None)[0]
    
    return A




def lr_sparse_fit(X, Y, density=0.01, n_iter=100, rel_tol=1e-6):
    """
        sparse linear regression using iterative thresholding,
        for data X, Y, fits a linear regression model Y = XA with a sparse A
        
        in each iteration, fits regression on the residuum and adds
        the top-k largest magnitude coefficients (at positions not yet selected),
        where k = max(1, round(density * n_inputs * n_outputs))

        stops early when relative error improvement falls below rel_tol

        :param X: numpy array, shape (n_samples, n_inputs)
        :param Y: numpy array, shape (n_samples, n_ouputs)
        :param density: float in (0, 1), fraction of non-zero coefficients added per iteration
        :param n_iter: int, maximum number of iterations
        :param rel_tol: float, relative improvement threshold for early stopping
    """

    n_samples, n_inputs = X.shape
    _, n_outputs        = Y.shape

    # number of elements to add per iteration (at least one)
    total_elements = n_inputs * n_outputs
    k = max(1, round(density * total_elements))

    A_result = numpy.zeros((n_inputs, n_outputs))

    # mask of already selected (non-zero) positions
    selected = numpy.zeros((n_inputs, n_outputs), dtype=bool)

    residuum  = Y.copy()
    err_prev  = (residuum**2).mean()

    for n in range(n_iter):
        # fit linear regression to current residuum
        A = lr_fit(X, residuum)

        # zero out already selected positions so we only pick new ones
        A[selected] = 0.0

        # select top-k largest magnitude coefficients from remaining
        abs_A = numpy.abs(A)
        flat_indices = numpy.argsort(abs_A, axis=None)[::-1][:k]
        new_positions = numpy.unravel_index(flat_indices, A.shape)

        # mark new positions as selected and add their values
        selected[new_positions] = True
        A_result[new_positions] += A[new_positions]

        # update residuum
        residuum = Y - X @ A_result

        err = (residuum**2).mean()

        # early stopping: relative improvement is negligible
        if err_prev > 0.0 and (err_prev - err) / err_prev < rel_tol:
            break

        err_prev = err

    return A_result
