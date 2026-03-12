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


def lr_lasso_fit(X, Y, lambda_=1.0, n_iter=10000, rel_tol=1e-6):
    """
        linear regression with L1 regularization (Lasso),
        for data X, Y, fits a linear regression model Y = XA with a sparse A

        uses coordinate descent with soft-thresholding (pure numpy, no sklearn)

        :param X: numpy array, shape (n_samples, n_inputs)
        :param Y: numpy array, shape (n_samples, n_ouputs)
        :param lambda_: float, regularization parameter controlling sparsity
        :param n_iter: int, maximum number of coordinate descent iterations
        :param rel_tol: float, relative improvement threshold for early stopping
    """

    n_samples, n_inputs = X.shape
    _, n_outputs        = Y.shape

    A_result = numpy.zeros((n_inputs, n_outputs))

    # precompute column norms (squared) for normalisation
    col_norms_sq = (X ** 2).sum(axis=0)                # shape (n_inputs,)

    for col in range(n_outputs):
        y = Y[:, col]
        w = numpy.zeros(n_inputs)
        residual = y.copy()

        for iteration in range(n_iter):
            w_old = w.copy()

            for j in range(n_inputs):
                # temporarily add back j-th feature contribution
                residual += X[:, j] * w[j]

                # compute un-regularised optimum for j-th coordinate
                rho_j = X[:, j] @ residual

                # soft-thresholding
                if col_norms_sq[j] == 0.0:
                    w[j] = 0.0
                else:
                    w[j] = numpy.sign(rho_j) * max(abs(rho_j) - lambda_ * n_samples, 0.0) / col_norms_sq[j]

                # update residual with new weight
                residual -= X[:, j] * w[j]

            # convergence check
            dw = numpy.linalg.norm(w - w_old)
            if dw < rel_tol * (numpy.linalg.norm(w) + 1e-30):
                break

        A_result[:, col] = w

    return A_result


# Sparse Relaxed Regularized Regression — SR3
def sr3_fit(X, Y, lambda_=0.01, rho=1.0, n_iter=100, rel_tol=1e-6):
    """
        sparse linear regression using SR3 algorithm,
        for data X, Y, fits a linear regression model Y = XA with a sparse A
        
        :param X: numpy array, shape (n_samples, n_inputs)
        :param Y: numpy array, shape (n_samples, n_ouputs)
        :param lambda_: float, regularization parameter controlling sparsity
        :param rho: float, penalty parameter for the augmented Lagrangian
        :param n_iter: int, maximum number of iterations
        :param rel_tol: float, relative improvement threshold for early stopping
    """

    n_samples, n_inputs = X.shape 
    _, n_outputs        = Y.shape

    A = numpy.zeros((n_inputs, n_outputs))
    Z = numpy.zeros_like(A)
    U = numpy.zeros_like(A)

    XtX = X.T @ X
    XtY = X.T @ Y

    for n in range(n_iter):
        # update A by solving least squares with augmented term
        A = numpy.linalg.lstsq(XtX + rho * numpy.eye(n_inputs), XtY + rho * (Z - U), rcond=None)[0]

        # update Z by applying soft thresholding to A + U
        Z_prev = Z.copy()
        Z = numpy.sign(A + U) * numpy.maximum(numpy.abs(A + U) - lambda_ / rho, 0)

        # update dual variable U
        U += A - Z

        # check ADMM convergence: primal and dual residuals
        primal_res = numpy.linalg.norm(A - Z)
        dual_res   = rho * numpy.linalg.norm(Z - Z_prev)

        if primal_res < rel_tol and dual_res < rel_tol:
            break

    # hard-threshold to remove numerical dust
    # any coefficient much smaller than the soft-threshold level lambda_/rho
    # is residual noise from ADMM, not a true sparse component
    threshold = 0.1 * lambda_ / rho
    Z[numpy.abs(Z) < threshold] = 0.0

    # return Z — the sparse solution
    return Z   