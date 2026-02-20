import numpy as np

def ransac_estimate_A_B(x, u, sample_multiplier=1.0, n_iter=1000, tol=1e-3):
    """
    RANSAC estimation of A, B in x_{n+1} = A x_n + B u_n.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_states)
        State sequence.
    u : ndarray, shape (n_samples, n_inputs)
        Control sequence.
    sample_multiplier : float
        Multiple of minimal sample size per hypothesis (>=1.0 recommended).
    n_iter : int
        Number of RANSAC iterations.
    tol : float
        Inlier residual threshold.

    Returns
    -------
    A_best, B_best : ndarray
        Estimated system matrices.
    inlier_mask : ndarray of bool
        Mask of inlier samples.
    """
    X_curr = x[:-1]
    X_next = x[1:]
    U_curr = u[:-1]
    Z = np.hstack([X_curr, U_curr])

    n_states = x.shape[1]
    n_inputs = u.shape[1]
    min_samples = n_states + n_inputs
    sample_size = int(np.ceil(sample_multiplier * min_samples))

    best_inliers = 0
    A_best, B_best = None, None
    inlier_mask_best = None

    rng = np.random.default_rng()

    for _ in range(n_iter):
        # Random sample without replacement
        idx = rng.choice(Z.shape[0], size=sample_size, replace=False)
        Z_sample = Z[idx]
        X_next_sample = X_next[idx]

        # Check rank to avoid singular systems
        if np.linalg.matrix_rank(Z_sample) < min_samples:
            continue

        # Fit model on sample
        M, _, _, _ = np.linalg.lstsq(Z_sample, X_next_sample, rcond=None)

        # Compute residuals on all data
        residuals = np.linalg.norm(X_next - Z @ M, axis=1)
        inliers = residuals < tol
        n_inliers = np.sum(inliers)

        # Update best model
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            inlier_mask_best = inliers

            # Refit on all inliers
            M_best, _, _, _ = np.linalg.lstsq(Z[inliers], X_next[inliers], rcond=None)
            A_best = M_best[:n_states].T
            B_best = M_best[n_states:].T

    return A_best, B_best, inlier_mask_best
