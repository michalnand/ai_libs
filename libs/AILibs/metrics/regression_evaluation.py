import numpy


def regression_evaluation(y_gt, y_pred, n_features=None):
    """
    Evaluate regression predictions against ground truth.

    Parameters
    ----------
    y_gt : array-like
        Ground truth continuous values.
    y_pred : array-like
        Predicted continuous values.
    n_features : int, optional
        Number of input features, used for adjusted R².
        If None, adjusted_r2 is omitted from the result.

    Returns
    -------
    dict
        JSON-serialisable dictionary with regression metrics.
    """
    y_gt   = numpy.asarray(y_gt, dtype=numpy.float32)
    y_pred = numpy.asarray(y_pred, dtype=numpy.float32)

    if y_gt.shape != y_pred.shape:
        raise Exception(
            "shapes are not matching " + str(y_gt.shape) +
            " must match " + str(y_pred.shape)
        )

    n = len(y_gt)

    # ---- residuals ----
    residuals     = y_gt - y_pred
    abs_residuals = numpy.abs(residuals)

    # ---- core error metrics ----
    mse   = numpy.mean(residuals ** 2)
    rmse  = numpy.sqrt(mse)
    mae   = numpy.mean(abs_residuals)
    medae = numpy.median(abs_residuals)
    max_ae = numpy.max(abs_residuals)

    # ---- relative metrics ----
    # MAPE – guard against zero ground-truth values
    non_zero_mask = numpy.abs(y_gt) > 1e-12
    if numpy.any(non_zero_mask):
        mape = numpy.mean(abs_residuals[non_zero_mask] / numpy.abs(y_gt[non_zero_mask])) * 100.0
    else:
        mape = float("inf")

    # R² (coefficient of determination)
    ss_res = numpy.sum(residuals ** 2)
    ss_tot = numpy.sum((y_gt - numpy.mean(y_gt)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Adjusted R²
    adjusted_r2 = None
    if n_features is not None and n > n_features + 1:
        adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1)

    # ---- residual distribution ----
    residual_mean = numpy.mean(residuals)
    residual_std  = numpy.std(residuals, ddof=1) if n > 1 else 0.0

    # ---- sigma-interval analysis ----
    # MSE of residuals falling within each sigma band
    sigma_mse = {}
    for k in (1, 2, 3):
        mask = abs_residuals <= k * residual_std
        if residual_std > 0 and numpy.any(mask):
            sigma_mse[k] = float(numpy.mean(residuals[mask] ** 2))
        else:
            sigma_mse[k] = float(mse)

    # ---- build result ----
    result = {
        "n_samples"      : int(n),
        "n_features"     : int(y_gt.shape[-1]),
        "mse"            : round(float(mse), 5),
        "rmse"           : round(float(rmse), 5),
        "mae"            : round(float(mae), 5),
        "medae"          : round(float(medae), 5),
        "max_ae"         : round(float(max_ae), 5),
        "mape"           : round(float(mape), 5),
        "r2"             : round(float(r2), 5),
        "residual_mean"  : round(float(residual_mean), 5),
        "residual_std"   : round(float(residual_std), 5),
        "mse_1sigma"     : round(float(sigma_mse[1]), 5),
        "mse_2sigma"     : round(float(sigma_mse[2]), 5),
        "mse_3sigma"     : round(float(sigma_mse[3]), 5),
    }   

    if adjusted_r2 is not None:
        result["adjusted_r2"] = round(float(adjusted_r2), 5)

    return result
