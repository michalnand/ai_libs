import numpy 


def tune_threshold(y_gt, y_scores, metric="f1", steps=100):
    """
    Find the optimal binarisation threshold for anomaly scores.

    Sweeps thresholds uniformly from 0 to 1 and returns the one that
    maximises the chosen metric.  Useful when a detector outputs
    continuous scores and you need to pick a decision boundary.

    Parameters
    ----------
    y_gt : array-like
        Binary ground truth labels (0 = normal, 1 = anomaly).
    y_scores : array-like
        Continuous anomaly scores (higher = more anomalous).
    metric : str, optional
        Metric to maximise.  One of:
        - ``'f1'``      – F1 score (harmonic mean of precision and recall).
        - ``'mcc'``     – Matthews Correlation Coefficient (balanced metric
          that accounts for all four confusion-matrix cells).
        - ``'youden'``  – Youden's J statistic (recall − FPR), equivalent to
          maximising the vertical distance to the ROC diagonal.
    steps : int, optional
        Number of equally-spaced thresholds to evaluate (default 100).

    Returns
    -------
    float
        Threshold value (rounded to 5 decimal places) that maximises the
        chosen metric.
    """
    y_gt     = numpy.asarray(y_gt, dtype=numpy.int32)
    y_scores = numpy.asarray(y_scores, dtype=numpy.float64)

    best_th  = 0.5
    best_val = -1.0

    for th in numpy.linspace(0.0, 1.0, steps):
        # Binarise predictions at the current threshold
        y_pred = (y_scores > th).astype(int)

        # Confusion matrix
        TP = numpy.sum((y_gt == 1) & (y_pred == 1))
        FP = numpy.sum((y_gt == 0) & (y_pred == 1))
        FN = numpy.sum((y_gt == 1) & (y_pred == 0))
        TN = numpy.sum((y_gt == 0) & (y_pred == 0))

        # Compute the requested metric
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        mcc_num = TP * TN - FP * FN
        mcc_den = numpy.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0

        # Youden's J = sensitivity − FPR  (optimal ROC operating point)
        fpr    = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        youden = recall - fpr

        val = {"f1": f1, "mcc": mcc, "youden": youden}[metric]

        if val > best_val:
            best_val = val
            best_th  = th

    return round(float(best_th), 5)