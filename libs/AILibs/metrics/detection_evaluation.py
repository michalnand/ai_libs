import numpy


def detection_evaluation(y_gt, y_pred, th=0.5):
    """
    Evaluate binary classification / detection predictions.

    Parameters
    ----------
    y_gt : array-like
        Ground truth values (continuous or binary).
    y_pred : array-like
        Predicted values (continuous or binary).
    th : float
        Threshold for binarising both arrays (value > th → 1).

    Returns
    -------
    dict
        JSON-serialisable dictionary with detection metrics.
    """
    y_gt   = numpy.asarray(y_gt, dtype=numpy.float64)
    y_pred = numpy.asarray(y_pred, dtype=numpy.float64)

    # shape check must precede thresholding to catch mismatches early
    if y_gt.shape != y_pred.shape:
        raise Exception(
            "shapes are not matching " + str(y_gt.shape) +
            " must match " + str(y_pred.shape)
        )

    n = int(y_gt.shape[0])

    # binarise using threshold
    y_gt   = numpy.array((y_gt > th), dtype=int)
    y_pred = numpy.array((y_pred > th), dtype=int)

    # ---- confusion matrix ----
    TP = numpy.sum((y_gt == 1) & (y_pred == 1))
    TN = numpy.sum((y_gt == 0) & (y_pred == 0))
    FP = numpy.sum((y_gt == 0) & (y_pred == 1))
    FN = numpy.sum((y_gt == 1) & (y_pred == 0))

    # ---- basic metrics ----
    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0       # sensitivity / TPR
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ---- Matthews Correlation Coefficient ----
    # ranges [-1, 1]; 0 = random, 1 = perfect, -1 = total disagreement
    numerator   = TP * TN - FP * FN
    denominator = numpy.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    mcc = numerator / denominator if denominator > 0 else 0.0

    # ---- specificity & balanced accuracy ----
    specificity  = TN / (TN + FP) if (TN + FP) > 0 else 0.0    # TNR
    balanced_acc = 0.5 * (recall + specificity)

    # ---- segmentation / overlap metrics ----
    # IoU (Jaccard index) and Dice (F1 over areas)
    iou  = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0

    return {
        "n_samples"         : n,
        "threshold"         : round(float(th), 5),
        "accuracy"          : round(float(accuracy), 5),
        "precision"         : round(float(precision), 5),
        "recall"            : round(float(recall), 5),
        "f1_score"          : round(float(f1), 5),
        "mcc"               : round(float(mcc), 5),
        "specificity"       : round(float(specificity), 5),
        "balanced_accuracy" : round(float(balanced_acc), 5),
        "iou"               : round(float(iou), 5),
        "dice"              : round(float(dice), 5),
        "tp"                : int(TP),
        "tn"                : int(TN),
        "fp"                : int(FP),
        "fn"                : int(FN),
    }