import numpy

from .detection_evaluation import detection_evaluation


def auc_roc(y_gt, y_scores):
    """
    Compute the Area Under the Receiver Operating Characteristic curve.

    The ROC curve plots True Positive Rate (recall) against False Positive
    Rate at every possible score threshold.  AUC-ROC summarises detection
    performance across all thresholds in a single number ∈ [0, 1].

    Parameters
    ----------
    y_gt : array-like
        Binary ground truth labels (0 = normal, 1 = anomaly).
    y_scores : array-like
        Continuous anomaly scores (higher = more anomalous).

    Returns
    -------
    float
        AUC-ROC value.
    """
    y_gt     = numpy.asarray(y_gt, dtype=numpy.int32)
    y_scores = numpy.asarray(y_scores, dtype=numpy.float64)

    # Sort samples by descending score so that the most anomalous come first
    desc_order  = numpy.argsort(-y_scores)
    y_gt_sorted = y_gt[desc_order]

    # Cumulative true-positive and false-positive counts along the ranking
    cum_tp = numpy.cumsum(y_gt_sorted)
    cum_fp = numpy.cumsum(1 - y_gt_sorted)

    total_pos = numpy.sum(y_gt)
    total_neg = len(y_gt) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.0

    # Convert counts to rates
    tpr = cum_tp / total_pos          # recall at each threshold
    fpr = cum_fp / total_neg          # false-positive rate

    # Integrate using the trapezoidal rule
    auc = float(numpy.trapezoid(tpr, fpr))
    return auc


def auc_pr(y_gt, y_scores):
    """
    Compute the Area Under the Precision-Recall curve.

    AUC-PR is more informative than AUC-ROC for highly imbalanced datasets
    (which is the typical case in anomaly detection) because it focuses on
    the positive (anomaly) class.

    Parameters
    ----------
    y_gt : array-like
        Binary ground truth labels (0 = normal, 1 = anomaly).
    y_scores : array-like
        Continuous anomaly scores (higher = more anomalous).

    Returns
    -------
    float
        AUC-PR value.
    """
    y_gt     = numpy.asarray(y_gt, dtype=numpy.int32)
    y_scores = numpy.asarray(y_scores, dtype=numpy.float64)

    # Sort by descending score
    desc_order  = numpy.argsort(-y_scores)
    y_gt_sorted = y_gt[desc_order]

    # Cumulative true positives at each rank position
    cum_tp = numpy.cumsum(y_gt_sorted)
    total_pos = numpy.sum(y_gt)

    if total_pos == 0:
        return 0.0

    # Precision and recall at each rank position
    positions = numpy.arange(1, len(y_gt) + 1)
    precision = cum_tp / positions
    recall    = cum_tp / total_pos

    # Integrate using the trapezoidal rule
    auc = float(numpy.trapezoid(precision, recall))
    return auc


def anomaly_evaluation(y_gt, y_scores, th=0.5):
    """
    Evaluate anomaly detection predictions.

    Combines threshold-independent metrics (AUC-ROC, AUC-PR, score
    distribution statistics) with threshold-dependent binary metrics
    obtained via :func:`detection_evaluation`.  This avoids duplicating
    confusion-matrix logic while adding anomaly-specific diagnostics.

    Parameters
    ----------
    y_gt : array-like
        Ground truth labels (0 = normal, 1 = anomaly).  Continuous values
        are binarised internally using *th*.
    y_scores : array-like
        Continuous anomaly scores produced by the detector
        (higher = more anomalous).
    th : float, optional
        Threshold for binarising both *y_gt* and *y_scores*
        (value > th → anomaly).  Default is 0.5.

    Returns
    -------
    dict
        JSON-serialisable dictionary with the following groups of metrics:

        **Sample counts**
        - ``n_samples``      - total number of samples
        - ``n_anomalies``    - anomalies in ground truth (after binarisation)
        - ``n_normal``       - normal samples in ground truth
        - ``anomaly_ratio``  - fraction of anomalies (class imbalance indicator)

        **Threshold-independent metrics**
        - ``auc_roc``  - Area Under ROC Curve
        - ``auc_pr``   - Area Under Precision-Recall Curve

        **Score distribution**
        - ``score_mean``, ``score_std``       - overall score statistics
        - ``score_mean_normal``               - mean score for normal samples
        - ``score_mean_anomaly``              - mean score for anomaly samples
        - ``score_separation``                - difference between anomaly and
          normal mean scores (larger = better separation)

        **Threshold-dependent binary metrics**  (from detection_evaluation)
        - ``threshold``, ``accuracy``, ``precision``, ``recall``,
          ``f1_score``, ``mcc``, ``specificity``, ``balanced_accuracy``,
          ``iou``, ``dice``, ``tp``, ``tn``, ``fp``, ``fn``

        **Additional anomaly-specific rates**
        - ``fpr``  - False Positive Rate  (false alarm rate)
        - ``fnr``  - False Negative Rate  (miss rate)
    """
    y_gt     = numpy.asarray(y_gt, dtype=numpy.float64)
    y_scores = numpy.asarray(y_scores, dtype=numpy.float64)

    if y_gt.shape != y_scores.shape:
        raise ValueError(
            "shapes are not matching " + str(y_gt.shape) +
            " must match " + str(y_scores.shape)
        )

    n = int(y_gt.shape[0])

    # ---- threshold-independent metrics ----
    # Computed on the raw continuous scores before binarisation
    roc = auc_roc(y_gt, y_scores)
    pr  = auc_pr(y_gt, y_scores)

    # ---- score distribution statistics ----
    # Useful for understanding how well the detector separates the two classes
    y_gt_bin = numpy.array(y_gt > th, dtype=int)

    score_mean = float(numpy.mean(y_scores))
    score_std  = float(numpy.std(y_scores, ddof=1)) if n > 1 else 0.0

    normal_mask  = (y_gt_bin == 0)
    anomaly_mask = (y_gt_bin == 1)

    score_mean_normal  = float(numpy.mean(y_scores[normal_mask]))  if numpy.any(normal_mask)  else 0.0
    score_mean_anomaly = float(numpy.mean(y_scores[anomaly_mask])) if numpy.any(anomaly_mask) else 0.0

    # How far apart the two class means are (higher = easier to threshold)
    score_separation = score_mean_anomaly - score_mean_normal

    n_anomalies = int(numpy.sum(anomaly_mask))
    n_normal    = int(numpy.sum(normal_mask))
    anomaly_ratio = n_anomalies / n if n > 0 else 0.0

    # ---- threshold-dependent binary metrics ----
    # Delegate to detection_evaluation to avoid duplicating confusion-matrix code
    det = detection_evaluation(y_gt, y_scores, th=th)

    tp = det["tp"]
    tn = det["tn"]
    fp = det["fp"]
    fn = det["fn"]

    # Anomaly-specific error rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0    # false alarm rate
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0    # miss rate

    # ---- build result ----
    result = {
        # sample counts
        "n_samples"          : n,
        "n_anomalies"        : n_anomalies,
        "n_normal"           : n_normal,
        "anomaly_ratio"      : round(float(anomaly_ratio), 5),

        # threshold-independent
        "auc_roc"            : round(float(roc), 5),
        "auc_pr"             : round(float(pr), 5),

        # score distribution
        "score_mean"         : round(float(score_mean), 5),
        "score_std"          : round(float(score_std), 5),
        "score_mean_normal"  : round(float(score_mean_normal), 5),
        "score_mean_anomaly" : round(float(score_mean_anomaly), 5),
        "score_separation"   : round(float(score_separation), 5),

        # threshold-dependent (from detection_evaluation)
        "threshold"          : det["threshold"],
        "accuracy"           : det["accuracy"],
        "precision"          : det["precision"],
        "recall"             : det["recall"],
        "specificity"        : det["specificity"],
        "f1_score"           : det["f1_score"],
        "mcc"                : det["mcc"],
        "balanced_accuracy"  : det["balanced_accuracy"],
        "iou"                : det["iou"],
        "dice"               : det["dice"],

        # anomaly-specific error rates
        "fpr"                : round(float(fpr), 5),
        "fnr"                : round(float(fnr), 5),

        # raw confusion matrix
        "tp"                 : tp,
        "tn"                 : tn,
        "fp"                 : fp,
        "fn"                 : fn,
    }

    return result