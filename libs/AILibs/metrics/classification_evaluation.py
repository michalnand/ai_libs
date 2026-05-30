import numpy


def classification_evaluation(y_gt, y_pred, num_classes):
    """
    Evaluate multiclass classification predictions.

    Parameters
    ----------
    y_gt : array-like
        Ground truth class indices, shape (num_samples,).
    y_pred : array-like
        Predicted logits or probabilities, shape (num_samples, num_classes).
    num_classes : int
        The total number of classes.

    Returns
    -------
    dict
        JSON-serialisable dictionary with macro-averaged classification metrics
        and per-class breakdowns.
    """
    y_gt = numpy.asarray(y_gt, dtype=numpy.int64)
    y_pred = numpy.asarray(y_pred, dtype=numpy.float64)

    # Validate shapes: y_pred should be (N, num_classes) and y_gt should be (N,)
    if y_pred.ndim != 2 or y_pred.shape[1] != num_classes:
        raise ValueError(f"y_pred shape {y_pred.shape} must be (num_samples, {num_classes})")
    if y_gt.shape != (y_pred.shape[0],):
        raise ValueError(f"y_gt shape {y_gt.shape} must match y_pred samples ({y_pred.shape[0]},)")

    n = int(y_gt.shape[0])

    # Convert logits to class predictions via argmax
    y_pred_indices = numpy.argmax(y_pred, axis=1)

    # Initialize confusion components arrays for each class
    tp_per_class = numpy.zeros(num_classes, dtype=int)
    fp_per_class = numpy.zeros(num_classes, dtype=int)
    fn_per_class = numpy.zeros(num_classes, dtype=int)
    tn_per_class = numpy.zeros(num_classes, dtype=int)

    # Calculate confusion matrix components per class
    for c in range(num_classes):
        gt_c = (y_gt == c)
        pred_c = (y_pred_indices == c)

        tp_per_class[c] = numpy.sum(gt_c & pred_c)
        fp_per_class[c] = numpy.sum(~gt_c & pred_c)
        fn_per_class[c] = numpy.sum(gt_c & ~pred_c)
        tn_per_class[c] = numpy.sum(~gt_c & ~pred_c)

    # ---- Global Metric ----
    accuracy = numpy.sum(tp_per_class) / n if n > 0 else 0.0

    # ---- Per-Class Metrics Setup ----
    precision_per_class = numpy.zeros(num_classes)
    recall_per_class = numpy.zeros(num_classes)
    f1_per_class = numpy.zeros(num_classes)
    mcc_per_class = numpy.zeros(num_classes)
    specificity_per_class = numpy.zeros(num_classes)
    iou_per_class = numpy.zeros(num_classes)
    dice_per_class = numpy.zeros(num_classes)


    for c in range(num_classes):
        tp, fp, fn, tn = tp_per_class[c], fp_per_class[c], fn_per_class[c], tn_per_class[c]

        precision_per_class[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_per_class[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        p, r = precision_per_class[c], recall_per_class[c]
        f1_per_class[c] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        # Matthews Correlation Coefficient per class
        numerator = float(tp * tn - fp * fn)
        denominator = numpy.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc_per_class[c] = numerator / denominator if denominator > 0 else 0.0

        specificity_per_class[c] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        iou_per_class[c] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice_per_class[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0


    # ---- Par class counts ---
    class_counts = numpy.zeros(num_classes, dtype=int)
    for c in range(num_classes):
        class_counts[c] = numpy.sum(y_gt == c)
    
    # ---- Macro-Averaging ----
    macro_precision = numpy.mean(precision_per_class)
    macro_recall = numpy.mean(recall_per_class)
    macro_f1 = numpy.mean(f1_per_class)
    macro_mcc = numpy.mean(mcc_per_class)
    macro_specificity = numpy.mean(specificity_per_class)
    macro_balanced_acc = 0.5 * (macro_recall + macro_specificity)
    macro_iou = numpy.mean(iou_per_class)
    macro_dice = numpy.mean(dice_per_class)

    return {
        "n_samples"         : n,
        "num_classes"       : num_classes,
        "accuracy"          : round(float(accuracy), 5),
        "macro_precision"   : round(float(macro_precision), 5),
        "macro_recall"      : round(float(macro_recall), 5),
        "macro_f1_score"    : round(float(macro_f1), 5),
        "macro_mcc"         : round(float(macro_mcc), 5),
        "macro_specificity" : round(float(macro_specificity), 5),
        "macro_balanced_accuracy": round(float(macro_balanced_acc), 5),
        "macro_iou"         : round(float(macro_iou), 5),
        "macro_dice"        : round(float(macro_dice), 5),
        "tp_per_class"      : tp_per_class.tolist(),
        "tn_per_class"      : tn_per_class.tolist(),
        "fp_per_class"      : fp_per_class.tolist(),
        "fn_per_class"      : fn_per_class.tolist(),
        "class_counts"      : class_counts.tolist(),    
        # New meaningful per-class metrics
        "precision_per_class": [round(float(x), 5) for x in precision_per_class],
        "recall_per_class"   : [round(float(x), 5) for x in recall_per_class],
        "f1_score_per_class" : [round(float(x), 5) for x in f1_per_class],
    }