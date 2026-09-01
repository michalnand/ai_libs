import numpy 


def tune_threshold(y_gt: numpy.ndarray, y_pred: numpy.ndarray, steps: int = 100, beta : float = 1.0):

    thresholds = numpy.linspace(y_pred.min(), y_pred.max(), steps) 

    
    y_gt_tmp = y_gt > 0.5 

    f_scores = []
    for threshold in thresholds:
        preds = y_pred >= threshold

        #print(">>> ", y_pred.shape, y_gt_tmp.shape, preds.shape)

        TP = (preds & y_gt_tmp).sum()
        FP = (preds & ~y_gt_tmp).sum() 
        FN = (~preds & y_gt_tmp).sum()

        precission  = TP/(TP + FP)
        recall      = TP/(TP + FN)


        num = (1 + beta**2) * (precission * recall)
        den = (beta**2 * precission) + recall

        f1 = num/(den + 10e-10)

        f_scores.append(f1)

    f_scores = numpy.array(f_scores)
    best_idx = numpy.argmax(f_scores)

    threshold = round(thresholds[best_idx], 4) 

    return threshold



def tune_threshold_OLD(y_gt: numpy.ndarray, y_pred: numpy.ndarray, beta: float = 1.0, steps: int = 100):
    """
    Tunes anomaly threshold using F-beta score (beta=1.0 is F1).    
    Evaluates 'steps' evenly spaced thresholds across the y_pred range.
    """
    y_gt = numpy.asarray(y_gt, dtype=bool)
    
    # Generate thresholds and reshape to (steps, 1) for broadcasting
    thresholds = numpy.linspace(y_pred.min(), y_pred.max(), steps)[:, None]
    
    # preds matrix shape: (steps, N)
    preds = y_pred >= thresholds
    
    # Vectorized TP, FP, FN calculations across all thresholds
    TP = (preds & y_gt).sum(axis=1)
    FP = (preds & ~y_gt).sum(axis=1)
    FN = (~preds & y_gt).sum(axis=1)
    
    # Safe Precision and Recall (avoids division by zero)
    P = numpy.divide(TP, TP + FP, out=numpy.zeros(steps), where=(TP + FP) > 0)
    R = numpy.divide(TP, TP + FN, out=numpy.zeros(steps), where=(TP + FN) > 0)
    
    # Safe F-beta Score
    num = (1 + beta**2) * (P * R)
    den = (beta**2 * P) + R
    f_scores = numpy.divide(num, den, out=numpy.zeros(steps), where=den > 0)
    
    # Extract best threshold and score
    best_idx = numpy.argmax(f_scores)
    
    return thresholds[best_idx, 0] #, f_scores[best_idx]