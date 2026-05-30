import numpy

def keypoints_evaluation(points_binary, points_blur, points_pred, th=0.5):
    """
    Evaluate keypoint predictions using distance-aware and tolerance metrics.
    Avoids Python loops by leveraging element-wise NumPy operations.

    Parameters
    ----------
    points_binary : array-like
        Sparse binary mask, exact ground truth coordinates (0 or 1).
    points_blur : array-like
        Gaussian blurred ground truth serving as a spatial tolerance mask [0, 1].
    points_pred : array-like
        Model predictions after sigmoid activation [0, 1].
    th : float
        Threshold to binarise the model's predictions.

    Returns
    -------
    dict
    """
    # Ensure inputs are numpy arrays and flattened/cast consistently if needed
    y_true_sp = numpy.asarray(points_binary, dtype=numpy.float64)
    y_true_bl = numpy.asarray(points_blur, dtype=numpy.float64)
    y_pred    = numpy.asarray(points_pred, dtype=numpy.float64)

    if not (y_true_sp.shape == y_true_bl.shape == y_pred.shape):
        raise ValueError(f"Shape mismatch! All inputs must have the same shape. Got: {y_true_sp.shape}, {y_true_bl.shape}, {y_pred.shape}")

    # Binarise model predictions based on confidence threshold
    y_pred_bin = (y_pred > th).astype(int)
    
    # Total number of true keypoints across the whole batch
    total_true_keypoints = int(numpy.sum(y_true_sp == 1))

    # 1. ---- Exact Peak Confidence ----
    # What is the model's raw continuous prediction exactly at the true coordinate?
    # Perfect score = 1.0
    if total_true_keypoints > 0:
        mean_peak_confidence = numpy.sum(y_pred * (y_true_sp == 1)) / total_true_keypoints
    else:
        mean_peak_confidence = 0.0

    # 2. ---- Distance-Aware Confusion Matrix (Using the Blur Mask) ----
    # A prediction is a "True Positive" if it's over the threshold AND lands inside the blur radius.
    # The `y_true_bl` acts as a continuous weight, but we can threshold it to define the "tolerance zone".
    # We use a small threshold (e.g., > 0.1) to define the boundaries of the blur blob.
    tolerance_zone = (y_true_bl > 0.1).astype(int)
    
    # Hit: Predicted positive inside the tolerance blob
    tp_spatial = numpy.sum((y_pred_bin == 1) & (tolerance_zone == 1))
    
    # Miss: Predicted positive completely outside any tolerance blob
    fp_spatial = numpy.sum((y_pred_bin == 1) & (tolerance_zone == 0))
    
    # Ignored / Left out: True keypoint locations where the model predicted 0
    fn_spatial = numpy.sum((y_true_sp == 1) & (y_pred_bin == 0))

    # 3. ---- Spatial Metrics Calculation ----
    precision_spatial = tp_spatial / (tp_spatial + fp_spatial) if (tp_spatial + fp_spatial) > 0 else 0.0
    recall_spatial    = tp_spatial / (total_true_keypoints) if total_true_keypoints > 0 else 0.0
    f1_spatial        = (2 * precision_spatial * recall_spatial) / (precision_spatial + recall_spatial) if (precision_spatial + recall_spatial) > 0 else 0.0

    # 4. ---- Mean Absolute Error (MAE) Restricted to Keypoint Regions ----
    # Standard MAE over the whole image is dominated by background zeros. 
    # Let's measure MAE ONLY inside the blur blobs to see how well the gradient shape matches.
    blob_mask = tolerance_zone == 1
    if numpy.sum(blob_mask) > 0:
        blob_mae = numpy.mean(numpy.abs(y_pred[blob_mask] - y_true_bl[blob_mask]))
    else:
        blob_mae = 0.0

    return {
        "total_true_keypoints": total_true_keypoints,
        "mean_peak_confidence": round(float(mean_peak_confidence), 5),
        "spatial_precision"   : round(float(precision_spatial), 5),
        "spatial_recall"      : round(float(recall_spatial), 5),
        "spatial_f1_score"    : round(float(f1_spatial), 5),
        "blob_region_mae"     : round(float(blob_mae), 5),
        "spatial_tp_pixels"   : int(tp_spatial),
        "spatial_fp_pixels"   : int(fp_spatial),
        "spatial_fn_points"   : int(fn_spatial)
    }