import torch

def keypoint_metrics(y_gt, y_pred, threshold=0.5, eps=1e-7):
    """
    Designed for sparse binary targets (keypoints)

    Returns both threshold-based and threshold-free metrics
    """

    # binarize
    y_pred_bin = (y_pred >= threshold).float()

    # flatten everything
    y_gt = y_gt.view(-1)
    y_pred = y_pred.view(-1)
    y_pred_bin = y_pred_bin.view(-1)

    TP = (y_gt * y_pred_bin).sum()
    FP = ((1 - y_gt) * y_pred_bin).sum()
    FN = (y_gt * (1 - y_pred_bin)).sum()

    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # ---- Average Precision (threshold independent) ----
    # sort by confidence
    sorted_idx = torch.argsort(y_pred, descending=True)
    y_gt_sorted = y_gt[sorted_idx]

    tp_cumsum = torch.cumsum(y_gt_sorted, dim=0)
    fp_cumsum = torch.cumsum(1 - y_gt_sorted, dim=0)

    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
    recall_curve = tp_cumsum / (y_gt.sum() + eps)

    # numerical integration
    ap = torch.trapz(precision_curve, recall_curve)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "ap": ap.item(),
    }