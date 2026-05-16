import torch

def segmentation_metrics(y_gt, y_pred, threshold=0.5, eps=1e-7):
    """
    y_gt:   (B, H, W) binary {0,1}
    y_pred: (B, H, W) float [0,1] (sigmoid output)

    Returns dict of metrics averaged over batch
    """

    y_pred_bin = (y_pred >= threshold).float()

    # flatten
    y_gt = y_gt.view(y_gt.size(0), -1)
    y_pred_bin = y_pred_bin.view(y_pred_bin.size(0), -1)

    TP = (y_gt * y_pred_bin).sum(dim=1)
    FP = ((1 - y_gt) * y_pred_bin).sum(dim=1)
    FN = (y_gt * (1 - y_pred_bin)).sum(dim=1)
    TN = ((1 - y_gt) * (1 - y_pred_bin)).sum(dim=1)

    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = TP / (TP + FP + FN + eps)

    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)

    return {
        "iou": iou.mean().item(),
        "f1": f1.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "accuracy": accuracy.mean().item(),
    }