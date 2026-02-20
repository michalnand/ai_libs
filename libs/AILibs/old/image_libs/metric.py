import torch

def compute_segmentation_metrics(y_gt, y_pred, threshold = 0.5, eps=1e-7):
    # Ensure binary (if not already)
    y_pred = (y_pred > threshold).float()
    y_gt   = (y_gt > threshold).float()

    # Flatten tensors
    y_pred_flat = y_pred.view(-1)
    y_gt_flat = y_gt.view(-1)

    # TP, FP, FN, TN
    TP = torch.sum(y_pred_flat * y_gt_flat)
    FP = torch.sum(y_pred_flat * (1 - y_gt_flat))
    FN = torch.sum((1 - y_pred_flat) * y_gt_flat)
    TN = torch.sum((1 - y_pred_flat) * (1 - y_gt_flat))

    # Metrics
    accuracy    = (TP + TN) / (TP + TN + FP + FN + eps)
    precision   = TP / (TP + FP + eps)
    recall      = TP / (TP + FN + eps)
    f1_score    = 2 * precision * recall / (precision + recall + eps)
    iou         = TP / (TP + FP + FN + eps)  # Also known as Jaccard Index
    dice        = 2 * TP / (2 * TP + FP + FN + eps)

    return {
        'accuracy': float(accuracy.item()),
        'precision': float(precision.item()),
        'recall': float(recall.item()),
        'f1_score': float(f1_score.item()),
        'iou': float(iou.item()),
        'dice': float(dice.item())
    }