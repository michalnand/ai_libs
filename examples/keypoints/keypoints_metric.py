import torch

def keypoint_metrics(y_gt, y_pred, threshold=0.5, eps=1e-7):
    """
    metrics for sparse keypoint heatmaps.
    Handles non-monotonic PR curves and operates safely on PyTorch tensors.
    """
    # Ensure tensors are float and flattened
    y_gt = y_gt.float().view(-1)
    y_pred = y_pred.float().view(-1)
    
    # Binarize for F1/Precision/Recall
    y_pred_bin = (y_pred >= threshold).float()

    # True Positives, False Positives, False Negatives
    TP = (y_gt * y_pred_bin).sum()
    FP = ((1 - y_gt) * y_pred_bin).sum()
    FN = (y_gt * (1 - y_pred_bin)).sum()

    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # ---- Average Precision (COCO/VOC Style) ----
    # 1. Sort predictions in descending order
    sorted_idx = torch.argsort(y_pred, descending=True)
    y_gt_sorted = y_gt[sorted_idx]

    # 2. Compute cumulative TP and FP
    tp_cumsum = torch.cumsum(y_gt_sorted, dim=0)
    fp_cumsum = torch.cumsum(1 - y_gt_sorted, dim=0)

    # 3. Precision and Recall curves
    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
    total_positives = y_gt.sum()
    
    if total_positives == 0:
        # Edge case: No ground truth keypoints in this batch
        ap = torch.tensor(0.0, device=y_pred.device)
    else:
        recall_curve = tp_cumsum / total_positives

        # 4. PR Curve Interpolation (Removes the zig-zag issue)
        # We walk backwards to find the maximum precision to the right of any point
        precision_interp = torch.flip(
            torch.cummax(torch.flip(precision_curve, dims=[0]), dim=0)[0], 
            dims=[0]
        )

        # 5. Find where recall changes to calculate the area under rectangles (VOC/COCO style)
        # Find indices where recall increases
        recall_diff = torch.diff(recall_curve, prepend=torch.tensor([0.0], device=y_pred.device))
        ap = torch.sum(precision_interp * recall_diff)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "ap": ap.item(),
    }