import os

import numpy as np
import torch


def iou_score(y_pred, y_true, smooth=1e-5):
    """Compute IoU for binary prediction (single-channel after sigmoid)."""
    if y_pred.dim() > y_true.dim():
        y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    y_true = (y_true > 0.5).float()

    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def dice_score(y_pred, y_true, smooth=1e-5):
    """Compute Dice for binary prediction or probability map."""
    if y_pred.dim() > y_true.dim():
        y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1).float()

    intersection = (y_pred * y_true).sum()
    return (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)


def _argmax_prediction(y_pred):
    """Convert network output (logits/probabilities) to class labels."""
    if y_pred.dim() == 5:
        # expected input: (B, C, D, H, W)
        if y_pred.shape[1] > 1:
            return torch.argmax(y_pred, dim=1)
        else:
            # binary single channel
            return (torch.sigmoid(y_pred) > 0.5).long().squeeze(1)
    raise ValueError("y_pred must be 5D tensor (B, C, D, H, W) for Brats multiclass evaluation")


def _to_label_tensor(y_true):
    """Convert truth (label or one-hot) to class label tensor."""
    if y_true.dim() == 5 and y_true.shape[1] > 1:
        return torch.argmax(y_true, dim=1)
    if y_true.dim() == 4:
        return y_true.long()
    raise ValueError("y_true must be either (B, C, D, H, W) one-hot or (B, D, H, W) class labels")


def compute_brats_metrics(y_pred, y_true, smooth=1e-5):
    """Compute BraTS standard metrics: Dice (ET, TC, WT, avg)."""
    y_pred_cls = _argmax_prediction(y_pred)
    y_true_cls = _to_label_tensor(y_true)

    def dice_region(pred, true, pred_classes, true_classes):
        pred_mask = torch.zeros_like(pred, dtype=torch.bool)
        true_mask = torch.zeros_like(true, dtype=torch.bool)

        for c in pred_classes:
            pred_mask |= (pred == c)
        for c in true_classes:
            true_mask |= (true == c)

        pred_fg = pred_mask.float()
        true_fg = true_mask.float()

        intersection = (pred_fg * true_fg).sum()
        union = pred_fg.sum() + true_fg.sum()
        return (2.0 * intersection + smooth) / (union + smooth)

    dice_et = dice_region(y_pred_cls, y_true_cls, [3], [3])
    dice_tc = dice_region(y_pred_cls, y_true_cls, [1, 3], [1, 3])
    dice_wt = dice_region(y_pred_cls, y_true_cls, [1, 2, 3], [1, 2, 3])

    dice_avg = (dice_et + dice_tc + dice_wt) / 3.0

    return {
        'dice_et': dice_et.item(),
        'dice_tc': dice_tc.item(),
        'dice_wt': dice_wt.item(),
        'dice_avg': dice_avg.item(),
    }


def brats_hausdorff_distance(y_pred, y_true):
    """Compute Hausdorff distance (HD95) for ET, TC, WT regions."""
    try:
        from scipy.spatial.distance import cdist
        from scipy.ndimage import binary_erosion
    except ImportError:
        raise ImportError("Scipy is required for Hausdorff distance. Install with 'pip install scipy'.")

    y_pred_cls = _argmax_prediction(y_pred).cpu().numpy()
    y_true_cls = _to_label_tensor(y_true).cpu().numpy()

    def _get_boundary_points(mask, max_points=5000):
        """Extract boundary points from a binary mask with sampling to limit memory."""
        # Erode the mask to get interior points, then subtract to get boundary
        eroded = binary_erosion(mask, iterations=1)
        boundary = mask & ~eroded
        boundary_pts = np.argwhere(boundary)
        
        # If too few boundary points, fall back to all points
        if len(boundary_pts) < 100:
            boundary_pts = np.argwhere(mask)
        
        # Sample if too many points to avoid memory issues
        if len(boundary_pts) > max_points:
            indices = np.random.choice(len(boundary_pts), max_points, replace=False)
            boundary_pts = boundary_pts[indices]
        
        return boundary_pts

    def _region_distance(pred, true, classes):
        pred_mask = np.isin(pred, classes)
        true_mask = np.isin(true, classes)

        if not pred_mask.any() and not true_mask.any():
            return 0.0
        if not pred_mask.any() or not true_mask.any():
            return float('inf')

        # Use boundary points with sampling to save memory
        pred_pts = _get_boundary_points(pred_mask)
        true_pts = _get_boundary_points(true_mask)

        # Compute directional distances
        d_pred_to_true = np.min(cdist(pred_pts, true_pts), axis=1)
        d_true_to_pred = np.min(cdist(true_pts, pred_pts), axis=1)

        # 95th percentile in each direction, then max for HD95
        h95_1 = np.percentile(d_pred_to_true, 95)
        h95_2 = np.percentile(d_true_to_pred, 95)
        return max(h95_1, h95_2)

    hd_et = _region_distance(y_pred_cls, y_true_cls, [3])
    hd_tc = _region_distance(y_pred_cls, y_true_cls, [1, 3])
    hd_wt = _region_distance(y_pred_cls, y_true_cls, [1, 2, 3])

    return {'hd95_et': hd_et, 'hd95_tc': hd_tc, 'hd95_wt': hd_wt}


if __name__ == '__main__':
    # quick sanity check
    print('metrics.py loaded successfully')
