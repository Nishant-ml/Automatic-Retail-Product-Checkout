"""IoU + RPC benchmark metrics.

This exact function was copy-pasted with tiny formatting differences into
all three original notebooks. Centralizing it means a fix here applies
everywhere, and there's one function to audit instead of three.
"""
import numpy as np


def calculate_iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    return inter / float(area_a + area_b - inter)


def compute_rpc_metrics(gt_counts, pred_counts):
    """Absolute Count Difference / mean Category Count Difference /
    mean Category IoU, as defined by the RPC checkout benchmark. This is
    computed the SAME way regardless of which embedder produced
    pred_counts, so YOLOv8 / +VAE / +DINOv2 rows stay comparable."""
    categories = set(gt_counts.keys()) | set(pred_counts.keys())

    total_gt = sum(gt_counts.values())
    total_pred = sum(pred_counts.values())
    acd = abs(total_gt - total_pred)

    mccd_list = []
    inter, union = 0, 0
    for c in categories:
        g = gt_counts.get(c, 0)
        p = pred_counts.get(c, 0)
        if g > 0:
            mccd_list.append(abs(p - g) / g)
        inter += min(g, p)
        union += max(g, p)

    mccd = float(np.mean(mccd_list)) if mccd_list else 0.0
    mciou = inter / union if union > 0 else 0.0
    return acd, mccd, mciou
