"""Segmentation evaluation: confusion-matrix-based mIoU."""

import numpy as np


class SegmentationMetrics:
    """Accumulate a confusion matrix; report per-class IoU + mIoU."""

    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def add_batch(self, pred, gt):
        """pred, gt: arrays of shape [B, H, W] or [H, W] with class indices."""
        pred = np.asarray(pred).flatten()
        gt = np.asarray(gt).flatten()
        valid = (gt != self.ignore_index) & (gt >= 0) & (gt < self.num_classes)
        idx = self.num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
        bincount = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += bincount.reshape(self.num_classes, self.num_classes)

    def evaluate(self):
        diag = np.diag(self.confusion).astype(np.float64)
        gt_sum = self.confusion.sum(axis=1).astype(np.float64)
        pred_sum = self.confusion.sum(axis=0).astype(np.float64)
        union = gt_sum + pred_sum - diag
        valid = union > 0
        per_class_iou = np.zeros(self.num_classes, dtype=np.float64)
        per_class_iou[valid] = diag[valid] / union[valid]
        miou = per_class_iou[valid].mean() if valid.any() else 0.0
        return per_class_iou, miou
