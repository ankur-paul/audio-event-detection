"""
Evaluation metrics for multi-label audio event detection.

Supports:
- Precision, Recall, F1-score
- mean Average Precision (mAP)
- Per-class metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)

from src.utils.logger import get_logger

logger = get_logger()


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute multi-label classification metrics.

    Args:
        y_true: Ground truth binary labels of shape (n_samples, n_classes).
        y_pred: Predicted binary labels of shape (n_samples, n_classes).
            If None, computed from y_prob using threshold.
        y_prob: Predicted probabilities of shape (n_samples, n_classes).
        threshold: Threshold for converting probabilities to binary predictions.
        class_names: Optional list of class names for per-class reporting.

    Returns:
        Dictionary with metric names and values.
    """
    # Binarize predictions if needed
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(np.float32)

    metrics = {}

    # Sample-averaged metrics
    metrics["precision_micro"] = precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    metrics["recall_micro"] = recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    metrics["f1_micro"] = f1_score(
        y_true, y_pred, average="micro", zero_division=0
    )

    metrics["precision_macro"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["recall_macro"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["f1_macro"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Mean Average Precision per class
    n_classes = y_true.shape[1]
    ap_per_class = []

    for i in range(n_classes):
        if y_true[:, i].sum() > 0:
            ap = average_precision_score(y_true[:, i], y_prob[:, i])
            ap_per_class.append(ap)
        else:
            ap_per_class.append(0.0)

    metrics["mAP"] = np.mean(ap_per_class)

    # Per-class metrics (optional detailed reporting)
    if class_names is not None:
        per_class = {}
        for i, name in enumerate(class_names):
            if y_true[:, i].sum() > 0:
                per_class[name] = {
                    "AP": ap_per_class[i],
                    "precision": precision_score(
                        y_true[:, i], y_pred[:, i], zero_division=0
                    ),
                    "recall": recall_score(
                        y_true[:, i], y_pred[:, i], zero_division=0
                    ),
                    "f1": f1_score(
                        y_true[:, i], y_pred[:, i], zero_division=0
                    ),
                    "support": int(y_true[:, i].sum()),
                }
        metrics["per_class"] = per_class

    return metrics


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> np.ndarray:
    """
    Find the optimal threshold per class that maximizes F1 score.

    Args:
        y_true: Ground truth binary labels (n_samples, n_classes).
        y_prob: Predicted probabilities (n_samples, n_classes).

    Returns:
        Array of optimal thresholds per class.
    """
    n_classes = y_true.shape[1]
    thresholds = np.full(n_classes, 0.5)

    for i in range(n_classes):
        if y_true[:, i].sum() == 0:
            continue

        precision, recall, thresh = precision_recall_curve(
            y_true[:, i], y_prob[:, i]
        )

        # Compute F1 for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        if best_idx < len(thresh):
            thresholds[i] = thresh[best_idx]

    return thresholds


def format_metrics_summary(metrics: Dict[str, float]) -> str:
    """Format metrics into a readable summary string."""
    lines = [
        "=" * 50,
        "Evaluation Metrics",
        "=" * 50,
        f"  mAP:              {metrics.get('mAP', 0.0):.4f}",
        f"  Precision (micro): {metrics.get('precision_micro', 0.0):.4f}",
        f"  Recall (micro):    {metrics.get('recall_micro', 0.0):.4f}",
        f"  F1 (micro):        {metrics.get('f1_micro', 0.0):.4f}",
        f"  Precision (macro): {metrics.get('precision_macro', 0.0):.4f}",
        f"  Recall (macro):    {metrics.get('recall_macro', 0.0):.4f}",
        f"  F1 (macro):        {metrics.get('f1_macro', 0.0):.4f}",
        "=" * 50,
    ]
    return "\n".join(lines)
