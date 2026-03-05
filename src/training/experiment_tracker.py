"""
Experiment tracking and metrics logging for Audio Event Detection.

Records training/validation metrics per epoch in CSV format
for persistence across Colab sessions.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger()


class MetricsTracker:
    """
    Tracks and persists training metrics to CSV files.

    Maintains:
    - Training loss per epoch
    - Validation loss per epoch
    - Precision, Recall, F1-score per epoch
    - mAP per epoch

    All data is saved to CSV so it can be loaded after restarting the environment.
    """

    def __init__(self, metrics_file: str = "logs/metrics_history.csv"):
        """
        Args:
            metrics_file: Path to the CSV file for metrics persistence.
        """
        self.metrics_file = metrics_file
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

        # Column names
        self.columns = [
            "epoch",
            "train_loss",
            "val_loss",
            "precision_micro",
            "recall_micro",
            "f1_micro",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "mAP",
            "learning_rate",
        ]

        # Initialize file with header if it doesn't exist
        if not os.path.exists(metrics_file):
            self._write_header()

    def _write_header(self) -> None:
        """Write the CSV header."""
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, float],
        learning_rate: float,
    ) -> None:
        """
        Log metrics for a single epoch.

        Args:
            epoch: Epoch number.
            train_loss: Training loss.
            val_loss: Validation loss.
            metrics: Dictionary of evaluation metrics.
            learning_rate: Current learning rate.
        """
        row = [
            epoch,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{metrics.get('precision_micro', 0.0):.6f}",
            f"{metrics.get('recall_micro', 0.0):.6f}",
            f"{metrics.get('f1_micro', 0.0):.6f}",
            f"{metrics.get('precision_macro', 0.0):.6f}",
            f"{metrics.get('recall_macro', 0.0):.6f}",
            f"{metrics.get('f1_macro', 0.0):.6f}",
            f"{metrics.get('mAP', 0.0):.6f}",
            f"{learning_rate:.8f}",
        ]

        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"mAP={metrics.get('mAP', 0.0):.4f}, "
            f"F1={metrics.get('f1_micro', 0.0):.4f}"
        )

    def load_history(self) -> Dict[str, List[float]]:
        """
        Load full metrics history from CSV.

        Returns:
            Dictionary mapping column names to lists of values.
        """
        history = {col: [] for col in self.columns}

        if not os.path.exists(self.metrics_file):
            return history

        with open(self.metrics_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in self.columns:
                    if col in row:
                        try:
                            history[col].append(float(row[col]))
                        except (ValueError, TypeError):
                            history[col].append(row[col])

        return history

    def get_last_epoch(self) -> int:
        """Get the last recorded epoch number."""
        history = self.load_history()
        if history["epoch"]:
            return int(history["epoch"][-1])
        return 0
