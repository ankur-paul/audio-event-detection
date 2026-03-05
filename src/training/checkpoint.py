"""
Checkpointing system for Audio Event Detection.

Supports:
- Saving/loading full training state (model, optimizer, scheduler, epoch, metrics)
- Automatic checkpoint management (keep last N, save best)
- Google Colab/Drive persistence
- Resumable training across sessions
"""

import os
import glob
import shutil
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger()


class CheckpointManager:
    """
    Manages training checkpoints for resumable training.

    Saves and loads:
    - Model weights
    - Optimizer state
    - Learning rate scheduler state
    - Current epoch
    - Training/validation loss history
    - Best metric value
    - Scaler state (for mixed precision)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 5,
        save_best: bool = True,
        best_metric: str = "mAP",
        drive_checkpoint_dir: Optional[str] = None,
    ):
        """
        Args:
            checkpoint_dir: Local directory for checkpoints.
            keep_last_n: Number of recent checkpoints to keep.
            save_best: Whether to save the best model separately.
            best_metric: Metric name to track for best model.
            drive_checkpoint_dir: Optional Google Drive path for persistence.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_metric_value = -float("inf")
        self.drive_checkpoint_dir = drive_checkpoint_dir

        if drive_checkpoint_dir:
            Path(drive_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        train_loss_history: Optional[List[float]] = None,
        val_loss_history: Optional[List[float]] = None,
        metrics_history: Optional[Dict[str, List[float]]] = None,
        current_metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Save a training checkpoint.

        Args:
            epoch: Current epoch number.
            model: PyTorch model.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler (optional).
            scaler: GradScaler for mixed precision (optional).
            train_loss_history: List of training losses per epoch.
            val_loss_history: List of validation losses per epoch.
            metrics_history: Dict of metric name -> list of values per epoch.
            current_metrics: Current epoch's metrics.

        Returns:
            Path to saved checkpoint file.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss_history": train_loss_history or [],
            "val_loss_history": val_loss_history or [],
            "metrics_history": metrics_history or {},
            "best_metric_value": self.best_metric_value,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        # Save epoch checkpoint
        filename = f"checkpoint_epoch_{epoch:04d}.pth"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

        # Save to Google Drive if configured
        if self.drive_checkpoint_dir:
            drive_path = Path(self.drive_checkpoint_dir) / filename
            shutil.copy2(filepath, drive_path)
            # Also save as 'latest.pth' on Drive
            latest_drive_path = Path(self.drive_checkpoint_dir) / "latest.pth"
            shutil.copy2(filepath, latest_drive_path)
            logger.info(f"Checkpoint synced to Drive: {drive_path}")

        # Save best model
        if self.save_best and current_metrics:
            metric_val = current_metrics.get(self.best_metric, None)
            if metric_val is not None and metric_val > self.best_metric_value:
                self.best_metric_value = metric_val
                checkpoint["best_metric_value"] = self.best_metric_value
                best_path = self.checkpoint_dir / "best_model.pth"
                torch.save(checkpoint, best_path)
                logger.info(
                    f"New best model saved ({self.best_metric}: {metric_val:.4f})"
                )

                if self.drive_checkpoint_dir:
                    drive_best = Path(self.drive_checkpoint_dir) / "best_model.pth"
                    shutil.copy2(best_path, drive_best)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return str(filepath)

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint.

        Args:
            model: PyTorch model to load weights into.
            optimizer: Optimizer to restore state (optional).
            scheduler: Scheduler to restore state (optional).
            scaler: GradScaler to restore state (optional).
            checkpoint_path: Specific checkpoint file to load.
            load_best: Whether to load the best model instead of latest.
            device: Device to load the checkpoint to.

        Returns:
            Dictionary with checkpoint metadata (epoch, histories, etc.).
        """
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = str(self.checkpoint_dir / "best_model.pth")
            else:
                checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            # Try loading from Google Drive
            if self.drive_checkpoint_dir:
                drive_latest = Path(self.drive_checkpoint_dir) / "latest.pth"
                if drive_latest.exists():
                    checkpoint_path = str(drive_latest)
                    logger.info(f"Loading checkpoint from Drive: {checkpoint_path}")

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            logger.warning("No checkpoint found. Starting from scratch.")
            return {"epoch": 0}

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Restore model
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore scaler
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore best metric
        self.best_metric_value = checkpoint.get("best_metric_value", -float("inf"))

        logger.info(
            f"Resumed from epoch {checkpoint['epoch']}, "
            f"best {self.best_metric}: {self.best_metric_value:.4f}"
        )

        return {
            "epoch": checkpoint["epoch"],
            "train_loss_history": checkpoint.get("train_loss_history", []),
            "val_loss_history": checkpoint.get("val_loss_history", []),
            "metrics_history": checkpoint.get("metrics_history", {}),
            "best_metric_value": self.best_metric_value,
        }

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint file."""
        pattern = str(self.checkpoint_dir / "checkpoint_epoch_*.pth")
        checkpoints = sorted(glob.glob(pattern))
        if checkpoints:
            return checkpoints[-1]
        return None

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        pattern = str(self.checkpoint_dir / "checkpoint_epoch_*.pth")
        checkpoints = sorted(glob.glob(pattern))

        if len(checkpoints) > self.keep_last_n:
            for old_ckpt in checkpoints[: -self.keep_last_n]:
                os.remove(old_ckpt)
                logger.debug(f"Removed old checkpoint: {old_ckpt}")

    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists (locally or on Drive)."""
        if self._find_latest_checkpoint() is not None:
            return True
        if self.drive_checkpoint_dir:
            drive_latest = Path(self.drive_checkpoint_dir) / "latest.pth"
            return drive_latest.exists()
        return False
