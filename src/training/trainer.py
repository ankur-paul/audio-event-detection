"""
PyTorch Lightning module for multi-label audio event detection.

All boilerplate (AMP, gradient clipping, checkpointing, LR scheduling,
device management) is handled by Lightning's Trainer.  This module defines
only the model-specific logic:

- training_step / validation_step  -> forward pass + loss
- on_validation_epoch_end          -> aggregate metrics (mAP, F1, etc.)
- configure_optimizers             -> optimizer + LR scheduler

Two companion callbacks are included:
- MetricsLoggerCallback   - writes per-epoch CSV (visualize_metrics.py compat)
- DriveCheckpointCallback - syncs checkpoints to Google Drive for Colab
"""

import os
import shutil
import json
import time
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.training.metrics import compute_metrics, format_metrics_summary
from src.training.experiment_tracker import MetricsTracker
from src.utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------


class AudioEventLightningModule(pl.LightningModule):
    """
    Lightning wrapper for AudioEventDetectionModel.

    Handles multi-label BCE training with clip-level loss and computes
    mAP / F1 / precision / recall at the end of each validation epoch.
    """

    def __init__(self, model: nn.Module, config, pos_weight: Optional[torch.Tensor] = None):
        """
        Args:
            model: An AudioEventDetectionModel instance.
            config: Full project configuration object.
            pos_weight: Optional per-class positive weights for BCE loss.
                        Shape (num_classes,). Compensates for class imbalance.
        """
        super().__init__()
        self.model = model
        self.config = config
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logger.info("Using class-balanced BCEWithLogitsLoss (pos_weight enabled)")
        else:
            self.pos_weight = None
            self.criterion = nn.BCEWithLogitsLoss()
        self.threshold = config.inference.threshold

        # Accumulate validation outputs for epoch-level metric computation
        self.validation_step_outputs: List[Dict[str, np.ndarray]] = []

    # -- forward ----------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    # -- training ---------------------------------------------------------

    def training_step(self, batch, batch_idx):
        spectrograms, labels = batch
        outputs = self.model(spectrograms)
        loss = self.criterion(outputs["clip_logits"], labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # -- validation -------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        spectrograms, labels = batch
        outputs = self.model(spectrograms)
        loss = self.criterion(outputs["clip_logits"], labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.append({
            "labels": labels.cpu().numpy(),
            "probs": outputs["clip_probs"].cpu().numpy(),
        })
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        y_true = np.concatenate([o["labels"] for o in self.validation_step_outputs])
        y_prob = np.concatenate([o["probs"] for o in self.validation_step_outputs])

        # Guard against NaN (can happen with random weights + mixed precision)
        if np.isnan(y_prob).any():
            logger.warning("NaN detected in validation predictions — replacing with 0.")
            y_prob = np.nan_to_num(y_prob, nan=0.0)

        y_pred = (y_prob >= self.threshold).astype(np.float32)

        metrics = compute_metrics(
            y_true=y_true, y_pred=y_pred, y_prob=y_prob, threshold=self.threshold,
        )

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log(f"val_{key}", value, prog_bar=(key == "mAP"))

        self.validation_step_outputs.clear()

    # -- optimizers & schedulers ------------------------------------------

    def configure_optimizers(self):
        train_cfg = self.config.training

        # Optimizer
        if train_cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=train_cfg.learning_rate,
                weight_decay=train_cfg.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=train_cfg.learning_rate,
                weight_decay=train_cfg.weight_decay,
            )

        # Scheduler
        sched_cfg = train_cfg.scheduler
        if not getattr(sched_cfg, "enabled", False):
            return optimizer

        warmup_epochs = getattr(sched_cfg, "warmup_epochs", 0)
        sched_type = getattr(sched_cfg, "type", "cosine")

        # Build main scheduler
        if sched_type == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_cfg.epochs - warmup_epochs,
                eta_min=getattr(sched_cfg, "min_lr", 1e-6),
            )
        elif sched_type == "step":
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=getattr(sched_cfg, "step_size", 20),
                gamma=getattr(sched_cfg, "gamma", 0.5),
            )
        elif sched_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=getattr(sched_cfg, "factor", 0.5),
                patience=getattr(sched_cfg, "patience", 5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                },
            }
        else:
            logger.warning(f"Unknown scheduler type: {sched_type}")
            return optimizer

        # Wrap with linear warmup if configured
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = main_scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class MetricsLoggerCallback(pl.Callback):
    """
    Writes per-epoch metrics to the CSV format expected by
    ``visualize_metrics.py`` and the existing ``MetricsTracker``.
    """

    def __init__(self, metrics_file: str):
        super().__init__()
        self.tracker = MetricsTracker(metrics_file=metrics_file)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip the sanity-check validation run
        if trainer.sanity_checking:
            return

        cb_metrics = trainer.callback_metrics
        epoch = trainer.current_epoch + 1  # 1-indexed

        val_metrics = {
            "precision_micro": cb_metrics.get("val_precision_micro", torch.tensor(0.0)).item(),
            "recall_micro": cb_metrics.get("val_recall_micro", torch.tensor(0.0)).item(),
            "f1_micro": cb_metrics.get("val_f1_micro", torch.tensor(0.0)).item(),
            "precision_macro": cb_metrics.get("val_precision_macro", torch.tensor(0.0)).item(),
            "recall_macro": cb_metrics.get("val_recall_macro", torch.tensor(0.0)).item(),
            "f1_macro": cb_metrics.get("val_f1_macro", torch.tensor(0.0)).item(),
            "mAP": cb_metrics.get("val_mAP", torch.tensor(0.0)).item(),
        }

        lr = trainer.optimizers[0].param_groups[0]["lr"]

        self.tracker.log_epoch(
            epoch=epoch,
            train_loss=cb_metrics.get("train_loss_epoch", torch.tensor(0.0)).item(),
            val_loss=cb_metrics.get("val_loss", torch.tensor(0.0)).item(),
            metrics=val_metrics,
            learning_rate=lr,
        )


class DriveCheckpointCallback(pl.Callback):
    """
    Syncs the best and latest Lightning checkpoints to Google Drive
    so that Colab training can be resumed after a disconnect.
    """

    def __init__(self, drive_checkpoint_dir: str):
        super().__init__()
        self.drive_dir = Path(drive_checkpoint_dir)
        self.drive_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        ckpt_cb = trainer.checkpoint_callback
        if ckpt_cb is None:
            return

        best_path = ckpt_cb.best_model_path
        if best_path and os.path.exists(best_path):
            dest = self.drive_dir / "best_model.ckpt"
            shutil.copy2(best_path, dest)
            logger.debug(f"Best model synced to Drive: {dest}")

        last_path = ckpt_cb.last_model_path
        if last_path and os.path.exists(last_path):
            dest = self.drive_dir / "latest.ckpt"
            shutil.copy2(last_path, dest)
            logger.debug(f"Latest checkpoint synced to Drive: {dest}")


class TrainingTimeCallback(pl.Callback):
    """
    Tracks wall-clock training time and persists cumulative totals across runs.

    Writes a JSON file that is updated at least once per epoch and at train end,
    so interrupted sessions still contribute to cumulative time.
    """

    def __init__(self, time_file: str):
        super().__init__()
        self.time_file = Path(time_file)
        self.time_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_start_time: Optional[float] = None
        self.base_cumulative_seconds: float = 0.0
        self.base_sessions_completed: int = 0

    def _load_existing_state(self) -> Dict[str, Any]:
        if not self.time_file.exists():
            return {}
        try:
            with self.time_file.open("r") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to read training time file '{self.time_file}': {exc}")
            return {}

    def _write_state(self, trainer, final: bool = False) -> None:
        if self.session_start_time is None:
            return

        now = time.time()
        session_seconds = max(0.0, now - self.session_start_time)
        cumulative_seconds = self.base_cumulative_seconds + session_seconds
        sessions_completed = self.base_sessions_completed + (1 if final else 0)

        payload = {
            "cumulative_seconds": round(cumulative_seconds, 3),
            "cumulative_hours": round(cumulative_seconds / 3600.0, 3),
            "last_session_seconds": round(session_seconds, 3),
            "sessions_completed": int(sessions_completed),
            "last_updated_epoch": int(getattr(trainer, "current_epoch", 0) + 1),
            "status": "final" if final else "in_progress",
        }

        try:
            with self.time_file.open("w") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            logger.warning(f"Failed to write training time file '{self.time_file}': {exc}")

    def on_train_start(self, trainer, pl_module):
        existing = self._load_existing_state()
        self.base_cumulative_seconds = float(existing.get("cumulative_seconds", 0.0))
        self.base_sessions_completed = int(existing.get("sessions_completed", 0))
        self.session_start_time = time.time()
        logger.info(
            f"Training time tracker active: {self.time_file} "
            f"(existing total: {self.base_cumulative_seconds / 3600.0:.2f}h, "
            f"sessions: {self.base_sessions_completed})"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        self._write_state(trainer, final=False)

    def on_train_end(self, trainer, pl_module):
        self._write_state(trainer, final=True)
