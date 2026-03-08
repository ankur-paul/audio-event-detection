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

    def __init__(self, model: nn.Module, config):
        """
        Args:
            model: An AudioEventDetectionModel instance.
            config: Full project configuration object.
        """
        super().__init__()
        self.model = model
        self.config = config
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
