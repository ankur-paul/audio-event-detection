"""
Training pipeline for multi-label audio event detection.

Features:
- Resumable training across Colab sessions
- Mixed-precision training
- Learning rate scheduling (cosine, step, plateau)
- Gradient clipping
- Automatic checkpointing
- Metrics tracking and logging
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Tuple

from src.training.checkpoint import CheckpointManager
from src.training.metrics import compute_metrics, format_metrics_summary
from src.training.experiment_tracker import MetricsTracker
from src.utils.logger import get_logger

logger = get_logger()


class Trainer:
    """
    Full training pipeline with checkpoint resumption and mixed-precision support.
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        train_loader,
        val_loader,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: The AudioEventDetectionModel.
            config: Full configuration object.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            device: Device string ('cuda', 'cpu'). Auto-detected if None.
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # Loss function - BCE for multi-label
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        train_cfg = config.training
        if train_cfg.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_cfg.learning_rate,
                weight_decay=train_cfg.weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_cfg.learning_rate,
                weight_decay=train_cfg.weight_decay,
            )

        # Learning rate scheduler
        self.scheduler = self._build_scheduler(train_cfg)

        # Mixed precision
        self.use_amp = train_cfg.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient clipping
        self.gradient_clip_norm = getattr(train_cfg, "gradient_clip_norm", 0.0)

        # Checkpoint manager
        ckpt_cfg = config.checkpoint
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.paths.checkpoint_dir,
            keep_last_n=ckpt_cfg.keep_last_n,
            save_best=ckpt_cfg.save_best,
            best_metric=ckpt_cfg.best_metric,
            drive_checkpoint_dir=getattr(config.paths, "drive_checkpoint_dir", None) or None,
        )

        # Metrics tracker
        self.metrics_tracker = MetricsTracker(
            metrics_file=config.paths.metrics_file,
        )

        # Training state
        self.start_epoch = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.metrics_history = {}

    def _build_scheduler(self, train_cfg):
        """Build learning rate scheduler from config."""
        sched_cfg = train_cfg.scheduler
        if not getattr(sched_cfg, "enabled", False):
            return None

        sched_type = getattr(sched_cfg, "type", "cosine")

        if sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg.epochs - getattr(sched_cfg, "warmup_epochs", 0),
                eta_min=getattr(sched_cfg, "min_lr", 1e-6),
            )
        elif sched_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=getattr(sched_cfg, "step_size", 20),
                gamma=getattr(sched_cfg, "gamma", 0.5),
            )
        elif sched_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=getattr(sched_cfg, "factor", 0.5),
                patience=getattr(sched_cfg, "patience", 5),
            )
        else:
            logger.warning(f"Unknown scheduler type: {sched_type}")
            scheduler = None

        return scheduler

    def resume_training(self) -> None:
        """
        Attempt to resume training from the latest checkpoint.
        Restores model, optimizer, scheduler, scaler, and training history.
        """
        if not self.checkpoint_manager.has_checkpoint():
            logger.info("No checkpoint found. Starting fresh training.")
            return

        state = self.checkpoint_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=str(self.device),
        )

        self.start_epoch = state.get("epoch", 0)
        self.train_loss_history = state.get("train_loss_history", [])
        self.val_loss_history = state.get("val_loss_history", [])
        self.metrics_history = state.get("metrics_history", {})

        logger.info(f"Resuming training from epoch {self.start_epoch}")

    def train(self) -> None:
        """
        Main training loop.

        Runs for configured number of epochs, with checkpointing,
        validation, and metrics logging at each epoch.
        """
        train_cfg = self.config.training
        total_epochs = train_cfg.epochs
        warmup_epochs = getattr(
            train_cfg.scheduler, "warmup_epochs", 0
        ) if hasattr(train_cfg, "scheduler") else 0

        logger.info(
            f"Starting training: epochs={total_epochs}, "
            f"start_epoch={self.start_epoch}, device={self.device}"
        )

        for epoch in range(self.start_epoch + 1, total_epochs + 1):
            epoch_start = time.time()

            # Warmup learning rate
            if epoch <= warmup_epochs:
                warmup_lr = train_cfg.learning_rate * (epoch / warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            # Train one epoch
            train_loss = self._train_epoch(epoch)
            self.train_loss_history.append(train_loss)

            # Validate
            val_loss, val_metrics = self._validate_epoch(epoch)
            self.val_loss_history.append(val_loss)

            # Update metrics history
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    if key not in self.metrics_history:
                        self.metrics_history[key] = []
                    self.metrics_history[key].append(value)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log metrics
            self.metrics_tracker.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=val_metrics,
                learning_rate=current_lr,
            )

            # Update scheduler
            if self.scheduler is not None and epoch > warmup_epochs:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save checkpoint
            ckpt_cfg = self.config.checkpoint
            if epoch % ckpt_cfg.save_every_n_epochs == 0:
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    train_loss_history=self.train_loss_history,
                    val_loss_history=self.val_loss_history,
                    metrics_history=self.metrics_history,
                    current_metrics=val_metrics,
                )

            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}/{total_epochs} completed in {epoch_time:.1f}s | "
                f"LR: {current_lr:.2e}"
            )

        logger.info("Training complete!")
        logger.info(format_metrics_summary(val_metrics))

    def _train_epoch(self, epoch: int) -> float:
        """
        Run one training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (spectrograms, labels) in enumerate(self.train_loader):
            spectrograms = spectrograms.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(spectrograms)
                    loss = self.criterion(outputs["clip_logits"], labels)

                self.scaler.scale(loss).backward()

                if self.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(spectrograms)
                loss = self.criterion(outputs["clip_logits"], labels)
                loss.backward()

                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )

                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Log progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                logger.info(
                    f"  Epoch {epoch} [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Run validation and compute metrics.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple of (average_val_loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_labels = []
        all_probs = []

        for spectrograms, labels in self.val_loader:
            spectrograms = spectrograms.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with autocast():
                    outputs = self.model(spectrograms)
                    loss = self.criterion(outputs["clip_logits"], labels)
            else:
                outputs = self.model(spectrograms)
                loss = self.criterion(outputs["clip_logits"], labels)

            total_loss += loss.item()
            num_batches += 1

            all_labels.append(labels.cpu().numpy())
            all_probs.append(outputs["clip_probs"].cpu().numpy())

        avg_loss = total_loss / max(num_batches, 1)

        # Compute metrics
        if all_labels:
            y_true = np.concatenate(all_labels, axis=0)
            y_prob = np.concatenate(all_probs, axis=0)
            y_pred = (y_prob >= self.config.inference.threshold).astype(np.float32)

            metrics = compute_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                threshold=self.config.inference.threshold,
            )
        else:
            metrics = {"mAP": 0.0, "f1_micro": 0.0, "precision_micro": 0.0, "recall_micro": 0.0}

        return avg_loss, metrics
