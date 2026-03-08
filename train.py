"""
Main training entry point for Audio Event Detection.

Usage:
    python train.py                          # train with default config
    python train.py --config configs/custom.yaml  # train with custom config
    python train.py --resume                 # resume from latest checkpoint
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.utils.config import load_config, save_config
from src.utils.logger import setup_logger, get_logger
from src.data.dataset_preparation import (
    create_class_map,
    load_class_map,
    parse_labels_csv,
    split_dataset,
)
from src.data.dataset import create_dataloaders
from src.models.audio_event_model import build_model
from src.training.trainer import (
    AudioEventLightningModule,
    MetricsLoggerCallback,
    DriveCheckpointCallback,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Audio Event Detection Model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom YAML config file (overrides default).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu).",
    )
    return parser.parse_args()


def _find_resume_checkpoint(config) -> str | None:
    """Return the path to the latest checkpoint, or None."""
    import glob

    ckpt_dir = config.paths.checkpoint_dir
    # Lightning checkpoints
    pattern = os.path.join(ckpt_dir, "*.ckpt")
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    if ckpts:
        return ckpts[-1]

    # Try Google Drive fallback
    drive_dir = getattr(config.paths, "drive_checkpoint_dir", None)
    if drive_dir:
        latest_drive = os.path.join(drive_dir, "latest.ckpt")
        if os.path.exists(latest_drive):
            return latest_drive

    return None


def main():
    args = parse_args()

    # Build overrides from CLI args
    overrides = {}
    if args.epochs:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        overrides.setdefault("training", {})["learning_rate"] = args.lr

    # Load config
    config = load_config(
        config_path=args.config,
        overrides=overrides if overrides else None,
    )

    # Setup logger
    log_cfg = config.logging
    logger = setup_logger(
        name="audio_event_detection",
        log_level=log_cfg.level,
        log_file=log_cfg.log_file if log_cfg.file else None,
        console=log_cfg.console,
    )

    logger.info("=" * 60)
    logger.info("Audio Event Detection - Training")
    logger.info("=" * 60)

    # Save active config for reproducibility
    save_config(config, os.path.join(config.paths.log_dir, "active_config.yaml"))

    # Load or create class map
    class_map_path = config.paths.class_map_file
    if os.path.exists(class_map_path):
        class_map = load_class_map(class_map_path)
        logger.info(f"Loaded class map: {len(class_map)} classes")
    else:
        class_map = create_class_map(save_path=class_map_path)
        logger.info(f"Created class map: {len(class_map)} classes")

    # Update num_classes in config if needed
    config.model.num_classes = len(class_map)

    # Load and split dataset
    labels_csv = config.paths.labels_csv
    if not os.path.exists(labels_csv):
        logger.error(
            f"Labels CSV not found: {labels_csv}\n"
            f"Please create a labels CSV file with columns: filename,labels\n"
            f"Example: clip001.wav,speech|footsteps"
        )
        sys.exit(1)

    entries = parse_labels_csv(labels_csv)
    logger.info(f"Loaded {len(entries)} entries from {labels_csv}")

    train_entries, val_entries, test_entries = split_dataset(
        entries,
        val_split=config.training.val_split,
        test_split=config.training.test_split,
    )

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_entries=train_entries,
        val_entries=val_entries,
        class_map=class_map,
        config=config,
    )

    # Build model
    model = build_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model: {total_params:,} total params, "
        f"{trainable_params:,} trainable params"
    )

    # Wrap in Lightning module
    lit_module = AudioEventLightningModule(model=model, config=config)

    # ---- Callbacks ----
    train_cfg = config.training
    ckpt_cfg = config.checkpoint

    callbacks = []

    # Model checkpointing (replaces the old CheckpointManager)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.checkpoint_dir,
        filename="epoch{epoch:04d}-val_mAP{val_mAP:.4f}",
        auto_insert_metric_name=False,
        monitor=f"val_{ckpt_cfg.best_metric}",
        mode="max",
        save_top_k=ckpt_cfg.keep_last_n,
        save_last=True,
        every_n_epochs=ckpt_cfg.save_every_n_epochs,
    )
    callbacks.append(checkpoint_callback)

    # CSV metrics logger (keeps visualize_metrics.py & MetricsTracker working)
    callbacks.append(MetricsLoggerCallback(metrics_file=config.paths.metrics_file))

    # LR monitor for logging
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Google Drive sync (Colab persistence)
    drive_dir = getattr(config.paths, "drive_checkpoint_dir", None)
    if drive_dir:
        callbacks.append(DriveCheckpointCallback(drive_checkpoint_dir=drive_dir))

    # ---- Accelerator / device ----
    if args.device == "cpu":
        accelerator, devices = "cpu", "auto"
    else:
        accelerator = "gpu" if (args.device == "cuda" or args.device is None) else "cpu"
        devices = "auto"

    # ---- Resume from checkpoint ----
    resume_ckpt = None
    if args.resume:
        resume_ckpt = _find_resume_checkpoint(config)
        if resume_ckpt:
            logger.info(f"Resuming training from: {resume_ckpt}")
        else:
            logger.info("No checkpoint found. Starting fresh training.")

    # ---- Build Lightning Trainer ----
    gradient_clip = getattr(train_cfg, "gradient_clip_norm", 0.0) or None

    trainer = pl.Trainer(
        max_epochs=train_cfg.epochs,
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if train_cfg.mixed_precision else 32,
        gradient_clip_val=gradient_clip,
        callbacks=callbacks,
        default_root_dir=config.paths.log_dir,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    # ---- Train ----
    trainer.fit(
        model=lit_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
