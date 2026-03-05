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
from src.training.trainer import Trainer


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

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
    )

    # Resume if requested
    if args.resume:
        trainer.resume_training()

    # Train
    trainer.train()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
