"""
Preprocessing entry point for Audio Event Detection.

Runs the full data preparation pipeline:
1. Create class map
2. Extract spectrograms from audio files
3. Split dataset into train/val/test

Usage:
    python prepare_data.py
    python prepare_data.py --config configs/custom.yaml
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config
from src.utils.logger import setup_logger, get_logger
from src.data.dataset_preparation import (
    create_class_map,
    parse_labels_csv,
    split_dataset,
    save_split_csv,
)
from src.data.features import batch_extract_spectrograms


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for Audio Event Detection")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config file.",
    )
    parser.add_argument(
        "--skip-spectrograms",
        action="store_true",
        help="Skip spectrogram extraction (if already done).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(config_path=args.config)

    logger = setup_logger(
        name="audio_event_detection",
        log_level="INFO",
        console=True,
    )

    logger.info("=" * 60)
    logger.info("Audio Event Detection - Data Preparation")
    logger.info("=" * 60)

    # Step 1: Create class map
    logger.info("\n[1/3] Creating class map...")
    class_map = create_class_map(save_path=config.paths.class_map_file)

    # Step 2: Extract spectrograms
    if not args.skip_spectrograms:
        logger.info("\n[2/3] Extracting spectrograms...")
        audio_dir = config.paths.raw_audio_dir
        if os.path.isdir(audio_dir) and os.listdir(audio_dir):
            feat_cfg = config.features
            audio_cfg = config.audio
            count = batch_extract_spectrograms(
                audio_dir=audio_dir,
                output_dir=config.paths.spectrogram_dir,
                sample_rate=audio_cfg.sample_rate,
                clip_duration=audio_cfg.clip_duration,
                n_mels=feat_cfg.n_mels,
                window_size_ms=feat_cfg.window_size_ms,
                hop_length_ms=feat_cfg.hop_length_ms,
                f_min=feat_cfg.f_min,
                f_max=feat_cfg.f_max,
            )
            logger.info(f"Extracted {count} spectrograms")
        else:
            logger.warning(
                f"Audio directory is empty or missing: {audio_dir}\n"
                f"Place audio files there and re-run."
            )
    else:
        logger.info("\n[2/3] Skipping spectrogram extraction (--skip-spectrograms)")

    # Step 3: Split dataset
    logger.info("\n[3/3] Splitting dataset...")
    labels_csv = config.paths.labels_csv
    if os.path.exists(labels_csv):
        entries = parse_labels_csv(labels_csv)
        train_entries, val_entries, test_entries = split_dataset(
            entries,
            val_split=config.training.val_split,
            test_split=config.training.test_split,
        )

        data_dir = config.paths.data_dir
        save_split_csv(train_entries, os.path.join(data_dir, "train.csv"))
        save_split_csv(val_entries, os.path.join(data_dir, "val.csv"))
        save_split_csv(test_entries, os.path.join(data_dir, "test.csv"))
    else:
        logger.warning(
            f"Labels CSV not found: {labels_csv}\n"
            f"Please create it with format: filename,labels\n"
            f"Example: clip001.wav,speech|footsteps"
        )

    logger.info("\nData preparation complete!")


if __name__ == "__main__":
    main()
