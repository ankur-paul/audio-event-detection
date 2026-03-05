"""
Visualization entry point — generate training curves from saved metrics.

Usage:
    python visualize_metrics.py
    python visualize_metrics.py --output plots/
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.training.experiment_tracker import MetricsTracker
from src.visualization.visualizer import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument(
        "--config", type=str, default=None, help="Config file."
    )
    parser.add_argument(
        "--output", type=str, default="outputs", help="Output directory for plots."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(config_path=args.config)

    logger = setup_logger(name="audio_event_detection", log_level="INFO", console=True)

    tracker = MetricsTracker(metrics_file=config.paths.metrics_file)
    history = tracker.load_history()

    if not history.get("epoch"):
        logger.warning("No training metrics found.")
        return

    os.makedirs(args.output, exist_ok=True)
    save_path = os.path.join(args.output, "training_curves.png")
    plot_training_curves(history, save_path=save_path)
    logger.info(f"Training curves saved to {save_path}")


if __name__ == "__main__":
    main()
