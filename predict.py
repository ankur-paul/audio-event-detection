"""
Inference entry point for Audio Event Detection.

Usage:
    python predict.py --audio path/to/audio.wav
    python predict.py --audio path/to/long_audio.wav --visualize
    python predict.py --audio path/to/audio.wav --checkpoint best
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config
from src.utils.logger import setup_logger, get_logger
from src.data.dataset_preparation import load_class_map, get_inverse_class_map
from src.models.audio_event_model import build_model
from src.training.checkpoint import find_best_checkpoint, load_model_from_checkpoint
from src.inference.inference_pipeline import InferencePipeline
from src.visualization.visualizer import (
    plot_event_timeline,
    plot_frame_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Audio Event Detection Inference")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file for inference.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path, or 'best' to load best model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Detection threshold (overrides config).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate event timeline visualization.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save outputs.",
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

    # Load config
    overrides = {}
    if args.threshold:
        overrides.setdefault("inference", {})["threshold"] = args.threshold

    config = load_config(
        config_path=args.config,
        overrides=overrides if overrides else None,
    )

    # Setup logger
    logger = setup_logger(
        name="audio_event_detection",
        log_level="INFO",
        console=True,
    )

    logger.info("=" * 60)
    logger.info("Audio Event Detection - Inference")
    logger.info("=" * 60)

    # Validate audio file
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)

    # Load class map
    class_map_path = config.paths.class_map_file
    if not os.path.exists(class_map_path):
        logger.error(f"Class map not found: {class_map_path}")
        sys.exit(1)

    class_map = load_class_map(class_map_path)
    inverse_class_map = get_inverse_class_map(class_map)
    config.model.num_classes = len(class_map)

    # Build and load model
    model = build_model(config)

    drive_dir = getattr(config.paths, "drive_checkpoint_dir", None) or None

    if args.checkpoint and args.checkpoint != "best":
        ckpt_path = args.checkpoint
    else:
        ckpt_path = find_best_checkpoint(
            checkpoint_dir=config.paths.checkpoint_dir,
            drive_dir=drive_dir,
        )

    if ckpt_path:
        load_model_from_checkpoint(
            model=model,
            checkpoint_path=ckpt_path,
            device=args.device or "cpu",
        )
    else:
        logger.warning("No trained checkpoint found. Using randomly initialized model.")

    # Create inference pipeline
    pipeline = InferencePipeline(
        model=model,
        class_map=class_map,
        config=config,
        device=args.device,
    )

    # Run inference
    result = pipeline.predict_file(args.audio)

    # Print results
    logger.info(f"\nDetected {len(result.events)} events:")
    logger.info("-" * 60)
    for event in result.events:
        logger.info(
            f"  {event.class_name:20s} | "
            f"{event.start_time:6.2f}s - {event.end_time:6.2f}s | "
            f"conf: {event.confidence:.3f}"
        )
    logger.info("-" * 60)

    # Save results as JSON
    os.makedirs(args.output_dir, exist_ok=True)
    audio_basename = os.path.splitext(os.path.basename(args.audio))[0]

    results_json = {
        "filename": args.audio,
        "duration": result.duration,
        "num_events": len(result.events),
        "events": [
            {
                "class": e.class_name,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "confidence": e.confidence,
            }
            for e in result.events
        ],
    }

    json_path = os.path.join(args.output_dir, f"{audio_basename}_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results saved: {json_path}")

    # Visualization
    if args.visualize:
        # Event timeline
        timeline_path = os.path.join(args.output_dir, f"{audio_basename}_timeline.png")
        plot_event_timeline(result, save_path=timeline_path)

        # Frame predictions heatmap
        if result.frame_predictions is not None:
            class_names = [inverse_class_map[i] for i in range(len(class_map))]
            heatmap_path = os.path.join(
                args.output_dir, f"{audio_basename}_frame_predictions.png"
            )
            plot_frame_predictions(
                result,
                class_names=class_names,
                save_path=heatmap_path,
            )


if __name__ == "__main__":
    main()
