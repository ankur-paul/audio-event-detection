"""
Visualization module for Audio Event Detection.

Generates:
- Sound event timeline plots (horizontal bar charts)
- Training metrics plots (loss curves, mAP, F1)
- Spectrogram visualizations
- Per-class detection confidence plots
"""

import os
import numpy as np
import matplotlib
# Only force non-interactive backend when no display is available (e.g. scripts).
# In notebooks (Colab/Jupyter), the inline backend is already set and must be kept.
if matplotlib.get_backend() == "agg" or not os.environ.get("DISPLAY"):
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from typing import Dict, List, Optional, Tuple

from src.inference.inference_pipeline import InferenceResult, DetectedEvent
from src.utils.logger import get_logger

logger = get_logger()

# Color palette for event classes
_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def plot_event_timeline(
    result: InferenceResult,
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 150,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show_confidence: bool = True,
) -> plt.Figure:
    """
    Generate a sound event timeline visualization.

    Creates a horizontal bar chart where each row is a sound class
    and bars show when that event occurs.

    Args:
        result: InferenceResult from the inference pipeline.
        figsize: Figure size (width, height).
        dpi: Figure DPI.
        save_path: Path to save the figure. If None, returns the figure.
        title: Plot title.
        show_confidence: Whether to show confidence values.

    Returns:
        matplotlib Figure object.
    """
    events = result.events
    if not events:
        logger.warning("No events to visualize.")
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No events detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        if save_path:
            _save_figure(fig, save_path)
        return fig

    # Get unique classes present
    class_names = sorted(set(e.class_name for e in events))
    class_to_y = {name: i for i, name in enumerate(class_names)}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for event in events:
        y = class_to_y[event.class_name]
        color_idx = y % len(_COLORS)
        color = _COLORS[color_idx]

        # Adjust alpha based on confidence
        alpha = 0.4 + 0.6 * event.confidence if show_confidence else 0.8
        bar_color = to_rgba(color, alpha)

        duration = event.end_time - event.start_time
        bar = ax.barh(
            y,
            duration,
            left=event.start_time,
            height=0.6,
            color=bar_color,
            edgecolor=color,
            linewidth=0.5,
        )

        # Add confidence text
        if show_confidence and duration > result.duration * 0.05:
            ax.text(
                event.start_time + duration / 2,
                y,
                f"{event.confidence:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )

    # Formatting
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_xlim(0, result.duration)
    ax.set_ylim(-0.5, len(class_names) - 0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    else:
        ax.set_title(
            f"Sound Event Timeline — {os.path.basename(result.filename)}",
            fontsize=13,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_frame_predictions(
    result: InferenceResult,
    class_names: Optional[List[str]] = None,
    top_k: int = 10,
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 150,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot frame-level prediction heatmap.

    Shows probability over time for the top-K most active classes.

    Args:
        result: InferenceResult with frame_predictions.
        class_names: List of all class names (ordered by index).
        top_k: Number of top classes to display.
        figsize: Figure size.
        dpi: DPI.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    if result.frame_predictions is None:
        logger.warning("No frame predictions available.")
        return plt.figure()

    frame_probs = result.frame_predictions  # (num_classes, time_frames)
    num_classes, num_frames = frame_probs.shape

    # Find top-K classes by max probability
    max_probs = frame_probs.max(axis=1)
    top_indices = np.argsort(max_probs)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create heatmap data
    heatmap_data = frame_probs[top_indices]

    if class_names:
        display_names = [class_names[i] for i in top_indices]
    else:
        display_names = [f"Class {i}" for i in top_indices]

    # Time axis
    if result.frame_times is not None:
        extent = [result.frame_times[0], result.frame_times[-1], len(top_indices) - 0.5, -0.5]
    else:
        extent = [0, result.duration, len(top_indices) - 0.5, -0.5]

    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        cmap="hot",
        vmin=0,
        vmax=1,
        extent=extent,
        interpolation="nearest",
    )

    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=9)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title("Frame-Level Predictions (Top Classes)", fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 150,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation curves.

    Shows loss, mAP, F1, precision, and recall over epochs.

    Args:
        metrics_history: Dictionary from MetricsTracker.load_history().
        figsize: Figure size.
        dpi: DPI.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    epochs = metrics_history.get("epoch", [])
    if not epochs:
        logger.warning("No training history to plot.")
        return fig

    # Plot 1: Loss
    ax = axes[0, 0]
    if "train_loss" in metrics_history:
        ax.plot(epochs, metrics_history["train_loss"], label="Train Loss", color="#1f77b4")
    if "val_loss" in metrics_history:
        ax.plot(epochs, metrics_history["val_loss"], label="Val Loss", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: mAP
    ax = axes[0, 1]
    if "mAP" in metrics_history:
        ax.plot(epochs, metrics_history["mAP"], label="mAP", color="#2ca02c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Mean Average Precision")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: F1 Score
    ax = axes[1, 0]
    if "f1_micro" in metrics_history:
        ax.plot(epochs, metrics_history["f1_micro"], label="F1 (micro)", color="#d62728")
    if "f1_macro" in metrics_history:
        ax.plot(epochs, metrics_history["f1_macro"], label="F1 (macro)", color="#9467bd")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Precision & Recall
    ax = axes[1, 1]
    if "precision_micro" in metrics_history:
        ax.plot(epochs, metrics_history["precision_micro"], label="Precision (micro)", color="#8c564b")
    if "recall_micro" in metrics_history:
        ax.plot(epochs, metrics_history["recall_micro"], label="Recall (micro)", color="#e377c2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall")
    ax.legend()
    ax.grid(alpha=0.3)

    # Learning rate subplot (if available)
    if "learning_rate" in metrics_history and any(metrics_history["learning_rate"]):
        pass  # Could add as a secondary axis if needed

    fig.suptitle("Training Metrics", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_spectrogram(
    spectrogram: np.ndarray,
    sample_rate: int = 16000,
    hop_length_ms: float = 10.0,
    figsize: Tuple[int, int] = (12, 4),
    dpi: int = 150,
    title: str = "Log-Mel Spectrogram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a log-mel spectrogram.

    Args:
        spectrogram: Spectrogram of shape (n_mels, time_frames).
        sample_rate: Audio sample rate.
        hop_length_ms: Hop length in ms (for time axis).
        figsize: Figure size.
        dpi: DPI.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    n_mels, n_time = spectrogram.shape
    duration = n_time * hop_length_ms / 1000.0

    im = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, duration, 0, n_mels],
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mel Frequency Bin")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Log Power")
    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def _save_figure(fig: plt.Figure, save_path: str) -> None:
    """Save a matplotlib figure to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=fig.dpi)
    plt.close(fig)
    logger.info(f"Figure saved: {save_path}")
