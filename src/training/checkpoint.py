"""
Checkpointing utilities for Audio Event Detection.

Provides helpers for loading model weights from Lightning checkpoints
at inference time, and for locating the best / latest checkpoint.
"""

import os
import glob
import torch
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger()


def find_best_checkpoint(checkpoint_dir: str, drive_dir: Optional[str] = None) -> Optional[str]:
    """
    Find the best model checkpoint for inference.

    Search order (first match wins):
        1. ``best_model.ckpt`` on Google Drive  (best validation metric)
        2. ``best_model.ckpt`` in local checkpoint dir
        3. ``last.ckpt`` in local checkpoint dir
        4. Any ``.ckpt`` in local dir (newest by mtime)
        5. ``latest.ckpt`` on Google Drive
    """
    # 1. Best model on Drive (preferred — survives runtime restarts)
    if drive_dir:
        best_drive = Path(drive_dir) / "best_model.ckpt"
        if best_drive.exists():
            return str(best_drive)

    ckpt_dir = Path(checkpoint_dir)

    # 2. Best model locally
    best_local = ckpt_dir / "best_model.ckpt"
    if best_local.exists():
        return str(best_local)

    # 3. last.ckpt saved by Lightning
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return str(last)

    # 4. Any .ckpt sorted by modification time (newest = latest epoch)
    pattern = str(ckpt_dir / "*.ckpt")
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    if ckpts:
        return ckpts[-1]

    # 5. Latest on Drive as last resort
    if drive_dir:
        latest_drive = Path(drive_dir) / "latest.ckpt"
        if latest_drive.exists():
            return str(latest_drive)

    return None


def load_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load model weights from a Lightning ``.ckpt`` file.

    Lightning stores model weights under the ``"state_dict"`` key, with
    each key prefixed by ``"model."``.  This function strips that prefix
    before loading into the raw ``AudioEventDetectionModel``.

    Args:
        model: The model to load weights into.
        checkpoint_path: Path to a ``.ckpt`` file saved by Lightning.
        device: Target device for the loaded tensors.

    Returns:
        The full checkpoint dictionary (for accessing ``epoch``, ``global_step``, etc.).
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", {}))

    # Strip the "model." prefix added by Lightning's LightningModule wrapper
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("model.")
        cleaned[new_key] = value

    model.load_state_dict(cleaned)

    epoch = checkpoint.get("epoch", 0)
    logger.info(f"Loaded model from epoch {epoch}")
    return checkpoint
