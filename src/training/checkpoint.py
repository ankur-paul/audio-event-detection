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
    Find the best model checkpoint (saved by Lightning's ModelCheckpoint).

    Looks for ``last.ckpt`` or the highest-mAP checkpoint in *checkpoint_dir*,
    falling back to *drive_dir* if nothing is found locally.
    """
    ckpt_dir = Path(checkpoint_dir)

    # 1. Explicit "last.ckpt" saved by Lightning
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return str(last)

    # 2. Any .ckpt sorted by modification time (newest = latest epoch)
    pattern = str(ckpt_dir / "*.ckpt")
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    if ckpts:
        return ckpts[-1]

    # 3. Google Drive fallback
    if drive_dir:
        for name in ("best_model.ckpt", "latest.ckpt"):
            p = Path(drive_dir) / name
            if p.exists():
                return str(p)

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
