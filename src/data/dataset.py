"""
PyTorch Dataset and DataLoader for Audio Event Detection.

Supports loading from:
- Pre-computed spectrogram .npy files (fast)
- Raw audio files with on-the-fly feature extraction (flexible)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.dataset_preparation import (
    labels_to_binary_vector,
    parse_labels_csv,
    load_class_map,
)
from src.data.preprocessing import preprocess_audio
from src.data.features import compute_log_mel_spectrogram
from src.data.augmentation import AudioAugmentor
from src.utils.logger import get_logger

logger = get_logger()


class AudioEventDataset(Dataset):
    """
    PyTorch Dataset for multi-label audio event detection.

    Supports two modes:
    1. Spectrogram mode: loads pre-computed .npy spectrograms (faster training)
    2. Audio mode: loads raw audio and computes spectrograms on-the-fly
    """

    def __init__(
        self,
        entries: List[Dict],
        class_map: Dict[str, int],
        spectrogram_dir: Optional[str] = None,
        audio_dir: Optional[str] = None,
        sample_rate: int = 16000,
        clip_duration: float = 5.0,
        n_mels: int = 128,
        window_size_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        f_min: float = 50.0,
        f_max: float = 8000.0,
        augmentor: Optional[AudioAugmentor] = None,
        is_training: bool = True,
    ):
        """
        Args:
            entries: List of dicts with 'filename' and 'labels'.
            class_map: Mapping from class name to index.
            spectrogram_dir: Directory with pre-computed .npy files.
            audio_dir: Directory with raw audio files (used if spectrogram_dir is None).
            sample_rate: Target sample rate.
            clip_duration: Target clip duration in seconds.
            n_mels: Number of mel bins.
            window_size_ms: Spectrogram window size in ms.
            hop_length_ms: Spectrogram hop length in ms.
            f_min: Min frequency for mel filterbank.
            f_max: Max frequency for mel filterbank.
            augmentor: Optional AudioAugmentor for on-the-fly augmentation.
            is_training: Whether this is a training dataset (enables augmentation).
        """
        self.entries = entries
        self.class_map = class_map
        self.num_classes = len(class_map)
        self.spectrogram_dir = spectrogram_dir
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.n_mels = n_mels
        self.window_size_ms = window_size_ms
        self.hop_length_ms = hop_length_ms
        self.f_min = f_min
        self.f_max = f_max
        self.augmentor = augmentor
        self.is_training = is_training

        # Determine mode
        self.use_precomputed = spectrogram_dir is not None and os.path.isdir(
            spectrogram_dir
        )
        if not self.use_precomputed and audio_dir is None:
            raise ValueError(
                "Either spectrogram_dir (with pre-computed .npy files) or "
                "audio_dir must be provided."
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]
        filename = entry["filename"]
        labels = entry["labels"]

        # Get label vector
        label_vector = labels_to_binary_vector(labels, self.class_map)

        # Load spectrogram
        if self.use_precomputed:
            spec_path = os.path.join(
                self.spectrogram_dir,
                Path(filename).stem + ".npy",
            )
            spectrogram = np.load(spec_path)

            # Apply spectrogram augmentation
            if self.is_training and self.augmentor is not None:
                spectrogram = self.augmentor.augment_spectrogram(spectrogram)
        else:
            # Load and preprocess audio
            audio_path = os.path.join(self.audio_dir, filename)
            audio = preprocess_audio(
                audio_path,
                sample_rate=self.sample_rate,
                clip_duration=self.clip_duration,
            )

            # Apply waveform augmentation
            if self.is_training and self.augmentor is not None:
                audio = self.augmentor.augment_waveform(
                    audio, sample_rate=self.sample_rate
                )

            # Compute spectrogram
            spectrogram = compute_log_mel_spectrogram(
                audio,
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                window_size_ms=self.window_size_ms,
                hop_length_ms=self.hop_length_ms,
                f_min=self.f_min,
                f_max=self.f_max,
            )

            # Apply spectrogram augmentation
            if self.is_training and self.augmentor is not None:
                spectrogram = self.augmentor.augment_spectrogram(spectrogram)

        # Add channel dimension: (1, n_mels, time_frames) for CNN input
        spectrogram = spectrogram[np.newaxis, :, :]

        return (
            torch.from_numpy(spectrogram).float(),
            torch.from_numpy(label_vector).float(),
        )


def create_dataloaders(
    train_entries: List[Dict],
    val_entries: List[Dict],
    class_map: Dict[str, int],
    config,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    Args:
        train_entries: Training dataset entries.
        val_entries: Validation dataset entries.
        class_map: Class name to index mapping.
        config: Full configuration object.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Create augmentor
    augmentor = None
    if hasattr(config, "augmentation"):
        augmentor = AudioAugmentor(config.augmentation)

    # Common feature params
    feat_cfg = config.features
    audio_cfg = config.audio

    common_kwargs = dict(
        class_map=class_map,
        spectrogram_dir=config.paths.spectrogram_dir,
        audio_dir=config.paths.processed_audio_dir,
        sample_rate=audio_cfg.sample_rate,
        clip_duration=audio_cfg.clip_duration,
        n_mels=feat_cfg.n_mels,
        window_size_ms=feat_cfg.window_size_ms,
        hop_length_ms=feat_cfg.hop_length_ms,
        f_min=feat_cfg.f_min,
        f_max=feat_cfg.f_max,
    )

    train_dataset = AudioEventDataset(
        entries=train_entries,
        augmentor=augmentor,
        is_training=True,
        **common_kwargs,
    )

    val_dataset = AudioEventDataset(
        entries=val_entries,
        augmentor=None,
        is_training=False,
        **common_kwargs,
    )

    train_cfg = config.training

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        drop_last=False,
    )

    logger.info(
        f"Created DataLoaders: train={len(train_dataset)} samples, "
        f"val={len(val_dataset)} samples, batch_size={train_cfg.batch_size}"
    )

    return train_loader, val_loader
