"""
Audio preprocessing module for Audio Event Detection.

Handles:
- Resampling to target sample rate
- Mono conversion
- Amplitude normalization
- Trimming/padding to fixed duration
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger()


def load_audio(
    file_path: str,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, resample, and convert to mono.

    Args:
        file_path: Path to audio file.
        target_sr: Target sample rate in Hz.
        mono: Whether to convert to mono.

    Returns:
        Tuple of (audio_waveform, sample_rate).
    """
    audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
    return audio, sr


def normalize_audio(audio: np.ndarray, target_db: Optional[float] = None) -> np.ndarray:
    """
    Normalize audio amplitude.

    If target_db is None, performs peak normalization to [-1, 1].
    If target_db is specified, normalizes to target loudness in dB.

    Args:
        audio: Audio waveform array.
        target_db: Target loudness in dB (optional).

    Returns:
        Normalized audio array.
    """
    if target_db is not None:
        # RMS normalization
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20.0)
            audio = audio * (target_rms / rms)
    else:
        # Peak normalization
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

    return audio


def pad_or_trim(
    audio: np.ndarray,
    target_length: int,
    pad_mode: str = "constant",
) -> np.ndarray:
    """
    Pad or trim audio to a fixed length.

    Args:
        audio: Audio waveform array.
        target_length: Target number of samples.
        pad_mode: Padding mode ('constant', 'reflect', 'wrap').

    Returns:
        Audio array of exactly target_length samples.
    """
    current_length = len(audio)

    if current_length > target_length:
        # Trim from the center
        start = (current_length - target_length) // 2
        audio = audio[start : start + target_length]
    elif current_length < target_length:
        # Pad symmetrically
        pad_total = target_length - current_length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode=pad_mode)

    return audio


def preprocess_audio(
    file_path: str,
    sample_rate: int = 16000,
    clip_duration: float = 5.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single audio file.

    Steps:
    1. Load and resample to target sample rate
    2. Convert to mono
    3. Normalize amplitude
    4. Pad or trim to fixed duration

    Args:
        file_path: Path to the audio file.
        sample_rate: Target sample rate.
        clip_duration: Target clip duration in seconds.
        normalize: Whether to normalize amplitude.

    Returns:
        Preprocessed audio waveform of shape (num_samples,).
    """
    # Load audio (librosa handles resampling and mono conversion)
    audio, sr = load_audio(file_path, target_sr=sample_rate, mono=True)

    # Normalize
    if normalize:
        audio = normalize_audio(audio)

    # Pad or trim to target length
    target_length = int(sample_rate * clip_duration)
    audio = pad_or_trim(audio, target_length)

    return audio


def preprocess_audio_from_array(
    audio: np.ndarray,
    sample_rate: int = 16000,
    clip_duration: float = 5.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess an already-loaded audio array.

    Args:
        audio: Raw audio waveform.
        sample_rate: Sample rate of the audio.
        clip_duration: Target clip duration in seconds.
        normalize: Whether to normalize amplitude.

    Returns:
        Preprocessed audio waveform.
    """
    if normalize:
        audio = normalize_audio(audio)

    target_length = int(sample_rate * clip_duration)
    audio = pad_or_trim(audio, target_length)

    return audio
