"""
Feature extraction module for Audio Event Detection.

Converts audio waveforms to log-mel spectrograms.
"""

import os
import numpy as np
import librosa
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger()


def compute_log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 128,
    window_size_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    f_min: float = 50.0,
    f_max: float = 8000.0,
    log_offset: float = 1e-6,
) -> np.ndarray:
    """
    Compute a log-mel spectrogram from an audio waveform.

    Args:
        audio: Audio waveform array.
        sample_rate: Sample rate in Hz.
        n_mels: Number of mel filter banks.
        window_size_ms: Window size in milliseconds.
        hop_length_ms: Hop length in milliseconds.
        f_min: Minimum frequency in Hz.
        f_max: Maximum frequency in Hz.
        log_offset: Small value added before log to avoid log(0).

    Returns:
        Log-mel spectrogram of shape (n_mels, time_frames).
    """
    # Convert ms to samples
    n_fft = int(sample_rate * window_size_ms / 1000.0)
    hop_length = int(sample_rate * hop_length_ms / 1000.0)

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0,
    )

    # Convert to log scale
    log_mel_spec = np.log(mel_spec + log_offset)

    return log_mel_spec.astype(np.float32)


def extract_and_save_spectrogram(
    audio: np.ndarray,
    save_path: str,
    sample_rate: int = 16000,
    n_mels: int = 128,
    window_size_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    f_min: float = 50.0,
    f_max: float = 8000.0,
    log_offset: float = 1e-6,
) -> np.ndarray:
    """
    Extract a log-mel spectrogram and save it as a .npy file.

    Args:
        audio: Audio waveform.
        save_path: Path to save the .npy file.
        (other args same as compute_log_mel_spectrogram)

    Returns:
        The computed log-mel spectrogram.
    """
    spec = compute_log_mel_spectrogram(
        audio=audio,
        sample_rate=sample_rate,
        n_mels=n_mels,
        window_size_ms=window_size_ms,
        hop_length_ms=hop_length_ms,
        f_min=f_min,
        f_max=f_max,
        log_offset=log_offset,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, spec)

    return spec


def batch_extract_spectrograms(
    audio_dir: str,
    output_dir: str,
    sample_rate: int = 16000,
    clip_duration: float = 5.0,
    n_mels: int = 128,
    window_size_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    f_min: float = 50.0,
    f_max: float = 8000.0,
    log_offset: float = 1e-6,
) -> int:
    """
    Extract spectrograms for all audio files in a directory.

    Args:
        audio_dir: Directory containing audio files.
        output_dir: Directory to save .npy spectrogram files.
        (other args same as compute_log_mel_spectrogram)

    Returns:
        Number of spectrograms extracted.
    """
    from src.data.preprocessing import preprocess_audio

    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = [
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in audio_extensions
    ]

    count = 0
    for audio_file in sorted(audio_files):
        try:
            audio = preprocess_audio(
                str(audio_file),
                sample_rate=sample_rate,
                clip_duration=clip_duration,
            )

            save_path = output_dir / f"{audio_file.stem}.npy"
            extract_and_save_spectrogram(
                audio=audio,
                save_path=str(save_path),
                sample_rate=sample_rate,
                n_mels=n_mels,
                window_size_ms=window_size_ms,
                hop_length_ms=hop_length_ms,
                f_min=f_min,
                f_max=f_max,
                log_offset=log_offset,
            )
            count += 1
        except Exception as e:
            logger.error(f"Failed to process {audio_file.name}: {e}")

    logger.info(f"Extracted {count} spectrograms to {output_dir}")
    return count
