"""
Data augmentation module for Audio Event Detection.

Supports:
- Time-domain augmentations (time shift, noise injection, pitch shift, time stretch)
- Mixup of two audio clips
- SpecAugment for spectrogram masking

All augmentations are modular and configurable.
"""

import numpy as np
import librosa
from typing import Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger()


# =============================================================================
# Time-domain augmentations (applied to raw waveforms)
# =============================================================================


def time_shift(
    audio: np.ndarray,
    max_shift_samples: int,
) -> np.ndarray:
    """
    Randomly shift audio in time (circular shift).

    Args:
        audio: Audio waveform.
        max_shift_samples: Maximum shift in samples.

    Returns:
        Time-shifted audio.
    """
    shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
    return np.roll(audio, shift)


def add_noise(
    audio: np.ndarray,
    min_snr_db: float = 5.0,
    max_snr_db: float = 20.0,
) -> np.ndarray:
    """
    Add random Gaussian noise at a random SNR level.

    Args:
        audio: Audio waveform.
        min_snr_db: Minimum signal-to-noise ratio in dB.
        max_snr_db: Maximum signal-to-noise ratio in dB.

    Returns:
        Audio with added noise.
    """
    snr_db = np.random.uniform(min_snr_db, max_snr_db)
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape)
    return (audio + noise).astype(np.float32)


def pitch_shift(
    audio: np.ndarray,
    sample_rate: int = 16000,
    min_semitones: float = -2.0,
    max_semitones: float = 2.0,
) -> np.ndarray:
    """
    Randomly shift pitch of the audio.

    Args:
        audio: Audio waveform.
        sample_rate: Sample rate in Hz.
        min_semitones: Minimum pitch shift in semitones.
        max_semitones: Maximum pitch shift in semitones.

    Returns:
        Pitch-shifted audio.
    """
    n_steps = np.random.uniform(min_semitones, max_semitones)
    shifted = librosa.effects.pitch_shift(
        y=audio, sr=sample_rate, n_steps=n_steps
    )
    return shifted.astype(np.float32)


def time_stretch(
    audio: np.ndarray,
    min_rate: float = 0.8,
    max_rate: float = 1.2,
    target_length: Optional[int] = None,
) -> np.ndarray:
    """
    Randomly stretch or compress audio in time.

    Args:
        audio: Audio waveform.
        min_rate: Minimum stretch rate (< 1 = slower).
        max_rate: Maximum stretch rate (> 1 = faster).
        target_length: If set, pad/trim to this length after stretching.

    Returns:
        Time-stretched audio.
    """
    rate = np.random.uniform(min_rate, max_rate)
    stretched = librosa.effects.time_stretch(y=audio, rate=rate)

    if target_length is not None:
        if len(stretched) > target_length:
            stretched = stretched[:target_length]
        elif len(stretched) < target_length:
            pad = target_length - len(stretched)
            stretched = np.pad(stretched, (0, pad), mode="constant")

    return stretched.astype(np.float32)


def mixup(
    audio1: np.ndarray,
    labels1: np.ndarray,
    audio2: np.ndarray,
    labels2: np.ndarray,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mix two audio samples and their labels (mixup augmentation).

    Args:
        audio1: First audio waveform.
        labels1: Binary label vector for first sample.
        audio2: Second audio waveform.
        labels2: Binary label vector for second sample.
        alpha: Beta distribution parameter for mixing ratio.

    Returns:
        Tuple of (mixed_audio, mixed_labels).
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5

    # Ensure same length
    min_len = min(len(audio1), len(audio2))
    mixed_audio = lam * audio1[:min_len] + (1 - lam) * audio2[:min_len]
    mixed_labels = np.clip(labels1 + labels2, 0, 1).astype(np.float32)

    return mixed_audio.astype(np.float32), mixed_labels


# =============================================================================
# Spectrogram-domain augmentations
# =============================================================================


def spec_augment(
    spectrogram: np.ndarray,
    freq_mask_param: int = 20,
    time_mask_param: int = 50,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> np.ndarray:
    """
    Apply SpecAugment to a spectrogram.

    Applies frequency and time masking to the spectrogram.

    Args:
        spectrogram: Log-mel spectrogram of shape (n_mels, time_frames).
        freq_mask_param: Maximum number of frequency bins to mask.
        time_mask_param: Maximum number of time frames to mask.
        num_freq_masks: Number of frequency masks to apply.
        num_time_masks: Number of time masks to apply.

    Returns:
        Augmented spectrogram (same shape).
    """
    spec = spectrogram.copy()
    n_mels, n_time = spec.shape

    # Frequency masking
    for _ in range(num_freq_masks):
        f = np.random.randint(0, min(freq_mask_param, n_mels))
        f0 = np.random.randint(0, max(n_mels - f, 1))
        spec[f0 : f0 + f, :] = 0.0

    # Time masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, min(time_mask_param, n_time))
        t0 = np.random.randint(0, max(n_time - t, 1))
        spec[:, t0 : t0 + t] = 0.0

    return spec


# =============================================================================
# Augmentation pipeline
# =============================================================================


class AudioAugmentor:
    """
    Configurable audio augmentation pipeline.

    Applies a sequence of augmentations based on configuration.
    """

    def __init__(self, config):
        """
        Args:
            config: Augmentation config subtree (config.augmentation).
        """
        self.config = config
        self.enabled = getattr(config, "enabled", True)

    def augment_waveform(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Apply waveform-level augmentations.

        Args:
            audio: Audio waveform.
            sample_rate: Sample rate.

        Returns:
            Augmented audio waveform.
        """
        if not self.enabled:
            return audio

        target_length = len(audio)

        # Time shift
        ts_cfg = self.config.get("time_shift")
        if ts_cfg and getattr(ts_cfg, "enabled", False):
            max_shift = int(getattr(ts_cfg, "max_shift_ms", 500) * sample_rate / 1000)
            if np.random.random() < 0.5:
                audio = time_shift(audio, max_shift)

        # Noise injection
        noise_cfg = self.config.get("noise_injection")
        if noise_cfg and getattr(noise_cfg, "enabled", False):
            if np.random.random() < 0.5:
                audio = add_noise(
                    audio,
                    min_snr_db=getattr(noise_cfg, "min_snr_db", 5),
                    max_snr_db=getattr(noise_cfg, "max_snr_db", 20),
                )

        # Pitch shift
        ps_cfg = self.config.get("pitch_shift")
        if ps_cfg and getattr(ps_cfg, "enabled", False):
            if np.random.random() < 0.3:
                audio = pitch_shift(
                    audio,
                    sample_rate=sample_rate,
                    min_semitones=getattr(ps_cfg, "min_semitones", -2),
                    max_semitones=getattr(ps_cfg, "max_semitones", 2),
                )

        # Time stretch
        stretch_cfg = self.config.get("time_stretch")
        if stretch_cfg and getattr(stretch_cfg, "enabled", False):
            if np.random.random() < 0.3:
                audio = time_stretch(
                    audio,
                    min_rate=getattr(stretch_cfg, "min_rate", 0.8),
                    max_rate=getattr(stretch_cfg, "max_rate", 1.2),
                    target_length=target_length,
                )

        return audio

    def augment_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Apply spectrogram-level augmentations (SpecAugment).

        Args:
            spectrogram: Log-mel spectrogram of shape (n_mels, time_frames).

        Returns:
            Augmented spectrogram.
        """
        if not self.enabled:
            return spectrogram

        sa_cfg = self.config.get("spec_augment")
        if sa_cfg and getattr(sa_cfg, "enabled", False):
            spectrogram = spec_augment(
                spectrogram,
                freq_mask_param=getattr(sa_cfg, "freq_mask_param", 20),
                time_mask_param=getattr(sa_cfg, "time_mask_param", 50),
                num_freq_masks=getattr(sa_cfg, "num_freq_masks", 2),
                num_time_masks=getattr(sa_cfg, "num_time_masks", 2),
            )

        return spectrogram
