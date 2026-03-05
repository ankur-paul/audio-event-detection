"""
Inference pipeline for Audio Event Detection.

Supports:
- Single clip inference
- Sliding window inference for long audio/video
- Frame-level event detection with temporal boundaries
- Merging overlapping predictions from sliding windows
"""

import os
import numpy as np
import torch
import librosa
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.data.preprocessing import preprocess_audio_from_array, load_audio
from src.data.features import compute_log_mel_spectrogram
from src.utils.logger import get_logger

logger = get_logger()


@dataclass
class DetectedEvent:
    """Represents a detected sound event with temporal bounds."""
    class_name: str
    class_idx: int
    start_time: float  # seconds
    end_time: float    # seconds
    confidence: float  # average probability


@dataclass
class InferenceResult:
    """Full inference result for an audio file."""
    filename: str
    duration: float                          # total audio duration in seconds
    events: List[DetectedEvent] = field(default_factory=list)
    frame_predictions: Optional[np.ndarray] = None  # (num_classes, total_frames)
    frame_times: Optional[np.ndarray] = None        # time in seconds for each frame
    clip_predictions: Optional[np.ndarray] = None   # (num_classes,)


class InferencePipeline:
    """
    Inference pipeline for audio event detection.

    Supports both single-clip and sliding-window inference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        class_map: Dict[str, int],
        config,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: Trained AudioEventDetectionModel.
            class_map: Class name to index mapping.
            config: Full configuration object.
            device: Device string. Auto-detected if None.
        """
        self.model = model
        self.class_map = class_map
        self.inverse_class_map = {v: k for k, v in class_map.items()}
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Feature extraction params
        self.sample_rate = config.audio.sample_rate
        self.clip_duration = config.audio.clip_duration
        self.n_mels = config.features.n_mels
        self.window_size_ms = config.features.window_size_ms
        self.hop_length_ms = config.features.hop_length_ms
        self.f_min = config.features.f_min
        self.f_max = config.features.f_max

        # Inference params
        inf_cfg = config.inference
        self.threshold = inf_cfg.threshold
        self.window_length = inf_cfg.window_length
        self.hop_length = inf_cfg.hop_length
        self.min_event_duration = getattr(inf_cfg, "min_event_duration", 0.2)
        self.merge_overlapping = getattr(inf_cfg, "merge_overlapping", True)

    @torch.no_grad()
    def predict_clip(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on a single audio clip.

        Args:
            audio: Audio waveform (should be clip_duration seconds at sample_rate).

        Returns:
            Dictionary with:
                - 'clip_probs': (num_classes,) array
                - 'frame_probs': (num_classes, time_frames) array (if frame_level)
        """
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

        # Add batch and channel dimensions: (1, 1, n_mels, time)
        tensor = torch.from_numpy(spectrogram).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        outputs = self.model(tensor)

        result = {
            "clip_probs": outputs["clip_probs"].cpu().numpy()[0],
        }

        if "frame_probs" in outputs:
            result["frame_probs"] = outputs["frame_probs"].cpu().numpy()[0]

        return result

    def predict_file(self, audio_path: str) -> InferenceResult:
        """
        Run inference on an audio file.

        For short files (≤ clip_duration): single clip inference.
        For long files: sliding window inference.

        Args:
            audio_path: Path to audio file.

        Returns:
            InferenceResult with detected events and predictions.
        """
        # Load full audio
        audio, sr = load_audio(audio_path, target_sr=self.sample_rate, mono=True)
        duration = len(audio) / self.sample_rate

        logger.info(f"Processing {audio_path}: duration={duration:.1f}s")

        if duration <= self.clip_duration + 0.5:
            # Short clip: single inference
            return self._predict_short_clip(audio, audio_path, duration)
        else:
            # Long audio: sliding window
            return self._predict_long_audio(audio, audio_path, duration)

    def _predict_short_clip(
        self, audio: np.ndarray, filename: str, duration: float
    ) -> InferenceResult:
        """Run inference on a short audio clip."""
        from src.data.preprocessing import pad_or_trim, normalize_audio

        # Preprocess
        audio = normalize_audio(audio)
        target_samples = int(self.sample_rate * self.clip_duration)
        audio = pad_or_trim(audio, target_samples)

        # Predict
        pred = self.predict_clip(audio)

        # Extract events from clip predictions
        events = self._probs_to_events(
            pred["clip_probs"],
            start_time=0.0,
            end_time=duration,
        )

        return InferenceResult(
            filename=filename,
            duration=duration,
            events=events,
            clip_predictions=pred["clip_probs"],
            frame_predictions=pred.get("frame_probs"),
        )

    def _predict_long_audio(
        self, audio: np.ndarray, filename: str, duration: float
    ) -> InferenceResult:
        """
        Run sliding window inference on long audio.

        Windows overlap and predictions are merged to produce a continuous
        event timeline.
        """
        from src.data.preprocessing import pad_or_trim, normalize_audio

        audio = normalize_audio(audio)

        window_samples = int(self.window_length * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        total_samples = len(audio)

        # Compute spectrogram time resolution
        spec_hop = int(self.sample_rate * self.hop_length_ms / 1000)
        frames_per_window = window_samples // spec_hop

        # Calculate total output frames
        total_time_steps = int(np.ceil(duration / (self.hop_length_ms / 1000)))
        num_classes = len(self.class_map)

        # Accumulator arrays for merging overlapping windows
        frame_probs_accum = np.zeros((num_classes, total_time_steps), dtype=np.float64)
        frame_counts = np.zeros(total_time_steps, dtype=np.float64)

        # Clip-level accumulator
        clip_probs_accum = np.zeros(num_classes, dtype=np.float64)
        num_windows = 0

        # Sliding window
        start = 0
        while start < total_samples:
            end = min(start + window_samples, total_samples)
            window_audio = audio[start:end]

            # Pad if window is shorter than expected
            if len(window_audio) < window_samples:
                window_audio = pad_or_trim(window_audio, window_samples)

            # Predict
            pred = self.predict_clip(window_audio)

            # Accumulate clip predictions
            clip_probs_accum += pred["clip_probs"]
            num_windows += 1

            # Accumulate frame predictions if available
            if "frame_probs" in pred:
                frame_preds = pred["frame_probs"]  # (num_classes, window_frames)
                window_frames = frame_preds.shape[1]

                # Map window frames to global frame indices
                start_frame = int(start / spec_hop)
                end_frame = min(start_frame + window_frames, total_time_steps)
                actual_frames = end_frame - start_frame

                if actual_frames > 0:
                    frame_probs_accum[:, start_frame:end_frame] += frame_preds[:, :actual_frames]
                    frame_counts[start_frame:end_frame] += 1.0

            start += hop_samples

        # Average overlapping predictions
        clip_probs = clip_probs_accum / max(num_windows, 1)

        mask = frame_counts > 0
        frame_probs_accum[:, mask] /= frame_counts[mask]

        # Generate frame time array
        frame_times = np.arange(total_time_steps) * (self.hop_length_ms / 1000)

        # Extract events from frame predictions
        events = self._frame_probs_to_events(
            frame_probs_accum, frame_times
        )

        logger.info(
            f"Sliding window: {num_windows} windows, "
            f"{len(events)} events detected"
        )

        return InferenceResult(
            filename=filename,
            duration=duration,
            events=events,
            frame_predictions=frame_probs_accum.astype(np.float32),
            frame_times=frame_times,
            clip_predictions=clip_probs.astype(np.float32),
        )

    def _probs_to_events(
        self,
        probs: np.ndarray,
        start_time: float,
        end_time: float,
    ) -> List[DetectedEvent]:
        """Convert clip-level probabilities to events."""
        events = []
        for idx, prob in enumerate(probs):
            if prob >= self.threshold:
                events.append(DetectedEvent(
                    class_name=self.inverse_class_map.get(idx, f"class_{idx}"),
                    class_idx=idx,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=float(prob),
                ))
        return events

    def _frame_probs_to_events(
        self,
        frame_probs: np.ndarray,
        frame_times: np.ndarray,
    ) -> List[DetectedEvent]:
        """
        Convert frame-level probabilities to detected events with temporal bounds.

        Finds contiguous segments where probability exceeds threshold.
        """
        num_classes, num_frames = frame_probs.shape
        events = []

        frame_dt = frame_times[1] - frame_times[0] if len(frame_times) > 1 else 0.01

        for class_idx in range(num_classes):
            probs = frame_probs[class_idx]
            active = probs >= self.threshold

            # Find contiguous active segments
            segments = self._find_segments(active)

            for seg_start, seg_end in segments:
                start_time = frame_times[seg_start] if seg_start < len(frame_times) else 0
                end_time = (
                    frame_times[min(seg_end, len(frame_times) - 1)] + frame_dt
                )
                event_duration = end_time - start_time

                # Skip very short events
                if event_duration < self.min_event_duration:
                    continue

                confidence = float(np.mean(probs[seg_start : seg_end + 1]))

                events.append(DetectedEvent(
                    class_name=self.inverse_class_map.get(class_idx, f"class_{class_idx}"),
                    class_idx=class_idx,
                    start_time=float(start_time),
                    end_time=float(end_time),
                    confidence=confidence,
                ))

        # Sort events by start time
        events.sort(key=lambda e: (e.start_time, e.class_name))

        # Optionally merge overlapping events of the same class
        if self.merge_overlapping:
            events = self._merge_events(events)

        return events

    @staticmethod
    def _find_segments(active: np.ndarray) -> List[Tuple[int, int]]:
        """Find contiguous True segments in a boolean array."""
        segments = []
        in_segment = False
        start = 0

        for i, val in enumerate(active):
            if val and not in_segment:
                start = i
                in_segment = True
            elif not val and in_segment:
                segments.append((start, i - 1))
                in_segment = False

        if in_segment:
            segments.append((start, len(active) - 1))

        return segments

    @staticmethod
    def _merge_events(events: List[DetectedEvent]) -> List[DetectedEvent]:
        """Merge overlapping events of the same class."""
        if not events:
            return events

        # Group by class
        class_events = {}
        for event in events:
            if event.class_name not in class_events:
                class_events[event.class_name] = []
            class_events[event.class_name].append(event)

        merged = []
        for class_name, evts in class_events.items():
            evts.sort(key=lambda e: e.start_time)

            current = evts[0]
            for evt in evts[1:]:
                if evt.start_time <= current.end_time:
                    # Merge
                    current = DetectedEvent(
                        class_name=class_name,
                        class_idx=current.class_idx,
                        start_time=current.start_time,
                        end_time=max(current.end_time, evt.end_time),
                        confidence=(current.confidence + evt.confidence) / 2,
                    )
                else:
                    merged.append(current)
                    current = evt
            merged.append(current)

        merged.sort(key=lambda e: (e.start_time, e.class_name))
        return merged
