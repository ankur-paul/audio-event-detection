# Audio Event Detection

Multi-label audio event detection system using EfficientNet-B0 for detecting overlapping environmental sound events. Designed for resumable training across Google Colab sessions.

## Features

- **Multi-label detection**: Detects ~47 overlapping sound event classes simultaneously
- **Frame-level predictions**: Weakly-supervised temporal localization of events within clips
- **Resumable training**: Full checkpoint system for Google Colab persistence
- **Sliding window inference**: Handles long audio recordings (minutes to hours)
- **Event timeline visualization**: Visual output showing detected events over time
- **Modular design**: Easy to swap backbones, add classes, or modify components

## Project Structure

```
audio-event-detection/
├── configs/
│   └── default.yaml            # Default configuration
├── src/
│   ├── data/
│   │   ├── dataset_preparation.py   # Class maps, label parsing, data splitting
│   │   ├── preprocessing.py         # Audio loading, resampling, normalization
│   │   ├── features.py              # Log-mel spectrogram extraction
│   │   ├── augmentation.py          # Time-domain & spectrogram augmentations
│   │   └── dataset.py               # PyTorch Dataset & DataLoader
│   ├── models/
│   │   └── audio_event_model.py     # EfficientNet + frame-level SED model
│   ├── training/
│   │   ├── trainer.py               # Full training loop with AMP & scheduling
│   │   ├── checkpoint.py            # Checkpoint save/load/Drive sync
│   │   ├── metrics.py               # Precision, Recall, F1, mAP
│   │   └── experiment_tracker.py    # CSV-based metrics logging
│   ├── inference/
│   │   └── inference_pipeline.py    # Single-clip & sliding-window inference
│   ├── visualization/
│   │   └── visualizer.py            # Timeline, heatmap, training curve plots
│   └── utils/
│       ├── config.py                # YAML config loader
│       └── logger.py                # Logging setup
├── data/
│   ├── raw/                    # Raw audio files
│   ├── processed/              # Preprocessed audio
│   └── spectrograms/           # Pre-computed .npy spectrograms
├── checkpoints/                # Model checkpoints
├── logs/                       # Training logs & metrics CSV
├── notebooks/                  # Jupyter/Colab notebooks
├── scripts/
│   ├── label_mapping.py             # Label mappings for all source datasets
│   ├── download_esc50.py            # Download & prepare ESC-50
│   ├── download_fsd50k.py           # Download & prepare FSD50K
│   ├── download_urbansound8k.py     # Prepare UrbanSound8K (manual download)
│   └── prepare_multi_dataset.py     # Merge all datasets into unified format
├── train.py                    # Training entry point
├── predict.py                  # Inference entry point
├── prepare_data.py             # Data preparation entry point
├── visualize_metrics.py        # Plot training curves
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Training Data (Multi-Dataset)

The system combines multiple public audio datasets, mapping their labels to our 47-class taxonomy:

| Dataset          | Size              | Auto-download | Classes covered |
| ---------------- | ----------------- | ------------- | --------------- |
| **ESC-50**       | 2k clips, ~600 MB | Yes           | ~30             |
| **FSD50K**       | 51k clips, ~30 GB | Yes           | ~45             |
| **UrbanSound8K** | 8.7k clips, ~6 GB | Manual        | ~9              |

**Step 2a: Download ESC-50** (fastest, good starting point):

```bash
python scripts/download_esc50.py
```

**Step 2b: Download FSD50K** (best coverage, but large):

```bash
# Download metadata only first (small)
python scripts/download_fsd50k.py --metadata-only

# Download dev audio (~22 GB)
python scripts/download_fsd50k.py --dev-only

# Download everything (~30 GB)
python scripts/download_fsd50k.py
```

**Step 2c: UrbanSound8K** (manual download):

1. Visit https://urbansounddataset.weebly.com/urbansound8k.html
2. Download and extract to `data/downloads/urbansound8k/UrbanSound8K/`
3. Run: `python scripts/download_urbansound8k.py`

**Step 2d: Merge all datasets** into unified format:

```bash
# Merge whichever datasets you downloaded
python scripts/prepare_multi_dataset.py

# Optionally limit samples per class for balanced training
python scripts/prepare_multi_dataset.py --max-per-class 500
```

This produces `data/labels.csv` and copies all audio to `data/raw/`.

### 3. Prepare Features

```bash
python prepare_data.py
```

This will:

- Create the class map (`data/class_map.json`)
- Extract log-mel spectrograms as `.npy` files
- Split data into train/val/test sets

> **Tip:** If you only want to use your own audio, place files in `data/raw/` and create `data/labels.csv` manually:
>
> ```csv
> filename,labels
> clip001.wav,speech|footsteps
> clip002.wav,rain
> ```

### 4. Train the Model

```bash
# Start fresh training
python train.py

# Train with custom config
python train.py --config configs/custom.yaml

# Resume from checkpoint (critical for Colab sessions)
python train.py --resume

# Override parameters via CLI
python train.py --epochs 50 --batch-size 16 --lr 3e-4
```

### 5. Resume Training (Google Colab)

```python
# In a new Colab session:
!python train.py --resume
```

The system automatically:

- Finds the latest checkpoint
- Restores model weights, optimizer state, and scheduler
- Continues from the last completed epoch
- Preserves all training history

For Google Drive persistence, set `paths.drive_checkpoint_dir` in your config:

```yaml
paths:
  drive_checkpoint_dir: "/content/drive/MyDrive/audio-event-detection/checkpoints"
```

### 6. Run Inference

```bash
# Inference on a single audio file
python predict.py --audio path/to/audio.wav

# With visualization
python predict.py --audio path/to/audio.wav --visualize

# Use best model checkpoint
python predict.py --audio path/to/audio.wav --checkpoint best

# Custom threshold
python predict.py --audio path/to/audio.wav --threshold 0.3 --visualize
```

### 7. Visualize Training Curves

```bash
python visualize_metrics.py --output plots/
```

## Model Architecture

```
Input: Log-Mel Spectrogram (1, 128, ~500)
  ↓
EfficientNet-B0 Feature Extractor
  ↓
Temporal Feature Maps (channels, time_steps)
  ↓
Frame-Level Classifier (1D Conv)
  ↓
Frame Predictions (num_classes, time_steps)  ← used for event timeline
  ↓
Temporal Pooling (attention / max / mean)
  ↓
Clip Predictions (num_classes)  ← used for training loss (BCE)
  ↓
Sigmoid Activation
```

The model is trained with **clip-level BCE loss** (weak supervision) but produces **frame-level predictions** for temporal event localization during inference.

## Configuration

All parameters are controlled via YAML configs in `configs/`. Key sections:

| Section        | Key Parameters                               |
| -------------- | -------------------------------------------- |
| `audio`        | sample_rate=16000, clip_duration=5.0         |
| `features`     | n_mels=128, window=25ms, hop=10ms            |
| `model`        | backbone=efficientnet_b0, pooling=attention  |
| `training`     | lr=1e-4, batch_size=32, mixed_precision=true |
| `augmentation` | time_shift, noise, pitch_shift, SpecAugment  |
| `inference`    | window=5s, hop=1s, threshold=0.5             |

## Sound Classes (~50)

The system includes 47 environmental sound events covering speech, animals, vehicles, weather, tools, and household sounds. See `src/data/dataset_preparation.py` for the full list.

## Extending the System

- **Larger backbone**: Change `model.backbone` to `efficientnet_b2`, `efficientnet_b4`, etc.
- **Transformer backbone**: Replace the backbone in `AudioEventDetectionModel._build_backbone()`
- **More classes**: Add entries to `DEFAULT_SOUND_CLASSES` and update `model.num_classes`
- **Real-time inference**: Use `InferencePipeline.predict_clip()` in a streaming loop
