"""
Unified multi-dataset preparation script.

Merges labels from ESC-50, UrbanSound8K, and FSD50K into a single
dataset, copies/resamples audio files, and generates the final labels.csv.

This script:
1. Reads per-dataset label CSVs (produced by the individual download scripts)
2. Copies and resamples all audio to data/raw/ (16kHz mono WAV)
3. Generates the unified data/labels.csv
4. Prints class distribution statistics
5. Optionally balances classes by oversampling/limiting

Usage:
    python scripts/prepare_multi_dataset.py
    python scripts/prepare_multi_dataset.py --max-per-class 500
    python scripts/prepare_multi_dataset.py --skip-audio-copy  # just merge labels
"""

import argparse
import csv
import json
import os
import sys
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.label_mapping import TARGET_CLASSES


def load_dataset_labels(csv_path: str) -> list:
    """Load a per-dataset labels CSV."""
    if not os.path.exists(csv_path):
        return []

    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "filename": row["filename"],
                "labels": row["labels"],
                "source": row.get("source", "unknown"),
                "source_path": row.get("source_path", ""),
            })
    return entries


def copy_and_resample_audio(
    source_path: str,
    dest_path: str,
    target_sr: int = 16000,
) -> bool:
    """
    Copy an audio file, resampling to target sample rate and converting to mono WAV.

    Returns True on success.
    """
    try:
        import librosa
        import soundfile as sf

        # Load and resample
        audio, sr = librosa.load(source_path, sr=target_sr, mono=True)

        # Save as WAV
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        sf.write(dest_path, audio, target_sr, subtype="PCM_16")
        return True

    except Exception as e:
        print(f"  ERROR processing {source_path}: {e}")
        return False


def merge_datasets(
    dataset_csvs: dict,
    output_dir: str,
    raw_audio_dir: str,
    max_per_class: int = 0,
    skip_audio_copy: bool = False,
    target_sr: int = 16000,
) -> str:
    """
    Merge multiple dataset label CSVs into a unified dataset.

    Args:
        dataset_csvs: Dict of {dataset_name: csv_path}.
        output_dir: Data output directory.
        raw_audio_dir: Directory to copy audio files to.
        max_per_class: Max samples per class (0 = unlimited).
        skip_audio_copy: If True, only merge labels without copying audio.
        target_sr: Target sample rate for audio files.

    Returns:
        Path to the output labels.csv.
    """
    # Load all entries
    all_entries = []
    for name, csv_path in dataset_csvs.items():
        entries = load_dataset_labels(csv_path)
        if entries:
            print(f"  Loaded {len(entries)} entries from {name}")
            all_entries.extend(entries)
        else:
            print(f"  Skipping {name} (no labels CSV or empty)")

    if not all_entries:
        print("ERROR: No entries found from any dataset!")
        return ""

    print(f"\nTotal entries before filtering: {len(all_entries)}")

    # Remove duplicates (by filename)
    seen = set()
    unique_entries = []
    for entry in all_entries:
        if entry["filename"] not in seen:
            seen.add(entry["filename"])
            unique_entries.append(entry)
    print(f"After deduplication: {len(unique_entries)}")

    # Class distribution before balancing
    class_counts = Counter()
    for entry in unique_entries:
        for label in entry["labels"].split("|"):
            class_counts[label] += 1

    print(f"\nClass distribution ({len(class_counts)} classes):")
    print("-" * 50)
    for cls in sorted(class_counts.keys()):
        bar = "█" * min(50, class_counts[cls] // 10)
        print(f"  {cls:25s} {class_counts[cls]:6d}  {bar}")
    print("-" * 50)

    # Check which target classes are missing
    missing = set(TARGET_CLASSES) - set(class_counts.keys())
    if missing:
        print(f"\n⚠  Missing classes ({len(missing)}):")
        for cls in sorted(missing):
            print(f"    - {cls}")

    # Balance by class (optional)
    if max_per_class > 0:
        unique_entries = _balance_by_class(unique_entries, max_per_class)
        print(f"\nAfter balancing (max {max_per_class}/class): {len(unique_entries)}")

    # Copy audio files
    if not skip_audio_copy:
        print(f"\nCopying and resampling audio to {raw_audio_dir}...")
        os.makedirs(raw_audio_dir, exist_ok=True)

        success = 0
        failed = 0
        valid_entries = []

        for i, entry in enumerate(unique_entries):
            source = entry["source_path"]
            dest = os.path.join(raw_audio_dir, entry["filename"])

            if os.path.exists(dest):
                valid_entries.append(entry)
                success += 1
                continue

            if not os.path.exists(source):
                failed += 1
                continue

            if copy_and_resample_audio(source, dest, target_sr):
                valid_entries.append(entry)
                success += 1
            else:
                failed += 1

            if (i + 1) % 500 == 0:
                print(f"  Progress: {i + 1}/{len(unique_entries)} "
                      f"(success: {success}, failed: {failed})")

        print(f"  Audio copy complete: {success} success, {failed} failed")
        unique_entries = valid_entries
    else:
        print("\nSkipping audio copy (--skip-audio-copy)")

    # Write unified labels.csv
    labels_csv = os.path.join(output_dir, "labels.csv")
    with open(labels_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "labels"])
        for entry in unique_entries:
            writer.writerow([entry["filename"], entry["labels"]])

    print(f"\n✓ Unified labels saved: {labels_csv}")
    print(f"  Total clips: {len(unique_entries)}")

    # Final class distribution
    final_counts = Counter()
    for entry in unique_entries:
        for label in entry["labels"].split("|"):
            final_counts[label] += 1

    print(f"  Active classes: {len(final_counts)}/{len(TARGET_CLASSES)}")

    # Save class_map.json (class name → index, sorted alphabetically)
    active_classes = sorted(final_counts.keys())
    class_map = {name: idx for idx, name in enumerate(active_classes)}
    class_map_path = os.path.join(output_dir, "class_map.json")
    with open(class_map_path, "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"  Class map saved: {class_map_path} ({len(class_map)} classes)")

    # Save class statistics
    stats_path = os.path.join(output_dir, "class_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(
            {
                "total_clips": len(unique_entries),
                "num_classes": len(final_counts),
                "class_counts": dict(final_counts.most_common()),
                "missing_classes": sorted(missing) if missing else [],
                "sources": dict(Counter(e["source"] for e in unique_entries if "source" in e)),
            },
            f,
            indent=2,
        )
    print(f"  Statistics saved: {stats_path}")

    return labels_csv


def _balance_by_class(entries: list, max_per_class: int) -> list:
    """
    Limit entries so no single class exceeds max_per_class.

    For multi-label entries, the entry is counted toward all its classes.
    We greedily include entries, skipping if ALL their classes are already at max.
    """
    class_counts = Counter()
    result = []

    # Shuffle for randomness
    import random
    random.seed(42)
    shuffled = entries.copy()
    random.shuffle(shuffled)

    for entry in shuffled:
        labels = entry["labels"].split("|")
        # Include if ANY class still has room
        if any(class_counts[l] < max_per_class for l in labels):
            result.append(entry)
            for l in labels:
                class_counts[l] += 1

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple datasets into unified training data"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Root data directory.",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=0,
        help="Max samples per class (0 = unlimited).",
    )
    parser.add_argument(
        "--skip-audio-copy", action="store_true",
        help="Only merge labels, don't copy audio files.",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Target sample rate for audio files.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    downloads_dir = os.path.join(data_dir, "downloads")
    raw_audio_dir = os.path.join(data_dir, "raw")

    print("=" * 60)
    print("Multi-Dataset Preparation")
    print("=" * 60)

    # Per-dataset label CSVs (produced by individual download scripts)
    dataset_csvs = {
        "ESC-50": os.path.join(downloads_dir, "esc50_labels.csv"),
        "UrbanSound8K": os.path.join(downloads_dir, "urbansound8k_labels.csv"),
        "FSD50K": os.path.join(downloads_dir, "fsd50k_labels.csv"),
    }

    print("\nLooking for dataset labels:")
    for name, path in dataset_csvs.items():
        exists = "✓" if os.path.exists(path) else "✗ (not found)"
        print(f"  {name}: {path} {exists}")

    # Merge
    output_csv = merge_datasets(
        dataset_csvs=dataset_csvs,
        output_dir=data_dir,
        raw_audio_dir=raw_audio_dir,
        max_per_class=args.max_per_class,
        skip_audio_copy=args.skip_audio_copy,
        target_sr=args.sample_rate,
    )

    if output_csv:
        print("\n" + "=" * 60)
        print("DONE! Next steps:")
        print("=" * 60)
        print(f"  1. Extract spectrograms:  python prepare_data.py --skip-spectrograms")
        print(f"     or (with spectrograms): python prepare_data.py")
        print(f"  2. Start training:         python train.py")
        print(f"  3. Resume after restart:   python train.py --resume")


if __name__ == "__main__":
    main()
