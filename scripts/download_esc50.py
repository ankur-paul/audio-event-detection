"""
Download and prepare ESC-50 dataset.

ESC-50: Environmental Sound Classification
- 2000 clips, 50 classes, 5 seconds each @ 44.1kHz
- Direct download from GitHub (no registration needed)
- Size: ~600 MB

Source: https://github.com/karolpiczak/ESC-50

Usage:
    python scripts/download_esc50.py
    python scripts/download_esc50.py --output-dir data/downloads/esc50
"""

import argparse
import csv
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.label_mapping import ESC50_MAPPING, map_labels, get_coverage_report

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
ESC50_ZIP = "ESC-50-master.zip"


def download_esc50(output_dir: str) -> str:
    """
    Download and extract ESC-50 dataset.

    Returns:
        Path to extracted dataset root directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / ESC50_ZIP
    extract_dir = output_dir / "ESC-50-master"

    if extract_dir.exists():
        print(f"ESC-50 already extracted at {extract_dir}")
        return str(extract_dir)

    if not zip_path.exists():
        print(f"Downloading ESC-50 (~600 MB)...")
        print(f"  URL: {ESC50_URL}")
        print(f"  Saving to: {zip_path}")

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r  Progress: {pct:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(ESC50_URL, str(zip_path), reporthook=_progress)
        print("\n  Download complete!")
    else:
        print(f"Using cached zip: {zip_path}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    print(f"Extracted to {extract_dir}")

    return str(extract_dir)


def prepare_esc50_labels(esc50_dir: str, output_csv: str) -> int:
    """
    Read ESC-50 metadata and produce labels in our unified format.

    ESC-50 metadata CSV has columns:
    filename, fold, target, category, esc10, src_file, take

    Args:
        esc50_dir: Path to ESC-50-master directory.
        output_csv: Path to save the output labels CSV.

    Returns:
        Number of clips with matched labels.
    """
    meta_path = os.path.join(esc50_dir, "meta", "esc50.csv")
    audio_dir = os.path.join(esc50_dir, "audio")

    if not os.path.exists(meta_path):
        print(f"ERROR: ESC-50 metadata not found at {meta_path}")
        return 0

    entries = []
    skipped = 0

    with open(meta_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_label = row["category"]
            target_labels = map_labels([source_label], ESC50_MAPPING)

            if not target_labels:
                skipped += 1
                continue

            audio_file = row["filename"]
            audio_path = os.path.join(audio_dir, audio_file)

            if not os.path.exists(audio_path):
                print(f"  Warning: audio file not found: {audio_file}")
                continue

            entries.append({
                "filename": audio_file,
                "labels": "|".join(target_labels),
                "source": "esc50",
                "source_path": audio_path,
            })

    # Write output CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "labels", "source", "source_path"])
        writer.writeheader()
        writer.writerows(entries)

    print(f"\nESC-50 preparation complete:")
    print(f"  Matched clips: {len(entries)}")
    print(f"  Skipped clips: {skipped} (no matching class)")
    print(f"  Output: {output_csv}")

    # Print coverage report
    report = get_coverage_report(ESC50_MAPPING)
    print(f"\n  Coverage: {report['covered_count']}/{report['total']} "
          f"classes ({report['coverage_pct']:.0f}%)")
    print(f"  Missing: {', '.join(report['missing'])}")

    return len(entries)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare ESC-50")
    parser.add_argument(
        "--output-dir", type=str, default="data/downloads/esc50",
        help="Directory to download ESC-50 to.",
    )
    parser.add_argument(
        "--labels-csv", type=str, default="data/downloads/esc50_labels.csv",
        help="Path to save the prepared labels CSV.",
    )
    args = parser.parse_args()

    esc50_dir = download_esc50(args.output_dir)
    prepare_esc50_labels(esc50_dir, args.labels_csv)


if __name__ == "__main__":
    main()
